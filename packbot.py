import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import json, math, os, sys, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import time
from cnn import load_and_resize_heightmap
sys.path.append(os.path.join(os.getcwd(), 'Pointnet_Pointnet2_pytorch', 'models'))
from pointnet2_cls_ssg import get_model as pointnet2_official_model
import imageio.v2 as imageio
from matplotlib.patches import Rectangle

# ==========================================
# LOG
# ==========================================
LOG_FILE = "training_log_cracker_box.txt"

def log_print(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write(f"\n{'='*60}\n🚀 RUN — {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")

# ==========================================
# CONFIG
# ==========================================
KEYWORD          = '003 Cracker Box'
EPOCHS           = 1500
K                = 5
SEED             = 42
L_ENTROPY        = 0.005   # pressão para distribuir pelos experts
L_SIGMA          = 0.05    # forçar sigma pequeno no winner
PATIENCE         = 200     # épocas sem melhoria para early stopping
ONLY_FIRST_STEP  = True    # True → apenas step==0 (bin vazio) → tudo train

# Paleta de cores para os K experts (global, usada nos plots e helper)
colors_k = plt.cm.tab10(np.linspace(0, 1, K))

# ==========================================
# FILTER
# ==========================================
def normalize_angle(angle):
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle

def is_upright(step, tol_deg=20, z_min=0.05, z_max=0.35):
    ax = normalize_angle(step['x_y_angles'][0])
    ay = normalize_angle(step['x_y_angles'][1])
    z  = step['human_action'][2]
    return abs(ax) < tol_deg and abs(ay) < tol_deg and z_min < z < z_max

# ==========================================
# SPLIT
# ==========================================
def split_dataset_scientific(data_list, seed,
                              test_pct=0.15, val_pct_of_remaining=0.20):
    """
    Split por sequência sem data leakage.
    - step==0 → sempre train (bin vazio, mais multimodal)
    - step>0  → dividido por sequência em train/val/test
    - Se não houver step>0 (ex: ONLY_FIRST_STEP=True), val e test ficam vazios.
    """
    first_steps = [s for s in data_list if s['step'] == 0]
    rest        = [s for s in data_list if s['step'] != 0]

    episodes = defaultdict(list)
    for s in rest:
        episodes[s['sequence_name']].append(s)
    ep_list = list(episodes.values())

    if not ep_list:
        return first_steps, [], []

    random.seed(seed)
    random.shuffle(ep_list)

    test_idx  = int(len(ep_list) * test_pct)
    remaining = ep_list[test_idx:]
    val_idx   = int(len(remaining) * val_pct_of_remaining)

    train = first_steps + [s for ep in remaining[val_idx:] for s in ep]
    val   = [s for ep in remaining[:val_idx]  for s in ep]
    test  = [s for ep in ep_list[:test_idx]   for s in ep]
    return train, val, test

# ==========================================
# METRICS
# ==========================================
def compute_responsibility_dist(pi, sig, mu, target, scales):
    """
    Distância ponderada pela responsabilidade posterior de cada expert.

    r_k = softmax( log π_k + Σ_d log N(GT_d | μ_kd, σ_kd) )

    Para cada amostra, cada expert recebe peso proporcional a quão bem
    explica o GT — usa tanto π como σ. Sem hard assignment.

    Robusto para:
      - unimodal: expert dominante recebe r≈1 → comporta-se como argmax
      - multimodal: cada GT encontra o expert mais relevante automaticamente

    Args:
        pi, sig, mu : saídas do MDN  [B, K] / [B, K, D] / [B, K, D]
        target      : GT normalizado [B, D]
        scales      : tensor [3] para converter xyz para metros

    Returns:
        dist_per_sample: [B] distância em metros
    """
    normal    = torch.distributions.Normal(mu, sig)
    log_prob  = normal.log_prob(target.unsqueeze(1).expand_as(mu)).sum(dim=2)  # [B, K]
    log_joint = torch.log(pi + 1e-7) + log_prob                               # [B, K]
    r         = torch.softmax(log_joint, dim=1)                                # [B, K]

    mu_xyz = mu[:, :, :3] * scales                                             # [B, K, 3]
    gt_xyz = (target[:, :3] * scales).unsqueeze(1)                             # [B, 1, 3]
    dists  = torch.norm(mu_xyz - gt_xyz, dim=2)                                # [B, K]

    return (r * dists).sum(dim=1)                                              # [B] metros

# ==========================================
# GUMBEL SAMPLING
# ==========================================
def gumbel_sample(pi_np):
    z = np.random.gumbel(loc=0, scale=1, size=pi_np.shape)
    return (np.log(pi_np + 1e-8) + z).argmax(axis=1)

def gumbel_sample_and_predict(pi, sig, mu):
    if isinstance(pi, torch.Tensor):
        pi_np  = pi.detach().cpu().numpy()
        sig_np = sig.detach().cpu().numpy()
        mu_np  = mu.detach().cpu().numpy()
    else:
        pi_np, sig_np, mu_np = pi, sig, mu

    k     = gumbel_sample(pi_np)
    B     = mu_np.shape[0]
    mu_k  = mu_np[np.arange(B), k, :]
    sig_k = sig_np[np.arange(B), k, :]
    return np.random.randn(*mu_k.shape) * sig_k + mu_k, k

# ==========================================
# LOSS
# ==========================================
def mdn_loss_packing(pi, sigma, mu, target,
                     l_sep=0.0, l_sigma=0.0, l_entropy=0.0):
    # NLL
    dist      = torch.distributions.Normal(mu, sigma)
    log_probs = dist.log_prob(target.unsqueeze(1).expand_as(mu)).sum(dim=2)
    log_pi    = torch.log(pi + 1e-7)
    log_joint = log_probs + log_pi
    nll       = -torch.logsumexp(log_joint, dim=1).mean()

    # Separation loss — margem adaptativa: K=5 → 0.08
    mu_xyz   = mu[:, :, :3]
    dist_sq  = ((mu_xyz.unsqueeze(2) - mu_xyz.unsqueeze(1)) ** 2).sum(dim=-1)
    K_       = mu.shape[1]
    mask     = 1 - torch.eye(K_, device=mu.device).unsqueeze(0)
    margin   = 2.0 / (K_ ** 2)
    sep_loss = (torch.clamp(margin - dist_sq, min=0) * mask).sum(dim=(1, 2)).mean()

    # Sigma winner — penaliza sigma grande do expert dominante
    winner       = log_joint.argmax(dim=1)
    mask_w       = torch.zeros(pi.shape[0], K_, device=mu.device)
    mask_w.scatter_(1, winner.unsqueeze(1), 1.0)
    sigma_winner = (mask_w.unsqueeze(-1) * sigma).sum(dim=1).mean()

    # Entropy — incentiva distribuição entre experts
    entropy = -(pi * torch.log(pi + 1e-8)).sum(dim=1).mean()

    loss = nll + l_sep * sep_loss + l_sigma * sigma_winner - l_entropy * entropy
    return loss, nll.item(), sep_loss.item(), sigma_winner.item(), entropy.item()

# ==========================================
# MODEL
# ==========================================
class MDNLayer(nn.Module):
    def __init__(self, input_dim, action_dim, k=1):
        super().__init__()
        self.k          = k
        self.action_dim = action_dim
        self.pi_logits  = nn.Linear(input_dim, k)
        self.sigma_layer = nn.Linear(input_dim, k * action_dim)
        self.mu          = nn.Linear(input_dim, k * action_dim)
        with torch.no_grad():
            xs = torch.linspace(0.1, 0.9, k)
            ys = torch.linspace(0.1, 0.9, k)
            for i in range(k):
                base = i * action_dim
                self.mu.bias.data[base]     = xs[i]
                self.mu.bias.data[base + 1] = ys[i]
                self.mu.bias.data[base + 2] = 0.2
                self.mu.bias.data[base + 3] = 0.0
                self.mu.bias.data[base + 4] = 1.0
            nn.init.constant_(self.pi_logits.bias, 0.0)
            nn.init.constant_(self.pi_logits.weight, 0.0)

    def forward(self, x, gate_temp=1.0):
        sigma = (F.softplus(self.sigma_layer(x)) + 0.02).view(-1, self.k, self.action_dim)
        mu    = self.mu(x).view(-1, self.k, self.action_dim)
        pi    = F.softmax(self.pi_logits(x) / gate_temp, dim=1)
        return pi, sigma, mu


class BCPackingPolicyMDN(nn.Module):
    def __init__(self, k=1):
        super().__init__()
        self.heightmap_encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in self.heightmap_encoder.parameters():
            param.requires_grad = False
        log_print("❄️  ResNet congelada.")
        self.heightmap_encoder.fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(512, 256)
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
        )
        self.pointcloud_encoder = PointNetEncoder()
        self.fusion_dropout     = nn.Dropout(0.0)
        self.mdn                = MDNLayer(input_dim=512, action_dim=5, k=k)

    def forward(self, h, p, gate_temp=1.0):
        h_f      = self.heightmap_encoder(h.repeat(1, 3, 1, 1))
        p_f      = self.pointcloud_encoder(p)
        combined = torch.cat((h_f, p_f), dim=1)
        combined = self.fusion_dropout(combined)
        combined = self.fusion_mlp(combined)
        return self.mdn(combined, gate_temp)


class PointNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        pretrained_path = 'Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'
        self.base_model = pointnet2_official_model(num_class=40, normal_channel=False)
        if os.path.exists(pretrained_path):
            log_print(f"🔄 A carregar pesos PointNet++: {pretrained_path}")
            ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            self.base_model.load_state_dict(ckpt['model_state_dict'])
            log_print("✅ Pesos PointNet++ carregados!")
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()
        log_print("❄️  PointNet++ congelada.")

    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(self, x):
        x = x.transpose(1, 2)
        l1_xyz, l1_points = self.base_model.sa1(x, None)
        l2_xyz, l2_points = self.base_model.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.base_model.sa3(l2_xyz, l2_points)
        feat = l3_points.view(-1, 1024)
        feat = F.relu(self.base_model.bn1(self.base_model.fc1(feat)))
        feat = F.relu(self.base_model.bn2(self.base_model.fc2(feat)))
        return feat

# ==========================================
# DATASET
# ==========================================
class PackingDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        self.pc_cache, self.hm_cache = {}, {}
        for step in self.data:
            p_path = step['obj_path'].replace('objects/', 'objects_npy/').replace('.obj', '.npy')
            h_path = step['heightmap_path']
            if p_path not in self.pc_cache:
                self.pc_cache[p_path] = np.load(p_path)
            if h_path not in self.hm_cache:
                self.hm_cache[h_path] = load_and_resize_heightmap(h_path).squeeze(0)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        step   = self.data[idx]
        h      = self.hm_cache[step['heightmap_path']].clone()
        p_path = step['obj_path'].replace('objects/', 'objects_npy/').replace('.obj', '.npy')
        p      = torch.tensor(self.pc_cache[p_path], dtype=torch.float32)
        x, y, z, yaw = step['human_action']
        y_rad  = math.radians(yaw)
        target = torch.tensor([
            x / 0.345987, y / 0.227554, z / 0.565,
            math.sin(y_rad), math.cos(y_rad)
        ], dtype=torch.float32)
        return h, p, target

# ==========================================
# HELPERS DE VISUALIZAÇÃO
# ==========================================
def get_expert_distribution(step, dataset, model, device):
    h = dataset.hm_cache[step['heightmap_path']].clone().unsqueeze(0).to(device)
    p_path = step['obj_path'].replace('objects/', 'objects_npy/').replace('.obj', '.npy')
    p = torch.tensor(dataset.pc_cache[p_path], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pi, sig, mu = model(h, p, gate_temp=0.3)
    return pi[0].cpu().numpy(), sig[0].cpu().numpy(), mu[0].cpu().numpy()

def plot_frame(pi_vals, sig_vals, mu_vals, step, frame_id, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    gx, gy, gz, _ = step['human_action']
    ax.scatter(gx, gy, c='black', marker='+', s=80, label='GT')
    for i in range(K):
        mx = mu_vals[i, 0] * 0.345987
        my = mu_vals[i, 1] * 0.227554
        sx = sig_vals[i, 0] * 0.345987
        sy = sig_vals[i, 1] * 0.227554
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(mx + sx * np.cos(theta), my + sy * np.sin(theta),
                color=colors_k[i], alpha=0.5)
        ax.scatter(mx, my, color=colors_k[i], s=100, marker='*',
                   label=f"E{i+1} ({pi_vals[i]*100:.0f}%)")
    th = 0.005
    ax.add_patch(Rectangle((-th/2, -th/2), 0.346+th, 0.2275+th,
                            linewidth=3, edgecolor='red', facecolor='none'))
    ax.set_xlim(-0.05, 0.40); ax.set_ylim(-0.025, 0.30)
    ax.set_title(f"Frame {frame_id} | π={np.round(pi_vals, 2)}")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    path = os.path.join(save_path, f"frame_{frame_id}.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path

# ==========================================
# MAIN — DADOS
# ==========================================
with open("dataset_packbot.json") as f:
    data = json.load(f)

filtered = [s for s in data
            if KEYWORD in s['obj_path']
            and is_upright(s)
            and (not ONLY_FIRST_STEP or s['step'] == 0)]
seqs     = set(s['sequence_name'] for s in filtered)
log_print(f"\n📦 {KEYWORD} — upright only")
log_print(f"   Total steps : {len(filtered)}")
log_print(f"   Sequências  : {len(seqs)}")

angles = np.array([[normalize_angle(s['x_y_angles'][0]),
                    normalize_angle(s['x_y_angles'][1])] for s in filtered])
zs     = np.array([s['human_action'][2] for s in filtered])
log_print(f"   Ângulo X: [{angles[:,0].min():.1f}°, {angles[:,0].max():.1f}°]")
log_print(f"   Ângulo Y: [{angles[:,1].min():.1f}°, {angles[:,1].max():.1f}°]")
log_print(f"   Z: [{zs.min():.3f}, {zs.max():.3f}]m")

train_d, val_d, test_d = split_dataset_scientific(filtered, SEED)
n0_tr = sum(1 for s in train_d if s['step'] == 0)
n0_v  = sum(1 for s in val_d   if s['step'] == 0)
n0_te = sum(1 for s in test_d  if s['step'] == 0)
log_print(f"   Train: {len(train_d)} (step=0: {n0_tr}) | "
          f"Val: {len(val_d)} (step=0: {n0_v}) | "
          f"Test: {len(test_d)} (step=0: {n0_te})")

if val_d or test_d:
    log_print(f"   ℹ️  Todos os step=0 → train garantido")
    if val_d:
        val_seqs = sorted(set(s['sequence_name'] for s in val_d))
        log_print(f"   Val  sequências ({len(val_seqs)}):")
        for sq in val_seqs:
            steps_in = [s['step'] for s in val_d if s['sequence_name'] == sq]
            log_print(f"      {sq}  (steps: {sorted(steps_in)})")
    if test_d:
        test_seqs = sorted(set(s['sequence_name'] for s in test_d))
        log_print(f"   Test sequências ({len(test_seqs)}):")
        for sq in test_seqs:
            steps_in = [s['step'] for s in test_d if s['sequence_name'] == sq]
            log_print(f"      {sq}  (steps: {sorted(steps_in)})")
else:
    log_print(f"   ⚠️  Sem step>0 suficientes — modo ALL-TRAIN (val=0, test=0)")

loader_t    = DataLoader(PackingDataset(train_d), batch_size=32, shuffle=True)
loader_v    = DataLoader(PackingDataset(val_d),   batch_size=32)
loader_test = DataLoader(PackingDataset(test_d),  batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scales = torch.tensor([0.345987, 0.227554, 0.565], device=device)

model     = BCPackingPolicyMDN(k=K).to(device)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-4, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

total = sum(p.numel() for p in model.parameters() if p.requires_grad)
log_print(f"   Trainable params: {total:,}")
log_print(f"   Device: {device}")

HAS_VAL  = len(val_d)  > 0
HAS_TEST = len(test_d) > 0

# ==========================================
# MAIN — TREINO
# ==========================================
# Melhor modelo guardado por Responsibility Dist (robusto para uni+multimodal)
best_resp_dist = float('inf')
best_weights   = None
_es_counter    = 0
_es_best       = float('inf')
_stopped_epoch = EPOCHS

history = {
    'train': [], 'val': [], 'test': [],
    'resp_dist': [], 'epochs': [], 'entropy': []
}

start = time.time()
for epoch in range(EPOCHS):
    progress  = epoch / EPOCHS
    gate_temp = max(0.3, 1.0 - 0.7 * progress)        # 1.0 → 0.3
    l_sep     = 0.05  * min(1.0, progress / 0.4)       # curriculum separação
    l_sigma   = L_SIGMA   * min(1.0, progress / 0.3)   # curriculum sigma
    l_ent     = L_ENTROPY                               # constante

    # ── Train step ──
    model.train()
    t_l = 0.0
    for h, p, a in loader_t:
        h, p, a = h.to(device), p.to(device), a.to(device)
        optimizer.zero_grad()
        pi, sig, mu = model(h, p, gate_temp=gate_temp)
        loss, *_ = mdn_loss_packing(pi, sig, mu, a,
                                    l_sep=l_sep, l_sigma=l_sigma, l_entropy=l_ent)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_l += loss.item()
    scheduler.step()

    # ── Val/Test NLL (só para monitorizar) ──
    model.eval()
    v_l = te_l = 0.0
    with torch.no_grad():
        if HAS_VAL:
            for h, p, a in loader_v:
                h, p, a = h.to(device), p.to(device), a.to(device)
                pi, sig, mu = model(h, p, gate_temp=0.3)
                vl, *_ = mdn_loss_packing(pi, sig, mu, a)
                v_l += vl.item()
        if HAS_TEST:
            for h, p, a in loader_test:
                h, p, a = h.to(device), p.to(device), a.to(device)
                pi, sig, mu = model(h, p, gate_temp=0.3)
                tl, *_ = mdn_loss_packing(pi, sig, mu, a)
                te_l += tl.item()

    avg_t  = t_l / len(loader_t)
    avg_v  = v_l  / len(loader_v)    if HAS_VAL  else float('nan')
    avg_te = te_l / len(loader_test) if HAS_TEST else float('nan')
    history['train'].append(avg_t)
    if HAS_VAL:  history['val'].append(avg_v)
    if HAS_TEST: history['test'].append(avg_te)

    # ── Avaliação a cada 10 épocas ──
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            # π e entropia no train
            pi_avg, ent_vals = [], []
            for ht, pt, at in loader_t:
                pi_t, _, _ = model(ht.to(device), pt.to(device), gate_temp=0.3)
                pi_avg.append(pi_t.mean(dim=0).cpu().numpy())
                ent_vals.append(
                    -(pi_t * torch.log(pi_t + 1e-8)).sum(dim=1).mean().item()
                )
            pi_mean     = np.mean(pi_avg, axis=0)
            avg_entropy = float(np.mean(ent_vals))

            # Responsibility Dist — mediana (robusta a outliers) + P95
            ref_loader = loader_v if HAS_VAL else loader_t
            ref_label  = "Val"    if HAS_VAL else "Train"
            all_rd = []
            for h, p, a in ref_loader:
                h, p, a = h.to(device), p.to(device), a.to(device)
                pi, sig, mu = model(h, p, gate_temp=0.3)
                rd = compute_responsibility_dist(pi, sig, mu, a, scales)
                all_rd.extend((rd * 100).cpu().numpy().tolist())   # cm
            all_rd        = np.array(all_rd)
            median_dist   = float(np.median(all_rd))
            p95_dist      = float(np.percentile(all_rd, 95))

            # Guardar melhor modelo por mediana (robusto a outliers)
            if median_dist < best_resp_dist:
                best_resp_dist = median_dist
                best_weights   = {k: v.cpu() for k, v in model.state_dict().items()}

            history['resp_dist'].append(median_dist)
            history['epochs'].append(epoch + 1)
            history['entropy'].append(avg_entropy)

            # Early stopping
            if median_dist < _es_best:
                _es_best    = median_dist
                _es_counter = 0
            else:
                _es_counter += 1

            pi_str   = " ".join(f"E{i+1}:{pi_mean[i]*100:.0f}%" for i in range(K))
            val_str  = f"{avg_v:.4f}"  if HAS_VAL  else "  n/a "
            test_str = f"{avg_te:.4f}" if HAS_TEST else "  n/a "
            es_str   = f"ES:{_es_counter}/{PATIENCE//10}"

            log_print(
                f"Epoch [{epoch+1:>4}/{EPOCHS}] "
                f"Train:{avg_t:.4f} Val:{val_str} Test:{test_str} | "
                f"{ref_label} Median:{median_dist:.2f}cm P95:{p95_dist:.2f}cm Best:{best_resp_dist:.2f}cm | "
                f"Ent:{avg_entropy:.3f} | {pi_str} | σ(X,Y) {sig_str} | "
                f"temp:{gate_temp:.2f} lr:{optimizer.param_groups[0]['lr']:.6f} | {es_str}"
            )

            if _es_counter >= PATIENCE // 10:
                _stopped_epoch = epoch + 1
                log_print(f"\n⏹️  Early stopping na época {_stopped_epoch} "
                          f"(sem melhoria há {PATIENCE} épocas)")
                break

# ==========================================
# GUARDAR MELHOR MODELO
# ==========================================
log_print("\n" + "=" * 60)
if _stopped_epoch < EPOCHS:
    log_print(f"⏹️  Parou na época {_stopped_epoch}/{EPOCHS} (early stopping)")
else:
    log_print(f"✅  Treino completo ({EPOCHS} épocas)")

model.load_state_dict(best_weights)
model_path = f"model_cracker_box_k{K}_seed{SEED}.pth"
torch.save(best_weights, model_path)
log_print(f"💾 Modelo guardado: {model_path}")
log_print(f"   Melhor {('Val' if HAS_VAL else 'Train')} Mediana RespDist: {best_resp_dist:.2f}cm")

# ==========================================
# AVALIAÇÃO FINAL — 3 MÉTRICAS
# ==========================================
model.eval()
ref_loader_final = loader_test if HAS_TEST else loader_t
ref_label_final  = "Test"      if HAS_TEST else "Train"

with torch.no_grad():
    all_rd_final = []
    all_argmax   = []
    all_gumbel   = []
    for h, p, a in ref_loader_final:
        h, p, a = h.to(device), p.to(device), a.to(device)
        pi, sig, mu = model(h, p, gate_temp=0.3)

        # Responsibility por amostra
        rd = compute_responsibility_dist(pi, sig, mu, a, scales)
        all_rd_final.extend((rd * 100).cpu().numpy().tolist())

        # Argmax μ (determinístico, sem ruído de σ)
        winner    = pi.argmax(dim=1)
        batch_idx = torch.arange(mu.size(0), device=device)
        mu_w      = mu[batch_idx, winner]
        d_am      = torch.norm(mu_w[:, :3] * scales - a[:, :3] * scales, dim=1)
        all_argmax.extend((d_am * 100).cpu().numpy().tolist())

        # Argmax μ com clamp à caixa (deployment seguro)
        sampled, _ = gumbel_sample_and_predict(pi, sig, mu)
        mu_s        = torch.tensor(sampled[:, :3], dtype=torch.float32, device=device)
        d_gb        = torch.norm(mu_s * scales - a[:, :3] * scales, dim=1)
        all_gumbel.extend((d_gb * 100).cpu().numpy().tolist())

all_rd_final = np.array(all_rd_final)
all_argmax   = np.array(all_argmax)
all_gumbel   = np.array(all_gumbel)

log_print(f"\n🏆 {ref_label_final} Dist Error (melhor modelo — mediana/P95):")
log_print(f"   Responsibility  | Mediana:{np.median(all_rd_final):.2f}cm  "
          f"Média:{all_rd_final.mean():.2f}cm  P95:{np.percentile(all_rd_final,95):.2f}cm  ← principal")
log_print(f"   Argmax μ        | Mediana:{np.median(all_argmax):.2f}cm  "
          f"Média:{all_argmax.mean():.2f}cm  P95:{np.percentile(all_argmax,95):.2f}cm")
log_print(f"   Gumbel sample   | Mediana:{np.median(all_gumbel):.2f}cm  "
          f"Média:{all_gumbel.mean():.2f}cm  P95:{np.percentile(all_gumbel,95):.2f}cm")
log_print(f"   Ruído σ (Gumbel−Argmax mediana): "
          f"{np.median(all_gumbel)-np.median(all_argmax):.2f}cm")
log_print(f"⏱️  {(time.time()-start)/60:.2f} min")

# ==========================================
# DIAGNÓSTICO DETALHADO POR STEP
# ==========================================
eval_d       = test_d if HAS_TEST else train_d
eval_label   = "TEST" if HAS_TEST else "TRAIN (sem test set)"
eval_dataset = PackingDataset(eval_d)

log_print(f"\n{'#':>4} | {'Seq':<35} | {'Expert':>6} | "
          f"{'PX':>7} {'PY':>7} {'PZ':>7} | "
          f"{'GTX':>7} {'GTY':>7} {'GTZ':>7} | {'Argmax':>8} {'RespD':>7}")
log_print(f"── {eval_label} ──")
log_print("-" * 120)

all_errors_argmax = []
all_errors_resp   = []
all_preds         = []
all_gts           = []
all_experts       = []
all_pi_vals       = []

model.eval()
with torch.no_grad():
    for idx, step in enumerate(eval_d):
        h = eval_dataset.hm_cache[step['heightmap_path']].clone().unsqueeze(0).to(device)
        p_path = step['obj_path'].replace('objects/', 'objects_npy/').replace('.obj', '.npy')
        p = torch.tensor(eval_dataset.pc_cache[p_path],
                         dtype=torch.float32).unsqueeze(0).to(device)

        pi_s, sig_s, mu_s = model(h, p, gate_temp=0.3)
        pi_v = pi_s[0].cpu().numpy()

        # GT como tensor normalizado
        gx, gy, gz, gyaw = step['human_action']
        y_rad    = math.radians(gyaw)
        target_t = torch.tensor([[
            gx/0.345987, gy/0.227554, gz/0.565,
            math.sin(y_rad), math.cos(y_rad)
        ]], dtype=torch.float32, device=device)

        # Argmax
        winner    = pi_s.argmax(dim=1)[0].item()
        mu_w      = mu_s[0, winner].cpu().numpy()
        px = mu_w[0]*0.345987; py = mu_w[1]*0.227554; pz = mu_w[2]*0.565
        dist_argmax = math.sqrt((px-gx)**2 + (py-gy)**2 + (pz-gz)**2) * 100

        # Responsibility
        rd        = compute_responsibility_dist(pi_s, sig_s, mu_s, target_t, scales)
        dist_resp = rd[0].item() * 100

        all_errors_argmax.append(dist_argmax)
        all_errors_resp.append(dist_resp)
        all_preds.append([px, py, pz])
        all_gts.append([gx, gy, gz])
        all_experts.append(winner)
        all_pi_vals.append(pi_v)

        seq_short = step['sequence_name'][:35]
        log_print(f"{idx+1:>4} | {seq_short:<35} | E{winner+1:>5} | "
                  f"{px:>7.3f} {py:>7.3f} {pz:>7.3f} | "
                  f"{gx:>7.3f} {gy:>7.3f} {gz:>7.3f} | "
                  f"{dist_argmax:>7.2f}cm {dist_resp:>6.2f}cm")

log_print("-" * 120)
log_print(f"{'Argmax':>8}  Avg:{np.mean(all_errors_argmax):.2f}cm | "
          f"Min:{np.min(all_errors_argmax):.2f}cm | "
          f"Max:{np.max(all_errors_argmax):.2f}cm | "
          f"Std:{np.std(all_errors_argmax):.2f}cm")
log_print(f"{'RespDist':>8}  Avg:{np.mean(all_errors_resp):.2f}cm | "
          f"Min:{np.min(all_errors_resp):.2f}cm | "
          f"Max:{np.max(all_errors_resp):.2f}cm | "
          f"Std:{np.std(all_errors_resp):.2f}cm")

all_preds   = np.array(all_preds)
all_gts     = np.array(all_gts)
all_pi_vals = np.array(all_pi_vals)

# Erro por dimensão
ex = np.abs(all_preds[:, 0] - all_gts[:, 0]) * 100
ey = np.abs(all_preds[:, 1] - all_gts[:, 1]) * 100
ez = np.abs(all_preds[:, 2] - all_gts[:, 2]) * 100
log_print(f"\n{'='*50}")
log_print("ERRO POR DIMENSÃO (argmax μ)")
log_print(f"{'='*50}")
log_print(f"X: {ex.mean():.2f}cm ± {ex.std():.2f}cm")
log_print(f"Y: {ey.mean():.2f}cm ± {ey.std():.2f}cm")
log_print(f"Z: {ez.mean():.2f}cm ± {ez.std():.2f}cm")

# ==========================================
# TABELA DE EXPERTS
# ==========================================
ref_step  = eval_d[0]
h_ref     = eval_dataset.hm_cache[ref_step['heightmap_path']].clone().unsqueeze(0).to(device)
p_path_r  = ref_step['obj_path'].replace('objects/', 'objects_npy/').replace('.obj', '.npy')
p_ref     = torch.tensor(eval_dataset.pc_cache[p_path_r],
                          dtype=torch.float32).unsqueeze(0).to(device)
with torch.no_grad():
    pi_ref, sig_ref, mu_ref = model(h_ref, p_ref, gate_temp=0.3)
pi_vals_ref  = pi_ref[0].cpu().numpy()
mu_vals_ref  = mu_ref[0].cpu().numpy()
sig_vals_ref = sig_ref[0].cpu().numpy()

log_print(f"\n{'='*70}")
log_print("EXPERTS — μ e σ aprendidos (sample ref)")
log_print(f"{'='*70}")
log_print(f"{'Expert':>8} | {'π':>7} | "
          f"{'μ_X':>8} {'μ_Y':>8} {'μ_Z':>8} | "
          f"{'σ_X':>8} {'σ_Y':>8} {'σ_Z':>8}")
log_print("-" * 75)
for i in range(K):
    mx = mu_vals_ref[i,0]*0.345987;  my = mu_vals_ref[i,1]*0.227554;  mz = mu_vals_ref[i,2]*0.565
    sx = sig_vals_ref[i,0]*0.345987; sy = sig_vals_ref[i,1]*0.227554; sz = sig_vals_ref[i,2]*0.565
    marker = " ← winner" if i == np.argmax(pi_vals_ref) else ""
    log_print(f"Expert {i+1:>1} | {pi_vals_ref[i]*100:>6.1f}% | "
              f"{mx:>8.3f} {my:>8.3f} {mz:>8.3f} | "
              f"{sx:>8.3f} {sy:>8.3f} {sz:>8.3f}{marker}")

# ==========================================
# FEATURE DISTANCES
# ==========================================
log_print(f"\n{'='*50}")
log_print("DIAGNÓSTICO — Feature distances (heightmaps)")
log_print(f"{'='*50}")
feats = []
with torch.no_grad():
    for step in eval_d:
        h = eval_dataset.hm_cache[step['heightmap_path']].clone().unsqueeze(0).to(device)
        feats.append(model.heightmap_encoder(h.repeat(1, 3, 1, 1)).cpu())
feats      = torch.cat(feats, dim=0)
dists_feat = [(feats[i]-feats[j]).norm().item()
              for i in range(len(feats)) for j in range(i+1, len(feats))]
log_print(f"Distância média: {np.mean(dists_feat):.4f}")
log_print(f"Distância mín:   {np.min(dists_feat):.4f}")
log_print(f"Distância máx:   {np.max(dists_feat):.4f}")
if np.mean(dists_feat) < 1.0:
    log_print("⚠️  Features idênticas — esperado para step=0 (bin vazio sempre igual)")
else:
    log_print("✅  Features discriminativas — modelo usa o heightmap")

# ==========================================
# PLOTS — TRAINING CURVES
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(history['train'], label='Train', color='C0', alpha=0.7)
if history['val']:  axes[0].plot(history['val'],  label='Val',  color='C1')
if history['test']: axes[0].plot(history['test'], label='Test', color='green')
axes[0].set_title(f'{KEYWORD} | Best RespDist: {best_resp_dist:.2f}cm')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MDN Loss')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history['epochs'], history['resp_dist'],
             color='C3', linewidth=2, label='RespDist')
axes[1].axhline(best_resp_dist, color='gray', linestyle='--',
                label=f'Best: {best_resp_dist:.2f}cm')
ax_ent = axes[1].twinx()
if history['entropy']:
    ax_ent.plot(history['epochs'], history['entropy'],
                color='C5', linewidth=1.5, linestyle='--', alpha=0.7, label='Entropy')
    ax_ent.axhline(math.log(K), color='C5', linestyle=':', alpha=0.4,
                   label=f'Max log{K}={math.log(K):.2f}')
    ax_ent.set_ylabel('Entropy', color='C5')
    ax_ent.tick_params(axis='y', labelcolor='C5')
    ax_ent.legend(loc='upper right', fontsize=8)
axes[1].set_title('Responsibility Dist + Entropy')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('cm')
axes[1].legend(loc='upper left', fontsize=8); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_cracker_box.png', bbox_inches='tight')
plt.show()

# ==========================================
# PLOTS — DIAGNÓSTICOS COMPLETOS (3×3)
# ==========================================
fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

# 1. Scatter argmax pred vs GT (X,Y) — cor=expert
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(all_gts[:,0], all_gts[:,1],
            c='black', marker='+', s=80, alpha=0.5, label='GT', zorder=2)
for i in range(len(all_preds)):
    ax1.scatter(all_preds[i,0], all_preds[i,1],
                c=[colors_k[all_experts[i]]], s=60, alpha=0.8, zorder=3)
ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
ax1.set_title('Pred (argmax) vs GT — cor=expert')
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# 2. Expert μ + 1σ elipse
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(all_gts[:,0], all_gts[:,1],
            c='black', marker='+', s=60, alpha=0.3, label='GT', zorder=1)
for i in range(K):
    mx = mu_vals_ref[i,0]*0.345987; my = mu_vals_ref[i,1]*0.227554
    sx = sig_vals_ref[i,0]*0.345987; sy = sig_vals_ref[i,1]*0.227554
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(mx+sx*np.cos(theta), my+sy*np.sin(theta),
             color=colors_k[i], alpha=0.6, linewidth=1.5)
    ax2.scatter(mx, my, c=[colors_k[i]], s=200, marker='*', zorder=4,
                label=f'E{i+1} π={pi_vals_ref[i]*100:.0f}%')
ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
ax2.set_title('Expert μ + elipse 1σ')
ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

# 3. Erro por sample (bar) — argmax
ax3 = fig.add_subplot(gs[0, 2])
bar_colors = [colors_k[e] for e in all_experts]
ax3.bar(range(1, len(all_errors_argmax)+1), all_errors_argmax,
        color=bar_colors, alpha=0.8)
ax3.axhline(np.mean(all_errors_argmax), color='red', linestyle='--',
            label=f'Avg {np.mean(all_errors_argmax):.1f}cm')
ax3.set_xlabel('Sample'); ax3.set_ylabel('Dist (cm)')
ax3.set_title('Erro por sample (argmax) — cor=expert')
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis='y')

# 4. Boxplot erro por dimensão
ax4 = fig.add_subplot(gs[1, 0])
bp = ax4.boxplot([ex, ey, ez], labels=['X', 'Y', 'Z'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['C0', 'C1', 'C2']):
    patch.set_facecolor(color); patch.set_alpha(0.6)
ax4.set_ylabel('Erro (cm)'); ax4.set_title('Erro por dimensão (argmax)')
ax4.grid(True, alpha=0.3, axis='y')

# 5. Pi por expert
ax5 = fig.add_subplot(gs[1, 1])
bars = ax5.bar([f'E{i+1}' for i in range(K)], pi_vals_ref*100,
               color=colors_k, alpha=0.85)
ax5.axhline(100/K, color='gray', linestyle='--', label=f'Uniform {100/K:.0f}%')
ax5.set_ylabel('π (%)'); ax5.set_title('Pi por expert (sample ref)')
ax5.set_ylim(0, 105); ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, pi_vals_ref):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f'{val*100:.1f}%', ha='center', fontsize=8)

# 6. Loss curves
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(history['train'], label='Train', color='C0', alpha=0.7)
if history['val']:  ax6.plot(history['val'],  label='Val',  color='C1')
if history['test']: ax6.plot(history['test'], label='Test', color='green')
ax6.set_xlabel('Epoch'); ax6.set_ylabel('Loss')
ax6.set_title('Loss curves'); ax6.legend(); ax6.grid(True, alpha=0.3)

# 7. RespDist ao longo do treino + Entropy
ax7 = fig.add_subplot(gs[2, 0])
ax7.plot(history['epochs'], history['resp_dist'],
         color='C3', linewidth=2, label='RespDist')
ax7.axhline(best_resp_dist, color='gray', linestyle='--',
            label=f'Best: {best_resp_dist:.2f}cm')
ax7.set_xlabel('Epoch'); ax7.set_ylabel('cm')
ax7.set_title('Responsibility Dist (seleção modelo)')
ax7.legend(fontsize=8); ax7.grid(True, alpha=0.3)
ax7b = ax7.twinx()
if history['entropy']:
    ax7b.plot(history['epochs'], history['entropy'],
              color='C5', linewidth=1.5, linestyle='--', alpha=0.7, label='Entropy')
    ax7b.axhline(math.log(K), color='C5', linestyle=':', alpha=0.4)
    ax7b.set_ylabel('Entropy', color='C5')
    ax7b.tick_params(axis='y', labelcolor='C5')

# 8. Pred vs GT (scatter diagonal)
ax8 = fig.add_subplot(gs[2, 1])
ax8.scatter(all_gts[:,0]*100, all_preds[:,0]*100, alpha=0.7, label='X', s=40)
ax8.scatter(all_gts[:,1]*100, all_preds[:,1]*100, alpha=0.7, label='Y', s=40, marker='s')
lims = [0, max(all_gts[:,0].max(), all_gts[:,1].max())*100 + 2]
ax8.plot(lims, lims, 'k--', alpha=0.5, label='Perfeito')
ax8.set_xlabel('GT (cm)'); ax8.set_ylabel('Pred (cm)')
ax8.set_title('Pred vs GT por dimensão')
ax8.legend(fontsize=8); ax8.grid(True, alpha=0.3)

# 9. Feature distances
ax9 = fig.add_subplot(gs[2, 2])
ax9.hist(dists_feat, bins=20, color='C4', alpha=0.7, edgecolor='black')
ax9.axvline(np.mean(dists_feat), color='red', linestyle='--',
            label=f'Avg: {np.mean(dists_feat):.2f}')
ax9.set_xlabel('Distância L2'); ax9.set_ylabel('Count')
ax9.set_title('Feature distances (heightmaps)')
ax9.legend(); ax9.grid(True, alpha=0.3, axis='y')

plt.suptitle(
    f'{KEYWORD} — k={K} | '
    f'RespDist: {np.mean(all_errors_resp):.2f}cm | '
    f'Argmax: {np.mean(all_errors_argmax):.2f}cm',
    fontsize=13
)
plt.savefig('diagnostics_full.png', bbox_inches='tight', dpi=150)
plt.show()
log_print("✅ Plot guardado: diagnostics_full.png")

# ==========================================
# PLOT — EXPERT OVERVIEW (4 samples)
# ==========================================
n_samples = min(4, len(eval_d))
indices   = np.linspace(0, len(eval_d)-1, n_samples, dtype=int)
samples   = [eval_d[i] for i in indices]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for idx, (ax, step) in enumerate(zip(axes, samples)):
    pi_v, sig_v, mu_v = get_expert_distribution(step, eval_dataset, model, device)
    gx, gy, gz, _ = step['human_action']
    ax.scatter(gx, gy, c='black', marker='+', s=80, label='GT')
    for i in range(K):
        mx = mu_v[i,0]*0.345987; my = mu_v[i,1]*0.227554
        sx = sig_v[i,0]*0.345987; sy = sig_v[i,1]*0.227554
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(mx+sx*np.cos(theta), my+sy*np.sin(theta), color=colors_k[i], alpha=0.6)
        ax.scatter(mx, my, color=colors_k[i], s=120, marker='*',
                   label=f"E{i+1} ({pi_v[i]*100:.0f}%)")
    th = 0.005
    ax.add_patch(Rectangle((-th/2,-th/2), 0.346+th, 0.2275+th,
                            linewidth=5, edgecolor='red', facecolor='none'))
    ax.set_xlim(-0.05, 0.40); ax.set_ylim(-0.025, 0.30)
    ax.set_title(f"Sample {idx+1} | π={np.round(pi_v, 2)}")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7)
plt.tight_layout()
plt.savefig('expert_overview.png', bbox_inches='tight')
plt.show()

# ==========================================
# GIF — DISTRIBUIÇÃO DOS EXPERTS POR STEP
# ==========================================
gif_dir = "gif_frames"
os.makedirs(gif_dir, exist_ok=True)
frame_paths = []
for i, step in enumerate(eval_d):
    pi_v, sig_v, mu_v = get_expert_distribution(step, eval_dataset, model, device)
    frame_paths.append(plot_frame(pi_v, sig_v, mu_v, step, i, gif_dir))
if frame_paths:
    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave("expert_distributions.gif", images, duration=0.8)
    log_print("✅ GIF guardado: expert_distributions.gif")

# ==========================================
# PLOT — GUMBEL SAMPLING NO TRAIN SET COMPLETO
# ==========================================
train_dataset = PackingDataset(train_d)

tr_gts  = []
tr_preds = []
tr_k    = []
tr_yaw  = []

model.eval()
with torch.no_grad():
    for step in train_d:
        h = train_dataset.hm_cache[step['heightmap_path']].clone().unsqueeze(0).to(device)
        p_path = step['obj_path'].replace('objects/', 'objects_npy/').replace('.obj', '.npy')
        p = torch.tensor(train_dataset.pc_cache[p_path],
                         dtype=torch.float32).unsqueeze(0).to(device)

        pi, sig, mu = model(h, p, gate_temp=0.3)
        sampled, k_idx = gumbel_sample_and_predict(pi, sig, mu)

        pred   = sampled[0]
        expert = k_idx[0]

        px   = pred[0] * 0.345987
        py   = pred[1] * 0.227554
        pyaw = math.degrees(math.atan2(pred[3], pred[4]))

        gx, gy, gz, gyaw = step['human_action']

        tr_gts.append([gx, gy])
        tr_preds.append([px, py])
        tr_k.append(expert)
        tr_yaw.append(pyaw)

tr_gts   = np.array(tr_gts)
tr_preds = np.array(tr_preds)
tr_k     = np.array(tr_k)
tr_yaw   = np.array(tr_yaw)

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.scatter(tr_gts[:, 0], tr_gts[:, 1],
            c='black', marker='+', s=80, alpha=0.5, label='GT', zorder=2)

for i in range(len(tr_preds)):
    x = tr_preds[i, 0]
    y = tr_preds[i, 1]
    k = tr_k[i]
    ax1.scatter(x, y, c=[colors_k[k]], s=60, alpha=0.85, zorder=3)
    ax1.text(x, y - 0.006, f"{tr_yaw[i]:.0f}°",
             fontsize=7, fontweight='bold', color=colors_k[k],
             ha='center', va='top')

ax1.add_patch(Rectangle(
    (-0.005/2, -0.005/2), 0.346 + 0.005, 0.2275 + 0.005,
    linewidth=3, edgecolor='red', facecolor='none', zorder=1
))
ax1.set_xlim(-0.05, 0.40)
ax1.set_ylim(-0.025, 0.30)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title(f'Gumbel Sampling vs GT — Train set completo (n={len(train_d)})')
ax1.grid(True, alpha=0.3)

legend_handles = []
for k in range(K):
    mask = (tr_k == k)
    if np.sum(mask) > 0:
        legend_handles.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=colors_k[k], markersize=8,
                       label=f"E{k+1} (n={np.sum(mask)})"))
ax1.legend(handles=legend_handles + [
    plt.Line2D([0], [0], marker='+', color='black', linestyle='', label='GT')
], fontsize=8)

plt.tight_layout()
plt.savefig('train_gumbel_full.png', bbox_inches='tight', dpi=150)
plt.show()
log_print(f"✅ Plot guardado: train_gumbel_full.png  (n={len(train_d)} samples)")

if __name__ == "__main__":
    pass