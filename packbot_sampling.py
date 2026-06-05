import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import json, math, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cnn import load_and_resize_heightmap
sys.path.append(os.path.join(os.getcwd(), 'Pointnet_Pointnet2_pytorch', 'models'))
from pointnet2_cls_ssg import get_model as pointnet2_official_model

# ==========================================
# CONFIG
# ==========================================
EPOCHS = 2000
K      = 5
L_PI   = 2.0

# ==========================================
# GUMBEL SAMPLING
# ==========================================
def gumbel_sample(pi_np):
    z = np.random.gumbel(loc=0, scale=1, size=pi_np.shape)
    return (np.log(pi_np + 1e-8) + z).argmax(axis=1)

def gumbel_sample_and_predict(pi, sig, mu):
    pi_np  = pi.cpu().numpy()
    k      = gumbel_sample(pi_np)
    B      = mu.shape[0]
    mu_k   = mu[np.arange(B), k, :].cpu().numpy()
    sig_k  = sig[np.arange(B), k, :].cpu().numpy()
    noise   = np.random.randn(*mu_k.shape)
    sampled = noise * sig_k + mu_k
    return sampled, k

# ==========================================
# LOSS
# ==========================================
def mdn_loss_packing(pi, sigma, mu, target,
                     l_sep=0.05, l_sigma=0.01, l_pi=0.5):
    dist      = torch.distributions.Normal(mu, sigma)
    log_probs = dist.log_prob(target.unsqueeze(1).expand_as(mu)).sum(dim=2)
    log_pi    = torch.log(pi + 1e-7)
    log_joint = log_probs + log_pi
    nll       = -torch.logsumexp(log_joint, dim=1).mean()

    mu_xyz  = mu[:, :, :3]
    mu_i    = mu_xyz.unsqueeze(2)
    mu_j    = mu_xyz.unsqueeze(1)
    dist_sq = ((mu_i - mu_j) ** 2).sum(dim=-1)
    K_      = mu.shape[1]
    mask    = 1 - torch.eye(K_, device=mu.device).unsqueeze(0)
    sep_loss = (torch.clamp(0.3 - dist_sq, min=0) * mask).sum(dim=(1, 2)).mean()

    winner   = torch.distributions.Normal(mu, sigma).log_prob(
        target.unsqueeze(1).expand_as(mu)).sum(dim=2).argmax(dim=1)
    mask_w   = torch.zeros(pi.shape[0], K_, device=mu.device)
    mask_w.scatter_(1, winner.unsqueeze(1), 1.0)
    sigma_winner = (mask_w.unsqueeze(-1) * sigma).sum(dim=1).mean()

    # KL de pi para uniforme — previne colapso
    pi_unif = torch.ones_like(pi) / K_
    kl_unif = (pi * (torch.log(pi + 1e-7) - torch.log(pi_unif))).sum(dim=1).mean()

    loss = nll + l_sep * sep_loss + l_sigma * sigma_winner + l_pi * kl_unif
    return loss, nll.item(), sep_loss.item(), sigma_winner.item(), kl_unif.item()

# ==========================================
# MODEL
# ==========================================
class MDNLayer(nn.Module):
    def __init__(self, input_dim, action_dim, k=1):
        super().__init__()
        self.k           = k
        self.action_dim  = action_dim
        self.pi_logits   = nn.Linear(input_dim, k)
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
        print("❄️  ResNet congelada.")
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
            print(f"🔄 A carregar pesos PointNet++: {pretrained_path}")
            ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            self.base_model.load_state_dict(ckpt['model_state_dict'])
            print("✅ Pesos PointNet++ carregados!")
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()
        print("❄️  PointNet++ congelada.")

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
class MultimodalDataset(Dataset):
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
# DATA FILTERING
# ==========================================
def normalize_angle(angle):
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle

def is_valid_placement(step, tol_deg=20, z_min=0.08, z_max=0.22):
    ax = normalize_angle(step['x_y_angles'][0])
    ay = normalize_angle(step['x_y_angles'][1])
    z  = step['human_action'][2]
    return abs(ax) < tol_deg and abs(ay) < tol_deg and z_min < z < z_max

# ==========================================
# MAIN
# ==========================================
with open("dataset_packbot.json") as f:
    data = json.load(f)

sequences = defaultdict(list)
for step in data:
    sequences[step['sequence_name']].append(step)
for seq in sequences:
    sequences[seq].sort(key=lambda s: s['step'])

first_steps = []
for seq_name, steps in sequences.items():
    first_step = steps[0]
    if '003 Cracker Box' in first_step['obj_path']:
        first_steps.append(first_step)

seq_data = [s for s in first_steps if is_valid_placement(s)]
gt_positions = np.array([s['human_action'][:2] for s in seq_data])

print(f"\n📦 Cracker Box — primeiro step upright: {len(seq_data)} amostras")

dataset = MultimodalDataset(seq_data)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scales = torch.tensor([0.345987, 0.227554, 0.565], device=device)

model     = BCPackingPolicyMDN(k=K).to(device)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {total:,}")
print(f"\n{'Epoch':>6} | {'Loss':>8} | {'DistErr':>9} | {'KL_pi':>8} | {'Pi':>30}")
print("-" * 70)

best_dist    = float('inf')
best_weights = None
history = {
    'loss': [], 'nll': [], 'sep': [], 'sigma_w': [],
    'kl_pi': [], 'dist_err': [], 'pi_mean': [], 'epochs': []
}

for epoch in range(EPOCHS):
    model.train()
    progress  = epoch / EPOCHS
    l_sep     = 0.10 * min(1.0, progress / 0.4)
    l_sigma   = 0.01 * min(1.0, progress / 0.3)
    gate_temp = max(0.1, 1.0 - 0.9 * progress)

    t_l = 0.0
    for h, p, a in loader:
        h, p, a = h.to(device), p.to(device), a.to(device)
        optimizer.zero_grad()
        pi, sig, mu = model(h, p, gate_temp=gate_temp)
        loss, nll, sep, sig_w, kl = mdn_loss_packing(
            pi, sig, mu, a,
            l_sep=l_sep, l_sigma=l_sigma, l_pi=L_PI
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_l += loss.item()
    scheduler.step()
    avg_loss = t_l / len(loader)

    if (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            dist_val = n_val = 0
            pi_avg   = []
            for h, p, a in loader:
                h, p, a = h.to(device), p.to(device), a.to(device)
                pi_e, sig_e, mu_e = model(h, p, gate_temp=0.1)
                sampled, _ = gumbel_sample_and_predict(pi_e, sig_e, mu_e)
                sampled_t  = torch.tensor(sampled, dtype=torch.float32).to(device)
                dist_val  += torch.norm(
                    sampled_t[:, :3] * scales - a[:, :3] * scales, dim=1
                ).sum().item()
                n_val    += a.size(0)
                pi_avg.append(pi_e.mean(dim=0).cpu().numpy())
            dist_err  = (dist_val / n_val) * 100
            pi_mean_e = np.mean(pi_avg, axis=0)

            history['loss'].append(avg_loss)
            history['nll'].append(nll)
            history['sep'].append(sep)
            history['sigma_w'].append(sig_w)
            history['kl_pi'].append(kl)
            history['dist_err'].append(dist_err)
            history['epochs'].append(epoch + 1)
            history['pi_mean'].append(pi_mean_e)

        pi_str = " ".join(f"E{i+1}:{pi_mean_e[i]*100:.0f}%" for i in range(K))
        flag = "✅ best" if dist_err < best_dist else ""
        print(f"{epoch+1:>6} | {avg_loss:>8.4f} | {dist_err:>7.2f}cm | {kl:>8.4f} | {pi_str} {flag}")

        if dist_err < best_dist:
            best_dist    = dist_err
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

model.load_state_dict(best_weights)
torch.save(best_weights, "multimodal_cracker_box.pth")
print(f"\n✅ Best: {best_dist:.2f}cm")

# ==========================================
# SAMPLING — 10 poses por heightmap
# ==========================================
model.eval()
N_SAMPLES = 10

# usa o primeiro sample (bin vazio) — todos têm o mesmo heightmap
step   = seq_data[0]
h_ref  = dataset.hm_cache[step['heightmap_path']].clone().unsqueeze(0).to(device)
p_path = step['obj_path'].replace('objects/', 'objects_npy/').replace('.obj', '.npy')
p_ref  = torch.tensor(dataset.pc_cache[p_path], dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    pi_ref, sig_ref, mu_ref = model(h_ref, p_ref, gate_temp=0.1)

pi_vals  = pi_ref[0].cpu().numpy()
mu_vals  = mu_ref[0].cpu().numpy()
sig_vals = sig_ref[0].cpu().numpy()

print(f"\n📊 Pi por expert: " + " | ".join(f"E{i+1}:{pi_vals[i]*100:.1f}%" for i in range(K)))
print(f"\n{'Expert':>8} | {'π':>7} | {'μ_X':>7} {'μ_Y':>7} {'μ_Z':>7} | {'σ_X':>7} {'σ_Y':>7}")
print("-" * 65)
for i in range(K):
    mx = mu_vals[i,0]*0.345987; my = mu_vals[i,1]*0.227554
    mz = mu_vals[i,2]*0.565
    sx = sig_vals[i,0]*0.345987; sy = sig_vals[i,1]*0.227554
    print(f"Expert {i+1:>1} | {pi_vals[i]*100:>6.1f}% | "
          f"{mx:>7.3f} {my:>7.3f} {mz:>7.3f} | {sx:>7.3f} {sy:>7.3f}")

# Gera N_SAMPLES amostras
samples_list = []
for _ in range(N_SAMPLES):
    pi_batch  = pi_ref.expand(1, -1)
    sig_batch = sig_ref.expand(1, -1, -1)
    mu_batch  = mu_ref.expand(1, -1, -1)
    s, k_s    = gumbel_sample_and_predict(pi_batch, sig_batch, mu_batch)
    samples_list.append((s[0], k_s[0]))

print(f"\n{'#':>3} | {'Expert':>6} | {'X':>7} {'Y':>7} {'Z':>7} {'Yaw':>8}")
print("-" * 45)
for i, (s, k_i) in enumerate(samples_list):
    sx   = s[0]*0.345987; sy = s[1]*0.227554; sz = s[2]*0.565
    syaw = math.degrees(math.atan2(s[3], s[4]))
    print(f"{i+1:>3} | E{k_i+1:>5} | {sx:>7.3f} {sy:>7.3f} {sz:>7.3f} {syaw:>8.1f}°")

# ==========================================
# PLOT — samples vs GT
# ==========================================
colors_k = cm.tab10(np.linspace(0, 1, K))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Esquerda — μ dos experts + samples
ax = axes[0]
ax.scatter(gt_positions[:, 0], gt_positions[:, 1],
           c='black', marker='+', s=60, alpha=0.4, label='GT (63 humans)', zorder=2)

# μ de cada expert (elipse 1σ)
for i in range(K):
    mx = mu_vals[i,0]*0.345987; my = mu_vals[i,1]*0.227554
    sx = sig_vals[i,0]*0.345987; sy = sig_vals[i,1]*0.227554
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(mx + sx*np.cos(theta), my + sy*np.sin(theta),
            color=colors_k[i], alpha=0.5, linewidth=1.5)
    ax.scatter(mx, my, c=[colors_k[i]], s=200, marker='*', zorder=4,
               label=f'E{i+1} μ  π={pi_vals[i]*100:.0f}%')

ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
ax.set_title('Experts μ + elipse 1σ vs GT')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Direita — 10 samples vs GT
ax2 = axes[1]
ax2.scatter(gt_positions[:, 0], gt_positions[:, 1],
            c='black', marker='+', s=60, alpha=0.4, label='GT', zorder=2)

for i, (s, k_i) in enumerate(samples_list):
    sx = s[0]*0.345987; sy = s[1]*0.227554
    ax2.scatter(sx, sy, c=[colors_k[k_i]], s=150,
                marker='o', edgecolors='black', linewidth=0.8, zorder=3)
    ax2.text(sx+0.005, sy, f'{i+1}', fontsize=8)

# Legenda por expert
for i in range(K):
    ax2.scatter([], [], c=[colors_k[i]], s=80,
                label=f'Expert {i+1} (π={pi_vals[i]*100:.0f}%)')

ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
ax2.set_title(f'Gumbel sampling — {N_SAMPLES} amostras')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

plt.suptitle(f'MDN k={K} + KL anti-collapse — 1st Cracker Box upright', fontsize=12)
plt.tight_layout()
plt.savefig('mdn_sampling_kl.png', bbox_inches='tight', dpi=150)
plt.show()

# ==========================================
# PLOT — evolução pi e loss
# ==========================================
epochs_plot = history['epochs']
pi_history  = np.array(history['pi_mean'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i in range(K):
    axes[0].plot(epochs_plot, pi_history[:, i]*100,
                 marker='o', markersize=3,
                 label=f'Expert {i+1}', color=colors_k[i], linewidth=2)
axes[0].axhline(100/K, color='gray', linestyle='--', alpha=0.5,
                label=f'Uniform (1/K={100/K:.0f}%)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Pi médio (%)')
axes[0].set_title(f'Evolução π — KL anti-collapse (l_pi={L_PI})')
axes[0].set_ylim(0, 105); axes[0].legend(ncol=K+1, fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_plot, history['kl_pi'], color='purple', linewidth=2, label='KL(π||uniform)')
axes[1].plot(epochs_plot, history['nll'],   color='C0',     linewidth=2, label='NLL')
axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].set_title('KL + NLL ao longo do treino')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pi_kl_evolution.png', bbox_inches='tight', dpi=150)
plt.show()

# ── Load best ──
model.load_state_dict(best_weights)
torch.save(best_weights, "multimodal_cracker_box.pth")
print(f"\n✅ Best: {best_dist:.2f}cm | Guardado: multimodal_cracker_box.pth")

# ── Per-sample table com Gumbel sampling ──
model.eval()
print(f"\n{'#':>4} | {'Expert':>6} | {'Pred X':>7} {'Pred Y':>7} {'Pred Z':>7} {'Pred Yaw':>9} "
      f"| {'GT X':>7} {'GT Y':>7} {'GT Z':>7} | {'Dist':>7}")
print("-" * 100)

with torch.no_grad():
    for idx, step in enumerate(seq_data):
        h = dataset.hm_cache[step['heightmap_path']].clone().unsqueeze(0).to(device)
        p_path = step['obj_path'].replace('objects/', 'objects_npy/').replace('.obj', '.npy')
        p = torch.tensor(dataset.pc_cache[p_path], dtype=torch.float32).unsqueeze(0).to(device)

        pi, sig, mu = model(h, p, gate_temp=0.1)

        # ── Gumbel sample ──
        sampled, k_idx = gumbel_sample_and_predict(pi, sig, mu)
        pred = sampled[0]   # (action_dim,)

        px   = pred[0] * 0.345987
        py   = pred[1] * 0.227554
        pz   = pred[2] * 0.565
        pyaw = math.degrees(math.atan2(pred[3], pred[4]))

        gx, gy, gz, gyaw = step['human_action']
        dist_cm = math.sqrt((px-gx)**2 + (py-gy)**2 + (pz-gz)**2) * 100

        print(f"{idx+1:>4} | E{k_idx[0]+1:>5} | {px:>7.3f} {py:>7.3f} {pz:>7.3f} {pyaw:>9.1f} "
              f"| {gx:>7.3f} {gy:>7.3f} {gz:>7.3f} | {dist_cm:>6.2f}cm")

# ── Expert specialisation scatter ──
model.eval()
expert_positions = [[] for _ in range(K)]
gt_positions     = np.array([s['human_action'][:2] for s in seq_data])
colors_pi        = cm.tab10(np.linspace(0, 1, K))

with torch.no_grad():
    for idx, step in enumerate(seq_data):
        h = dataset.hm_cache[step['heightmap_path']].clone().unsqueeze(0).to(device)
        p_path = step['obj_path'].replace('objects/', 'objects_npy/').replace('.obj', '.npy')
        p = torch.tensor(dataset.pc_cache[p_path], dtype=torch.float32).unsqueeze(0).to(device)
        pi, sig, mu = model(h, p, gate_temp=0.1)
        sampled, k_idx = gumbel_sample_and_predict(pi, sig, mu)
        pred = sampled[0]
        expert_positions[k_idx[0]].append((pred[0]*0.345987, pred[1]*0.227554))

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(gt_positions[:, 0], gt_positions[:, 1],
           c='black', marker='+', s=80, alpha=0.4, label='GT', zorder=2)
for k_i in range(K):
    if expert_positions[k_i]:
        pts = np.array(expert_positions[k_i])
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=[colors_pi[k_i]], s=80, alpha=0.8,
                   label=f'Expert {k_i+1} ({len(pts)} samples)', zorder=3)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'Gumbel Sampling — k={K} — 1st Cracker Box upright')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('expert_specialisation_gumbel.png', bbox_inches='tight', dpi=150)
plt.show()

# ── Pi evolution ──
pi_history = np.array(history['pi_mean'])
epochs_plot = history['epochs']
fig, ax = plt.subplots(figsize=(12, 5))
for i in range(K):
    ax.plot(epochs_plot, pi_history[:, i] * 100,
            marker='o', markersize=4,
            label=f'Expert {i+1}', color=colors_pi[i], linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Pi médio (%)')
ax.set_title(f'Evolução do π — k={K} — Gumbel Sampling')
ax.set_ylim(0, 105)
ax.legend(ncol=K)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pi_evolution_gumbel.png', bbox_inches='tight', dpi=150)
plt.show()

# ── Load best & evaluate ──
model.load_state_dict(best_weights)
torch.save(best_weights, "multimodal_cracker_box.pth")
print(f"\n✅ Best: {best_dist:.2f}cm | Guardado: multimodal_cracker_box.pth")

# ── Per-sample table ──
model.eval()
print(f"\n{'#':>4} | {'σ_winner':>10} | {'Pred X':>7} {'Pred Y':>7} {'Pred Z':>7} {'Pred Yaw':>9} "
      f"| {'GT X':>7} {'GT Y':>7} {'GT Z':>7} {'GT Yaw':>9} | {'Dist':>7} | Expert")
print("-" * 115)

with torch.no_grad():
    for idx, step in enumerate(seq_data):
        h = dataset.hm_cache[step['heightmap_path']].clone().unsqueeze(0).to(device)
        p_path = step['obj_path'].replace('objects/', 'objects_npy/').replace('.obj', '.npy')
        p = torch.tensor(dataset.pc_cache[p_path], dtype=torch.float32).unsqueeze(0).to(device)

        pi, sig, mu = model(h, p, gate_temp=0.1)
        best_idx    = torch.argmax(pi, dim=1)
        sig_winner  = sig[0, best_idx[0], :].mean().item()
        pred        = mu[0, best_idx[0], :].cpu().numpy()

        px   = pred[0] * 0.345987
        py   = pred[1] * 0.227554
        pz   = pred[2] * 0.565
        pyaw = math.degrees(math.atan2(pred[3], pred[4]))

        gx, gy, gz, gyaw = step['human_action']
        dist_cm = math.sqrt((px-gx)**2 + (py-gy)**2 + (pz-gz)**2) * 100

        print(f"{idx+1:>4} | {sig_winner:>10.4f} | {px:>7.3f} {py:>7.3f} {pz:>7.3f} {pyaw:>9.1f} "
              f"| {gx:>7.3f} {gy:>7.3f} {gz:>7.3f} {gyaw:>9.1f} | {dist_cm:>6.2f}cm | E{best_idx[0].item()+1}")

# ── Loss components plot ──
epochs_plot = history['epochs']
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
for ax, key, title, color in zip(
    axes.flat,
    ['loss','nll','sep','sigma_w','dist_err'],
    ['Loss Total','NLL','NLL WTA','Separation Loss','Sigma Winner','Dist Error (cm)'],
    ['black','C0','C1','C2','C3','C4']
):
    ax.plot(epochs_plot, history[key], color=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
plt.suptitle(f'Loss components — k={K} — Multimodal Cracker Box', fontsize=13)
plt.tight_layout()
plt.savefig('loss_components_multimodal.png', bbox_inches='tight', dpi=150)
plt.show()

# ── Pi evolution plot ──
pi_history = np.array(history['pi_mean'])
colors_pi  = cm.tab10(np.linspace(0, 1, K))
fig, ax = plt.subplots(figsize=(12, 5))
for i in range(K):
    ax.plot(epochs_plot, pi_history[:, i] * 100,
            marker='o', markersize=4,
            label=f'Expert {i+1}', color=colors_pi[i], linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Pi médio (%)')
ax.set_title(f'Evolução do π médio por expert — k={K}')
ax.set_ylim(0, 105)
ax.legend(ncol=K)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pi_evolution_multimodal.png', bbox_inches='tight', dpi=150)
plt.show()

# ── Expert specialisation scatter ──
expert_positions = [[] for _ in range(K)]
gt_positions     = np.array([s['human_action'][:2] for s in seq_data])

model.eval()
with torch.no_grad():
    for idx, step in enumerate(seq_data):
        h = dataset.hm_cache[step['heightmap_path']].clone().unsqueeze(0).to(device)
        p_path = step['obj_path'].replace('objects/', 'objects_npy/').replace('.obj', '.npy')
        p = torch.tensor(dataset.pc_cache[p_path], dtype=torch.float32).unsqueeze(0).to(device)
        pi, sig, mu = model(h, p, gate_temp=0.1)
        best_idx    = torch.argmax(pi, dim=1).item()
        pred        = mu[0, best_idx, :].cpu().numpy()
        expert_positions[best_idx].append((pred[0]*0.345987, pred[1]*0.227554))

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(gt_positions[:, 0], gt_positions[:, 1],
           c='black', marker='+', s=80, alpha=0.4, label='GT', zorder=2)
for k_i in range(K):
    if expert_positions[k_i]:
        pts = np.array(expert_positions[k_i])
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=[colors_pi[k_i]], s=80, alpha=0.8,
                   label=f'Expert {k_i+1} ({len(pts)} samples)', zorder=3)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'Expert specialisation — k={K} — 1st Cracker Box upright')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('expert_specialisation.png', bbox_inches='tight', dpi=150)
plt.show()

# ── Pi gating weights per sample ──
pi_matrix = []
with torch.no_grad():
    for step in seq_data:
        h = dataset.hm_cache[step['heightmap_path']].clone().unsqueeze(0).to(device)
        p_path = step['obj_path'].replace('objects/', 'objects_npy/').replace('.obj', '.npy')
        p = torch.tensor(dataset.pc_cache[p_path], dtype=torch.float32).unsqueeze(0).to(device)
        pi, _, _ = model(h, p, gate_temp=0.1)
        pi_matrix.append(pi.squeeze(0).cpu().numpy())

pi_matrix = np.array(pi_matrix)
n_steps   = len(seq_data)
steps_idx = np.arange(1, n_steps + 1)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
bottom = np.zeros(n_steps)
for i in range(K):
    axes[0].bar(steps_idx, pi_matrix[:, i] * 100, bottom=bottom * 100,
                label=f'Expert {i+1}', color=colors_pi[i], alpha=0.85)
    bottom += pi_matrix[:, i]
axes[0].set_title(f'Gating weights π per sample — k={K}')
axes[0].set_xlabel('Sample')
axes[0].set_ylabel('Pi (%)')
axes[0].set_ylim(0, 100)
axes[0].legend(ncol=K)
axes[0].grid(True, alpha=0.3, axis='y')

for i in range(K):
    axes[1].plot(steps_idx, pi_matrix[:, i] * 100,
                 marker='o', markersize=3,
                 label=f'Expert {i+1}', color=colors_pi[i], linewidth=1.5)
axes[1].set_xlabel('Sample')
axes[1].set_ylabel('Pi (%)')
axes[1].set_title('Pi per expert per sample')
axes[1].legend(ncol=K)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pi_per_sample_multimodal.png', bbox_inches='tight', dpi=150)
plt.show()

print(f"\n{'Sample':>6} | " + " | ".join(f"E{i+1:>6}" for i in range(K)) + " | Winner")
print("-" * (10 + 11 * K))
for s in range(n_steps):
    winner = np.argmax(pi_matrix[s]) + 1
    vals   = " | ".join(f"{pi_matrix[s,i]*100:>6.1f}%" for i in range(K))
    print(f"{s+1:>6} | {vals} | E{winner}")

# ── Visualização 2D dos experts no bin ──
model.eval()
expert_positions = [[] for _ in range(K)]

with torch.no_grad():
    for idx, step in enumerate(seq_data):
        h = dataset.hm_cache[step['heightmap_path']].clone().unsqueeze(0).to(device)
        p_path = step['obj_path'].replace('objects/', 'objects_npy/').replace('.obj', '.npy')
        p = torch.tensor(dataset.pc_cache[p_path], dtype=torch.float32).unsqueeze(0).to(device)

        pi, sig, mu = model(h, p, gate_temp=0.1)
        best_idx    = torch.argmax(pi, dim=1).item()
        pred        = mu[0, best_idx, :].cpu().numpy()
        px = pred[0] * 0.345987
        py = pred[1] * 0.227554
        expert_positions[best_idx].append((px, py))

fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.tab10(np.linspace(0, 1, K))

# GT positions
ax.scatter(gt_positions[:, 0], gt_positions[:, 1],
           c='black', marker='+', s=60, alpha=0.4, label='GT', zorder=2)

# Expert predictions
for k_i in range(K):
    if expert_positions[k_i]:
        pts = np.array(expert_positions[k_i])
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=[colors[k_i]], s=80, alpha=0.7,
                   label=f'Expert {k_i+1} ({len(pts)} steps)', zorder=3)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'Expert specialisation — k={K} — 1st Cracker Box upright')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('expert_specialisation.png', bbox_inches='tight', dpi=150)
plt.show()

# ==========================================
# DIAGNÓSTICO 1 — O modelo usa o heightmap?
# ==========================================
print("=" * 60)
print("DIAGNÓSTICO 1 — Sensibilidade ao heightmap")
print("=" * 60)

step = seq_data[0]  # usa o primeiro step como exemplo
p_path = step['obj_path'].replace('objects/', 'objects_npy/').replace('.obj', '.npy')
pc_single = torch.tensor(dataset.pc_cache[p_path], dtype=torch.float32).unsqueeze(0).to(device)

h_real   = dataset.hm_cache[step['heightmap_path']].clone().unsqueeze(0).to(device)
h_zeros  = torch.zeros_like(h_real)
h_random = torch.rand_like(h_real)

# heightmap de outro step (completamente diferente)
other_step = seq_data[10]
h_other  = dataset.hm_cache[other_step['heightmap_path']].clone().unsqueeze(0).to(device)

with torch.no_grad():
    def get_pred(h_in):
        pi_, _, mu_ = model(h_in, pc_single, gate_temp=0.1)
        idx = torch.argmax(pi_, dim=1)
        return mu_[0, idx[0], :3].cpu().numpy()

    pred_real   = get_pred(h_real)
    pred_zeros  = get_pred(h_zeros)
    pred_random = get_pred(h_random)
    pred_other  = get_pred(h_other)

def denorm(p):
    return np.array([p[0]*0.345987, p[1]*0.227554, p[2]*0.565])

pr, pz, prn, po = denorm(pred_real), denorm(pred_zeros), denorm(pred_random), denorm(pred_other)
gt = np.array(step['human_action'][:3])

print(f"GT               : x={gt[0]:.3f}  y={gt[1]:.3f}  z={gt[2]:.3f}")
print(f"Real heightmap   : x={pr[0]:.3f}  y={pr[1]:.3f}  z={pr[2]:.3f}  err={np.linalg.norm(pr-gt)*100:.2f}cm")
print(f"Empty heightmap  : x={pz[0]:.3f}  y={pz[1]:.3f}  z={pz[2]:.3f}  err={np.linalg.norm(pz-gt)*100:.2f}cm")
print(f"Random heightmap : x={prn[0]:.3f}  y={prn[1]:.3f}  z={prn[2]:.3f}  err={np.linalg.norm(prn-gt)*100:.2f}cm")
print(f"Other step hmap  : x={po[0]:.3f}  y={po[1]:.3f}  z={po[2]:.3f}  err={np.linalg.norm(po-gt)*100:.2f}cm")

# ==========================================
# DIAGNÓSTICO 2 — Erro por dimensão
# ==========================================
print("\n" + "=" * 60)
print("DIAGNÓSTICO 2 — Erro por dimensão X / Y / Z")
print("=" * 60)

errs_x, errs_y, errs_z = [], [], []
with torch.no_grad():
    for h, p, a in loader:
        h, p, a = h.to(device), p.to(device), a.to(device)
        pi_d, _, mu_d = model(h, p, gate_temp=0.1)
        best_idx = torch.argmax(pi_d, dim=1)
        pred = mu_d[torch.arange(mu_d.size(0)), best_idx, :3]
        errs_x.append(((pred[:,0] - a[:,0]) * 0.345987).abs() * 100)
        errs_y.append(((pred[:,1] - a[:,1]) * 0.227554).abs() * 100)
        errs_z.append(((pred[:,2] - a[:,2]) * 0.565).abs()    * 100)

ex = torch.cat(errs_x).mean().item()
ey = torch.cat(errs_y).mean().item()
ez = torch.cat(errs_z).mean().item()
print(f"Erro médio X : {ex:.3f}cm")
print(f"Erro médio Y : {ey:.3f}cm")
print(f"Erro médio Z : {ez:.3f}cm")
print(f"Total 3D     : {math.sqrt(ex**2 + ey**2 + ez**2):.3f}cm")

# ==========================================
# DIAGNÓSTICO 3 — Heightmaps são diferentes?
# ==========================================
print("\n" + "=" * 60)
print("DIAGNÓSTICO 3 — Distância entre features de heightmaps")
print("=" * 60)

feats = []
with torch.no_grad():
    for step in seq_data:
        h = dataset.hm_cache[step['heightmap_path']].clone().unsqueeze(0).to(device)
        # extrai features antes do MDN
        h_f = model.heightmap_encoder(h.repeat(1, 3, 1, 1))
        feats.append(h_f.cpu())

feats = torch.cat(feats, dim=0)   # (20, 256)

# distância média entre todas as features
dists = []
for i in range(len(feats)):
    for j in range(i+1, len(feats)):
        dists.append((feats[i] - feats[j]).norm().item())

print(f"Distância média entre features de heightmaps diferentes: {np.mean(dists):.4f}")
print(f"Distância mín: {np.min(dists):.4f}  |  Distância máx: {np.max(dists):.4f}")
print()
print("Se distâncias forem perto de 0 → ResNet produz features")
print("quase iguais para heightmaps diferentes → modelo não usa heightmap")
# ==========================================
# PLOT — Pi por expert em cada step
# ==========================================
import matplotlib.pyplot as plt
import matplotlib.cm as cm

model.eval()
pi_matrix = []  # (n_steps, k)

with torch.no_grad():
    for step in seq_data:
        h = dataset.hm_cache[step['heightmap_path']].clone().unsqueeze(0).to(device)
        p_path = step['obj_path'].replace('objects/', 'objects_npy/').replace('.obj', '.npy')
        p = torch.tensor(dataset.pc_cache[p_path], dtype=torch.float32).unsqueeze(0).to(device)

        pi, _, _ = model(h, p, gate_temp=0.1)
        pi_matrix.append(pi.squeeze(0).cpu().numpy())  # (k,)

pi_matrix = np.array(pi_matrix)  # (20, k)
n_steps, k = pi_matrix.shape
steps = np.arange(1, n_steps + 1)
colors = cm.tab10(np.linspace(0, 1, k))

# ── Stacked bar chart ──
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Gráfico 1 — Stacked bars (percentagem)
bottom = np.zeros(n_steps)
for i in range(k):
    axes[0].bar(steps, pi_matrix[:, i] * 100,
                bottom=bottom * 100,
                label=f'Expert {i+1}',
                color=colors[i], alpha=0.85)
    bottom += pi_matrix[:, i]

axes[0].set_xlabel("Step")
axes[0].set_ylabel("Pi (%)")
axes[0].set_title(f"Gating weights π por step — k={k}, gate_temp=0.1")
axes[0].set_xticks(steps)
axes[0].set_xlim(0.5, n_steps + 0.5)
axes[0].set_ylim(0, 100)
axes[0].legend(loc='upper right', ncol=k)
axes[0].grid(True, alpha=0.3, axis='y')

# Gráfico 2 — Linha por expert
for i in range(k):
    axes[1].plot(steps, pi_matrix[:, i] * 100,
                 marker='o', label=f'Expert {i+1}',
                 color=colors[i], linewidth=1.5, markersize=5)

axes[1].set_xlabel("Step")
axes[1].set_ylabel("Pi (%)")
axes[1].set_title("Evolução do π por expert ao longo da sequência")
axes[1].set_xticks(steps)
axes[1].set_xlim(0.5, n_steps + 0.5)
axes[1].legend(loc='upper right', ncol=k)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pi_experts.png", bbox_inches='tight', dpi=150)
plt.show()

# ── Tabela resumo ──
print(f"\n{'Step':>4} | " + " | ".join(f"E{i+1:>6}" for i in range(k)) + " | Winner")
print("-" * (8 + 11 * k))
for s in range(n_steps):
    winner = np.argmax(pi_matrix[s]) + 1
    vals   = " | ".join(f"{pi_matrix[s,i]*100:>6.1f}%" for i in range(k))
    print(f"{s+1:>4} | {vals} | E{winner}")

# ==========================================
# PLOT — Evolução da loss e componentes
# ==========================================
epochs_plot = history['epochs']
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Total loss
axes[0,0].plot(epochs_plot, history['loss'], color='black', linewidth=2)
axes[0,0].set_title('Loss Total')
axes[0,0].set_xlabel('Epoch')
axes[0,0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0,0].grid(True, alpha=0.3)

# NLL
axes[0,1].plot(epochs_plot, history['nll'], color='C0', linewidth=2)
axes[0,1].set_title('NLL')
axes[0,1].set_xlabel('Epoch')
axes[0,1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0,1].grid(True, alpha=0.3)


# Separation loss
axes[1,1].plot(epochs_plot, history['sep'], color='C2', linewidth=2)
axes[1,1].set_title('Separation Loss')
axes[1,1].set_xlabel('Epoch')
axes[1,1].grid(True, alpha=0.3)

# Sigma winner
axes[2,0].plot(epochs_plot, history['sigma_w'], color='C3', linewidth=2)
axes[2,0].set_title('Sigma Winner')
axes[2,0].set_xlabel('Epoch')
axes[2,0].grid(True, alpha=0.3)

# Dist error
axes[2,1].plot(epochs_plot, history['dist_err'], color='C4', linewidth=2)
axes[2,1].set_title('Dist Error (cm)')
axes[2,1].set_xlabel('Epoch')
axes[2,1].grid(True, alpha=0.3)

plt.suptitle(f'Evolução das componentes da loss — k={K}', fontsize=13)
plt.tight_layout()
plt.savefig('loss_components.png', bbox_inches='tight', dpi=150)
plt.show()

# ==========================================
# PLOT — Evolução do Pi ao longo do treino
# ==========================================
pi_history = np.array(history['pi_mean'])  # (n_checkpoints, k)
colors_pi  = plt.cm.tab10(np.linspace(0, 1, K))

fig, ax = plt.subplots(figsize=(12, 5))
for i in range(K):
    ax.plot(epochs_plot, pi_history[:, i] * 100,
            marker='o', markersize=4,
            label=f'Expert {i+1}', color=colors_pi[i], linewidth=2)

ax.set_xlabel('Epoch')
ax.set_ylabel('Pi médio (%)')
ax.set_title(f'Evolução do π médio por expert — k={K}')
ax.set_ylim(0, 105)
ax.legend(ncol=K)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pi_evolution.png', bbox_inches='tight', dpi=150)
plt.show()# ==========================================
# PLOT — Evolução da loss e componentes
# ==========================================
epochs_plot = history['epochs']
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Total loss
axes[0,0].plot(epochs_plot, history['loss'], color='black', linewidth=2)
axes[0,0].set_title('Loss Total')
axes[0,0].set_xlabel('Epoch')
axes[0,0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0,0].grid(True, alpha=0.3)

# NLL
axes[0,1].plot(epochs_plot, history['nll'], color='C0', linewidth=2)
axes[0,1].set_title('NLL')
axes[0,1].set_xlabel('Epoch')
axes[0,1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0,1].grid(True, alpha=0.3)


# Separation loss
axes[1,1].plot(epochs_plot, history['sep'], color='C2', linewidth=2)
axes[1,1].set_title('Separation Loss')
axes[1,1].set_xlabel('Epoch')
axes[1,1].grid(True, alpha=0.3)

# Sigma winner
axes[2,0].plot(epochs_plot, history['sigma_w'], color='C3', linewidth=2)
axes[2,0].set_title('Sigma Winner')
axes[2,0].set_xlabel('Epoch')
axes[2,0].grid(True, alpha=0.3)

# Dist error
axes[2,1].plot(epochs_plot, history['dist_err'], color='C4', linewidth=2)
axes[2,1].set_title('Dist Error (cm)')
axes[2,1].set_xlabel('Epoch')
axes[2,1].grid(True, alpha=0.3)

plt.suptitle(f'Evolução das componentes da loss — k={K}', fontsize=13)
plt.tight_layout()
plt.savefig('loss_components.png', bbox_inches='tight', dpi=150)
plt.show()

# ==========================================
# PLOT — Evolução do Pi ao longo do treino
# ==========================================
pi_history = np.array(history['pi_mean'])  # (n_checkpoints, k)
colors_pi  = plt.cm.tab10(np.linspace(0, 1, K))

fig, ax = plt.subplots(figsize=(12, 5))
for i in range(K):
    ax.plot(epochs_plot, pi_history[:, i] * 100,
            marker='o', markersize=4,
            label=f'Expert {i+1}', color=colors_pi[i], linewidth=2)

ax.set_xlabel('Epoch')
ax.set_ylabel('Pi médio (%)')
ax.set_title(f'Evolução do π médio por expert — k={K}')
ax.set_ylim(0, 105)
ax.legend(ncol=K)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pi_evolution.png', bbox_inches='tight', dpi=150)
plt.show()# ==========================================
# PLOT — Evolução da loss e componentes
# ==========================================
epochs_plot = history['epochs']
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Total loss
axes[0,0].plot(epochs_plot, history['loss'], color='black', linewidth=2)
axes[0,0].set_title('Loss Total')
axes[0,0].set_xlabel('Epoch')
axes[0,0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0,0].grid(True, alpha=0.3)

# NLL
axes[0,1].plot(epochs_plot, history['nll'], color='C0', linewidth=2)
axes[0,1].set_title('NLL')
axes[0,1].set_xlabel('Epoch')
axes[0,1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0,1].grid(True, alpha=0.3)


# Separation loss
axes[1,1].plot(epochs_plot, history['sep'], color='C2', linewidth=2)
axes[1,1].set_title('Separation Loss')
axes[1,1].set_xlabel('Epoch')
axes[1,1].grid(True, alpha=0.3)

# Sigma winner
axes[2,0].plot(epochs_plot, history['sigma_w'], color='C3', linewidth=2)
axes[2,0].set_title('Sigma Winner')
axes[2,0].set_xlabel('Epoch')
axes[2,0].grid(True, alpha=0.3)

# Dist error
axes[2,1].plot(epochs_plot, history['dist_err'], color='C4', linewidth=2)
axes[2,1].set_title('Dist Error (cm)')
axes[2,1].set_xlabel('Epoch')
axes[2,1].grid(True, alpha=0.3)

plt.suptitle(f'Evolução das componentes da loss — k={K}', fontsize=13)
plt.tight_layout()
plt.savefig('loss_components.png', bbox_inches='tight', dpi=150)
plt.show()

# ==========================================
# PLOT — Evolução do Pi ao longo do treino
# ==========================================
pi_history = np.array(history['pi_mean'])  # (n_checkpoints, k)
colors_pi  = plt.cm.tab10(np.linspace(0, 1, K))

fig, ax = plt.subplots(figsize=(12, 5))
for i in range(K):
    ax.plot(epochs_plot, pi_history[:, i] * 100,
            marker='o', markersize=4,
            label=f'Expert {i+1}', color=colors_pi[i], linewidth=2)

ax.set_xlabel('Epoch')
ax.set_ylabel('Pi médio (%)')
ax.set_title(f'Evolução do π médio por expert — k={K}')
ax.set_ylim(0, 105)
ax.legend(ncol=K)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pi_evolution.png', bbox_inches='tight', dpi=150)
plt.show()
    
step = seq_data[0]
h = dataset.hm_cache[step['heightmap_path']].clone().unsqueeze(0).to(device)
p_path = step['obj_path'].replace('objects/', 'objects_npy/').replace('.obj', '.npy')
p = torch.tensor(dataset.pc_cache[p_path], dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    pi, sig, mu = model(h, p, gate_temp=0.1)
    best_idx = torch.argmax(pi, dim=1)
    mu_w  = mu[0, best_idx[0], :3]
    sig_w = sig[0, best_idx[0], :3]

# Amostras da gaussiana vencedora
samples = torch.normal(mu_w.expand(50, -1), sig_w.expand(50, -1))

# Desnormaliza
samples_m = samples.cpu().numpy()
samples_m[:, 0] *= 0.345987
samples_m[:, 1] *= 0.227554
samples_m[:, 2] *= 0.565

gt = step['human_action'][:3]
print(f"GT: {gt}")
print(f"μ:  {mu_w.cpu().numpy() * np.array([0.345987, 0.227554, 0.565])}")
print(f"σ:  {sig_w.cpu().numpy() * np.array([0.345987, 0.227554, 0.565])}")
print(f"Spread das amostras (std): {samples_m.std(axis=0)}")

model.eval()
# Como o input é sempre igual, basta um sample
step = seq_data[0]
h = dataset.hm_cache[step['heightmap_path']].clone().unsqueeze(0).to(device)
p_path = step['obj_path'].replace('objects/', 'objects_npy/').replace('.obj', '.npy')
p = torch.tensor(dataset.pc_cache[p_path], dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    pi, sig, mu = model(h, p, gate_temp=0.1)

pi_vals  = pi[0].cpu().numpy()         # (K,)
mu_vals  = mu[0].cpu().numpy()         # (K, 5)
sig_vals = sig[0].cpu().numpy()        # (K, 5)

# Desnormaliza
mu_m  = mu_vals.copy()
mu_m[:, 0] *= 0.345987
mu_m[:, 1] *= 0.227554
mu_m[:, 2] *= 0.565
sig_m = sig_vals.copy()
sig_m[:, 0] *= 0.345987
sig_m[:, 1] *= 0.227554
sig_m[:, 2] *= 0.565

print(f"\n{'Expert':>8} | {'π':>8} | {'μ_X':>8} {'μ_Y':>8} {'μ_Z':>8} | {'σ_X':>8} {'σ_Y':>8} {'σ_Z':>8}")
print("-" * 70)
for i in range(K):
    print(f"Expert {i+1:>1} | {pi_vals[i]*100:>7.1f}% | "
          f"{mu_m[i,0]:>8.3f} {mu_m[i,1]:>8.3f} {mu_m[i,2]:>8.3f} | "
          f"{sig_m[i,0]:>8.3f} {sig_m[i,1]:>8.3f} {sig_m[i,2]:>8.3f}")

# Scatter com TODOS os experts
colors = plt.cm.tab10(np.linspace(0, 1, K))
positions = np.array([s['human_action'][:2] for s in seq_data])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Esquerda — μ de cada expert (tamanho proporcional a π)
ax = axes[0]
ax.scatter(gt_positions[:, 0], gt_positions[:, 1],
           c='black', marker='+', s=60, alpha=0.4, label='GT', zorder=2)
for i in range(K):
    size = max(50, pi_vals[i] * 2000)   # tamanho proporcional a π
    ax.scatter(mu_m[i, 0], mu_m[i, 1],
               c=[colors[i]], s=size, alpha=0.9, zorder=3,
               label=f'E{i+1}: π={pi_vals[i]*100:.1f}%  μ=({mu_m[i,0]:.2f},{mu_m[i,1]:.2f})')
    # Elipse de 1σ
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(mu_m[i,0] + sig_m[i,0]*np.cos(theta),
            mu_m[i,1] + sig_m[i,1]*np.sin(theta),
            color=colors[i], alpha=0.4, linewidth=1.5)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Todos os experts — μ e elipse 1σ')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Direita — amostras da mistura completa
ax2 = axes[1]
ax2.scatter(positions[:, 0], positions[:, 1],
            c='black', marker='+', s=60, alpha=0.4, label='GT', zorder=2)
n_samples = 200
all_samples = []
for _ in range(n_samples):
    # Escolhe expert via amostragem de π
    k_chosen = np.random.choice(K, p=pi_vals)
    s_x = np.random.normal(mu_m[k_chosen, 0], sig_m[k_chosen, 0])
    s_y = np.random.normal(mu_m[k_chosen, 1], sig_m[k_chosen, 1])
    all_samples.append((s_x, s_y, k_chosen))

for i in range(K):
    pts = [(s[0], s[1]) for s in all_samples if s[2] == i]
    if pts:
        pts = np.array(pts)
        ax2.scatter(pts[:, 0], pts[:, 1],
                    c=[colors[i]], s=20, alpha=0.5,
                    label=f'E{i+1} ({len(pts)} samples)')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_title(f'Amostragem da mistura completa (N={n_samples})')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

plt.suptitle('Distribuição MDN — todos os experts — bin vazio', fontsize=12)
plt.tight_layout()
plt.savefig('all_experts_distribution.png', bbox_inches='tight', dpi=150)
plt.show()