import trimesh

def escalar_glb(input_file, output_file, fator_escala):
    # 1. Carregar a cena ou malha
    # O trimesh carrega GLBs como uma 'Scene' (Cena)
    scene = trimesh.load(input_file)
    
    # 2. Aplicar a transformação de escala
    # Criamos uma matriz de escala 4x4
    matrix = trimesh.transformations.scale_matrix(fator_escala)
    
    # Aplicamos a matriz à geometria da cena
    scene.apply_transform(matrix)
    
    # 3. Exportar o ficheiro escalado
    # Certificamos que exportamos no formato glb
    scene.export(output_file)
    print(f"Sucesso! Ficheiro escalado por {fator_escala}x e guardado como: {output_file}")

# Exemplo de uso:
# Se o teu modelo mede 1 metro e queres que meça 10 metros, o fator é 10.
escalar_glb('GP23.glb', 'Carro_Grande.glb', 100.0)