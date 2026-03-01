import cv2
import os
import glob
from segmentacao import segmentar_folha  # usa a função que você já tem

pasta = "banco_fotos"
padrao = os.path.join(pasta, "*.jpg")   # ajuste para *.png se precisar
arquivos = glob.glob(padrao)

sucessos = 0
falhas = 0

print(f"Processando {len(arquivos)} imagens da pasta {pasta}...\n")

for caminho in arquivos:
    img = cv2.imread(caminho)
    if img is None:
        print(f"[AVISO] Não consegui ler: {caminho}")
        continue

    mask, resultado = segmentar_folha(img)

    area_branca = cv2.countNonZero(mask)
    area_total = mask.shape[0] * mask.shape[1]

    # critério simples de sucesso: pelo menos 5% da área como folha
    if area_branca > area_total * 0.05:
        sucessos += 1
    else:
        falhas += 1
        print(f"[FALHA] Segmentação fraca em: {caminho}")

print("\n--- RELATÓRIO ETAPA 2 ---")
print(f"Total de imagens: {len(arquivos)}")
print(f"Sucessos (segmentou a folha): {sucessos}")
print(f"Falhas: {falhas}")
