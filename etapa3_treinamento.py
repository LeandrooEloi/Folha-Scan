import cv2
import numpy as np
import os
import glob
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from segmentacao import segmentar_folha  # Reutiliza sua função da etapa 2

def extrair_descritores(img):
    """
    Extrai características da imagem segmentada: Hu Moments e Média de Cor HSV
    """
    # Segmentar folha
    mask, img_segmentada = segmentar_folha(img)
    
    # 1. Hu Moments (Forma)
    gray = cv2.cvtColor(img_segmentada, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray, binaryImage=True)
    hu_moments = cv2.HuMoments(moments).flatten()
    # Log transform para escala melhor
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    # 2. Cor Média (HSV)
    hsv = cv2.cvtColor(img_segmentada, cv2.COLOR_BGR2HSV)
    mean_val = cv2.mean(hsv, mask=mask)[:3]
    
    # Junta tudo (10 features)
    features = np.hstack([hu_moments, mean_val])
    return features

def listar_imagens(pasta):
    """
    Busca imagens JPG, JPEG e PNG recursivamente na pasta
    """
    extensoes = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
    lista_arquivos = []
    for ext in extensoes:
        # recursive=True permite buscar em subpastas
        encontrados = glob.glob(os.path.join(pasta, "**", ext), recursive=True)
        lista_arquivos.extend(encontrados)
    return lista_arquivos

# --- CONFIGURAÇÃO ---
PASTA_POSITIVA = "banco_fotos"      # Suas fotos
PASTA_NEGATIVA = "Folhas_512x512"   # Fotos de outras plantas

print("--- INICIANDO TREINAMENTO ETAPA 3 ---")

dados_X = []
labels_y = []

# 1. Processar Classe POSITIVA (Label 1)
print(f"Lendo pasta positiva: {PASTA_POSITIVA}...")
arquivos_pos = listar_imagens(PASTA_POSITIVA)

if not arquivos_pos:
    print(f"AVISO: Nenhuma imagem encontrada em {PASTA_POSITIVA}")

for arq in arquivos_pos:
    img = cv2.imread(arq)
    if img is not None:
        try:
            feats = extrair_descritores(img)
            dados_X.append(feats)
            labels_y.append(1)
        except Exception as e:
            pass

# 2. Processar Classe NEGATIVA (Label 0)
print(f"Lendo pasta negativa: {PASTA_NEGATIVA}...")
arquivos_neg = listar_imagens(PASTA_NEGATIVA)

if not arquivos_neg:
    print(f"ERRO: Nenhuma imagem encontrada em {PASTA_NEGATIVA}! Verifique se a pasta existe.")
else:
    print(f"Encontradas {len(arquivos_neg)} imagens negativas.")
    
    for arq in arquivos_neg:
        img = cv2.imread(arq)
        if img is not None:
            try:
                feats = extrair_descritores(img)
                dados_X.append(feats)
                labels_y.append(0)
            except Exception as e:
                pass

    # Treinamento
    if len(dados_X) > 10: # Garante um mínimo de dados
        dados_X = np.array(dados_X)
        labels_y = np.array(labels_y)
        
        # Dividir Treino e Teste (80% / 20%)
        X_train, X_test, y_train, y_test = train_test_split(dados_X, labels_y, test_size=0.2, random_state=42)
        
        print("Treinando modelo Random Forest...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Avaliar
        if len(X_test) > 0:
            preds = clf.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print(f"\n--- RESULTADOS ---")
            print(f"Acurácia no Teste: {acc*100:.2f}%")
            print(classification_report(y_test, preds, target_names=["Outra", "Syzygium"], zero_division=0))
        
        # SALVAR O MODELO
        with open("modelo_folhas.pkl", "wb") as f:
            pickle.dump(clf, f)
        print("Modelo salvo com sucesso em: modelo_folhas.pkl")
        
    else:
        print("Sem dados suficientes para treinar. Preciso de imagens nas duas pastas.")
