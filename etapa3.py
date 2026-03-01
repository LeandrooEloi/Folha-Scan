# ========================
# ETAPA 3 – CLASSIFICAÇÃO (versão otimizada e robusta)
# ========================

import cv2
import numpy as np
import pandas as pd
import glob
from pathlib import Path

# Bibliotecas extras
import albumentations as A
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import skew, kurtosis

# ======================================================================
#  FUNÇÕES AUXILIARES
# ======================================================================

def get_contour(mask):
    """Retorna o maior contorno da imagem binária."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea) if contours else None


# ======================================================================
# 1) DESCRITORES DE FRONTEIRA
# ======================================================================

def extract_boundary_features(mask):
    """Extrai 15 descritores baseados no contorno."""
    cnt = get_contour(mask)
    if cnt is None:
        return None

    area = float(cv2.contourArea(cnt))
    perimeter = max(float(cv2.arcLength(cnt, True)), 1e-6)

    # Bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h != 0 else 0

    # Circularidade + compacidade
    circularity = (4 * np.pi * area) / (perimeter * perimeter)
    compactness = (perimeter * perimeter) / (4 * np.pi * area) if area != 0 else 0

    # Momentos de Hu (7)
    hu = cv2.HuMoments(cv2.moments(cnt)).flatten().astype(float).tolist()

    # Fourier robusto
    try:
        cnt_flat = cnt.reshape(-1, 2)
        cnt_complex = cnt_flat[:, 0].astype(float) + 1j * cnt_flat[:, 1].astype(float)
        fourier = np.abs(np.fft.fft(cnt_complex)[:4]).astype(float).tolist()
    except:
        fourier = [0.0] * 4

    features = [area, perimeter, circularity, aspect_ratio, compactness] + hu + fourier

    # normalizar para garantir 15 valores
    if len(features) < 15:
        features += [0] * (15 - len(features))

    return features[:15]


# ======================================================================
# 2) DESCRITORES DE REGIÃO
# ======================================================================

def safe_glcm(crop):
    """Retorna GLCM seguro."""
    try:
        if crop.size < 4:
            return 0, 0, 0, 0
        gl = graycomatrix(crop, [2], [0], 256, symmetric=True, normed=True)
        return (float(graycoprops(gl, 'contrast')[0, 0]),
                float(graycoprops(gl, 'energy')[0, 0]),
                float(graycoprops(gl, 'homogeneity')[0, 0]),
                float(graycoprops(gl, 'correlation')[0, 0]))
    except:
        return 0, 0, 0, 0


def extract_region_features(mask, img_gray):
    """Extrai 15 descritores de textura e região."""
    cnt = get_contour(mask)
    if cnt is None:
        return None

    area = float(cv2.contourArea(cnt))
    perimeter = max(float(cv2.arcLength(cnt, True)), 1e-6)

    hull = cv2.convexHull(cnt)
    area_conv = float(cv2.contourArea(hull))
    solidity = area / area_conv if area_conv != 0 else 0

    x, y, w, h = cv2.boundingRect(cnt)
    crop = img_gray[y:y+h, x:x+w]
    crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    mean_int = float(np.mean(crop))
    var_int = float(np.var(crop))
    ent = float(shannon_entropy(crop))

    contrast, energy, homog, corr = safe_glcm(crop)

    flat = crop.flatten().astype(float)
    skewness = float(skew(flat)) if flat.size > 2 else 0
    kurtv = float(kurtosis(flat)) if flat.size > 2 else 0

    aspect_ratio = w / h if h != 0 else 0

    features = [
        area, area_conv, solidity, mean_int, var_int, ent,
        contrast, energy, homog, corr,
        skewness, kurtv, area / perimeter, aspect_ratio, perimeter
    ]

    return features


# ======================================================================
# 3) DATA AUGMENTATION
# ======================================================================

aug = A.Compose([
    A.Rotate(limit=20, p=0.8),
    A.RandomBrightnessContrast(p=0.6),
    A.HorizontalFlip(p=0.5),
    A.Blur(blur_limit=3, p=0.4)
])


def apply_augmentation(img, mask):
    out = aug(image=img, mask=mask)
    return out["image"], out["mask"]


# ======================================================================
# 4) CARREGAMENTO E EXTRAÇÃO DE FEATURES
# ======================================================================

paths = [
    (r"C:\Users\Dell\Documents\pdi\resultados\*.jpg", +1),
    (r"C:\Users\Dell\Documents\pdi\Folhas_512x512\*.png", -1)
]

boundary_data = []
region_data = []

print("\nProcessando imagens...\n")

for pattern, classe in paths:
    files = glob.glob(pattern)
    print(f"Encontrados {len(files)} arquivos em: {pattern}")

    for file in files:
        img = cv2.imread(file)
        if img is None:
            print(f"Erro ao abrir {file}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        fb = extract_boundary_features(mask)
        fr = extract_region_features(mask, gray)

        if fb is None or fr is None:
            continue

        img_id = Path(file).name
        boundary_data.append([img_id] + fb + [classe])
        region_data.append([img_id] + fr + [classe])

        # aumentar 5x a base
        for i in range(5):
            a_img, a_mask = apply_augmentation(img, mask)
            fb2 = extract_boundary_features(a_mask)
            fr2 = extract_region_features(a_mask, cv2.cvtColor(a_img, cv2.COLOR_BGR2GRAY))

            if fb2 is None or fr2 is None:
                continue

            boundary_data.append([f"{img_id}_aug{i}"] + fb2 + [classe])
            region_data.append([f"{img_id}_aug{i}"] + fr2 + [classe])

print("\nExtração concluída.\n")

dfb = pd.DataFrame(boundary_data, columns=["id"]+[f"f{i}" for i in range(1,16)] + ["classe"])
dfr = pd.DataFrame(region_data,   columns=["id"]+[f"f{i}" for i in range(1,16)] + ["classe"])

dfb.to_csv("fronteira.csv", index=False)
dfr.to_csv("regiao.csv", index=False)

print("Arquivos salvos: fronteira.csv, regiao.csv")


# ======================================================================
# 5) TREINANDO O MODELO – COM GRIDSEARCH
# ======================================================================

print("\nTreinando modelo com GridSearchCV...")

X = dfb.iloc[:, 1:-1]
y = dfb["classe"]

params = {
    "C": [1, 10, 100],
    "gamma": [0.001, 0.01, 0.1],
    "kernel": ["rbf"]
}

grid = GridSearchCV(SVC(), params, cv=5, verbose=1)
grid.fit(X, y)

best = grid.best_estimator_

print("\nMelhores parâmetros encontrados:")
print(grid.best_params_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

best.fit(X_train, y_train)
pred = best.predict(X_test)

print("\n===== RELATÓRIO FINAL =====")
print(classification_report(y_test, pred))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, pred))

import joblib
joblib.dump(best, "modelo_folhas_svm.joblib")
