# classificacao_folhas.py

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis
import joblib

# carrega o modelo SVM treinado com descritores de fronteira
modelo = joblib.load("modelo_folhas_svm.joblib")


def get_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea) if contours else None


def extract_boundary_features(mask):
    cnt = get_contour(mask)
    if cnt is None:
        return None

    area = float(cv2.contourArea(cnt))
    perimeter = max(float(cv2.arcLength(cnt, True)), 1e-6)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h != 0 else 0

    circularity = (4 * np.pi * area) / (perimeter * perimeter)
    compactness = (perimeter * perimeter) / (4 * np.pi * area) if area != 0 else 0

    hu = cv2.HuMoments(cv2.moments(cnt)).flatten().astype(float).tolist()

    try:
        cnt_flat = cnt.reshape(-1, 2)
        cnt_complex = cnt_flat[:, 0].astype(float) + 1j * cnt_flat[:, 1].astype(float)
        fourier = np.abs(np.fft.fft(cnt_complex)[:4]).astype(float).tolist()
    except:
        fourier = [0.0] * 4

    features = [area, perimeter, circularity, aspect_ratio, compactness] + hu + fourier

    if len(features) < 15:
        features += [0] * (15 - len(features))

    return features[:15]


def classificar_imagem(img_bgr):
    """
    Recebe a imagem BGR original, gera máscara igual à etapa 3,
    extrai descritores de fronteira e usa o SVM treinado.
    Retorna (pertence_especie: bool, confianca: float 0–1).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    features = extract_boundary_features(mask)
    if features is None:
        # se der ruim, retorna baixa confiança e não pertence
        return False, 0.0

    X = [features]
    pred = modelo.predict(X)[0]

    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(X)[0]
        confianca = float(max(proba))
    else:
        # sem probabilidades, usa confiança média
        confianca = 0.8

    # no seu treinamento: +1 = espécie-alvo, -1 = outra espécie
    pertence = (pred == 1)

    return pertence, confianca
