import cv2
import os
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import csv
import shutil


@dataclass
class PilhaInfo:
    centro: Tuple[int, int]
    raio: int
    pixels_por_mm: float


@dataclass
class FolhaInfo:
    contorno: np.ndarray
    mascara: np.ndarray
    altura_px: float
    largura_px: float
    altura_cm: float
    largura_cm: float
    area_px: float
    sucesso: bool
    orientacao: str = 'baixo'
    erro: Optional[str] = None


@dataclass
class SegmentadorFolhas:

    DIAMETRO_PILHA = 20.0  # CR2032

    def __init__(self):
        self.resultados = []

    def detectar_pilha(self, imagem: np.ndarray) -> Optional[PilhaInfo]:
        hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

        # Segmentar FUNDO AZUL com saturacao otimizada
        lower_blue = np.array([90, 60, 50])  # Saturacao: 60
        upper_blue = np.array([130, 255, 255])
        mask_fundo = cv2.inRange(hsv, lower_blue, upper_blue)

        # INVERTER
        mask_objetos = cv2.bitwise_not(mask_fundo)

        # Aplicar blur para suavizar
        mask_blurred = cv2.GaussianBlur(mask_objetos, (9, 9), 2)

        # Detectar circulos
        circles = cv2.HoughCircles(
            mask_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=25,
            minRadius=15,
            maxRadius=100
        )

        if circles is None:
            return None

        circles = np.uint16(np.around(circles))

        melhor_circulo = None
        melhor_score = -999999

        for circle in circles[0, :]:
            x, y, r = int(circle[0]), int(circle[1]), int(circle[2])

            if x - r < 0 or y - r < 0 or x + r >= imagem.shape[1] or y + r >= imagem.shape[0]:
                continue

            # Criar mascara do circulo
            mask_circulo = np.zeros(mask_objetos.shape, dtype=np.uint8)
            cv2.circle(mask_circulo, (x, y), r, 255, -1)

            # Verificar densidade de pixels brancos
            area_circulo = np.pi * r * r
            pixels_brancos = cv2.countNonZero(cv2.bitwise_and(mask_objetos, mask_circulo))
            densidade = pixels_brancos / area_circulo

            # Densidade mínima ajustada
            if densidade < 0.6:
                continue

            # Verificar circularidade
            contours_temp, _ = cv2.findContours(
                cv2.bitwise_and(mask_objetos, mask_circulo),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours_temp:
                continue

            maior_contorno = max(contours_temp, key=cv2.contourArea)
            perimetro = cv2.arcLength(maior_contorno, True)
            if perimetro == 0:
                continue

            area_contorno = cv2.contourArea(maior_contorno)
            circularidade = (4 * np.pi * area_contorno) / (perimetro * perimetro)

            if circularidade < 0.6:
                continue

            tamanho_ideal = 95
            penalidade_tamanho = 1.0 / (1.0 + abs(r - tamanho_ideal) / 15.0)

            score = densidade * circularidade * penalidade_tamanho * 100

            if score > melhor_score:
                melhor_score = score
                melhor_circulo = circle

        if melhor_circulo is None:
            return None

        x, y, r = melhor_circulo
        pixels_por_mm = (2 * r) / self.DIAMETRO_PILHA

        return PilhaInfo(
            centro=(int(x), int(y)),
            raio=int(r),
            pixels_por_mm=pixels_por_mm
        )

    def segmentar_folha(self, imagem: np.ndarray, pilha_info: Optional[PilhaInfo] = None) -> FolhaInfo:
        try:
            hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

            # Segmentar FUNDO AZUL
            lower_blue = np.array([90, 60, 50])
            upper_blue = np.array([130, 255, 255])

            # mascara do fundo azul e inverter
            mask_fundo = cv2.inRange(hsv, lower_blue, upper_blue)
            mask = cv2.bitwise_not(mask_fundo)

            # Operacoes morfologicas
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)

            # Suavizacao
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Remover pilha
            if pilha_info:
                cv2.circle(mask, pilha_info.centro, pilha_info.raio + 10, 0, -1)

            # Contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return FolhaInfo(
                    contorno=None,
                    mascara=mask,
                    altura_px=0, largura_px=0,
                    altura_cm=0, largura_cm=0,
                    area_px=0,
                    sucesso=False,
                    orientacao='baixo',
                    erro="Nenhum contorno encontrado"
                )

            folha_contorno = max(contours, key=cv2.contourArea)
            area_px = cv2.contourArea(folha_contorno)

            altura_px, largura_px = self.calcular_dimensoes(folha_contorno)

            if pilha_info:
                altura_cm = (altura_px / pilha_info.pixels_por_mm) / 10
                largura_cm = (largura_px / pilha_info.pixels_por_mm) / 10
            else:
                altura_cm = 0
                largura_cm = 0

            return FolhaInfo(
                contorno=folha_contorno,
                mascara=mask,
                altura_px=altura_px,
                largura_px=largura_px,
                altura_cm=altura_cm,
                largura_cm=largura_cm,
                area_px=area_px,
                sucesso=True,
            )

        except Exception as e:
            return FolhaInfo(
                contorno=None,
                mascara=None,
                altura_px=0, largura_px=0,
                altura_cm=0, largura_cm=0,
                area_px=0,
                sucesso=False,
                orientacao='baixo',
                erro=f"Erro no processamento: {str(e)}"
            )

    def calcular_dimensoes(self, contorno: np.ndarray) -> Tuple[float, float]:
        rect = cv2.minAreaRect(contorno)
        dimensoes = rect[1]
        altura = max(dimensoes)
        largura = min(dimensoes)
        return altura, largura


def main():
    # (o resto do código de processamento em lote fica igual)
    ...
    # não mexemos aqui porque serve para seus testes em lote


if __name__ == "__main__":
    main()
