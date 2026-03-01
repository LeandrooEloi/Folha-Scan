import cv2
import numpy as np
import os

def segmentar_folha(img_original):
    """
    Recebe uma imagem (OpenCV) e retorna:
    - A máscara binária (Preto e Branco)
    - A imagem recortada (Fundo preto)
    """
    # 1. Pré-processamento: Filtro Gaussiano para reduzir ruído (suavizar)
    img_blur = cv2.GaussianBlur(img_original, (5, 5), 0)

    # 2. Converter para HSV (Melhor para detectar cores)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    # 3. Definir o intervalo da cor VERDE (Ajuste esses valores se precisar)
    # H (Matiz): 35 a 90 pega bem verdes amarelados até verdes escuros
    lower_green = np.array([30, 40, 40])  
    upper_green = np.array([90, 255, 255])

    # 4. Criar a Máscara (Limiarização)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 5. Morfologia Matemática (Limpeza)
    kernel = np.ones((5, 5), np.uint8)
    # Fechamento (Closing): Tapa buracos dentro da folha
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Abertura (Opening): Remove ruídos (pontinhos) fora da folha
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 6. Selecionar apenas o MAIOR contorno (A Folha)
    # Isso evita pegar a moeda ou sujeiras grandes
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask_final = np.zeros_like(mask) # Começa com tudo preto
    
    if contornos:
        # Pega o maior contorno por área
        maior_contorno = max(contornos, key=cv2.contourArea)
        # Desenha ele preenchido de branco na máscara final
        cv2.drawContours(mask_final, [maior_contorno], -1, 255, thickness=cv2.FILLED)
    
    # 7. Aplicar a máscara na imagem original (Recorte)
    resultado = cv2.bitwise_and(img_original, img_original, mask=mask_final)

    return mask_final, resultado

# --- BLOCO DE TESTE (Para rodar agora no PC) ---
if __name__ == "__main__":
    import glob
    import os

    # pega qualquer jpg dentro de banco_fotos
    arquivos = glob.glob(os.path.join("banco_fotos", "*.jpg"))

    if not arquivos:
        print("Nenhuma imagem .jpg encontrada em banco_fotos")
    else:
        nome_imagem = arquivos[0]   # pega a primeira só para testar
        print(f"Usando a imagem: {nome_imagem}")

        img = cv2.imread(nome_imagem)
        mask, final = segmentar_folha(img)

        cv2.imshow("Original", img)
        cv2.imshow("Mascara (Binaria)", mask)
        cv2.imshow("Resultado (Sem Fundo)", final)

        print("Pressione qualquer tecla para fechar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
