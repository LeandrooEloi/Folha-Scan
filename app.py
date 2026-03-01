import streamlit as st
import cv2
import numpy as np
import pickle
from segmentacao import segmentar_folha
from etapa3_treinamento import extrair_descritores

# --- CARREGAR O MODELO TREINADO ---
try:
    with open("modelo_folhas.pkl", "rb") as f:
        clf = pickle.load(f)
    MODELO_CARREGADO = True
except:
    MODELO_CARREGADO = False

# --- FUNÇÕES DA ETAPA 4 (MEDIÇÃO E ROTAÇÃO) ---
def alinhar_e_medir(img_original, mask_folha):
    # Encontra o contorno da folha
    contornos, _ = cv2.findContours(mask_folha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return img_original, 0, 0
    
    cnt = max(contornos, key=cv2.contourArea)
    
    # Rotação para alinhar (Fit Ellipse ou PCA)
    # Usando minAreaRect para achar o ângulo
    rect = cv2.minAreaRect(cnt)
    (x_center, y_center), (w_rect, h_rect), angulo = rect
    
    # Corrige ângulo para deixar em pé
    if w_rect < h_rect:
        angulo = angulo + 90
        
    M = cv2.getRotationMatrix2D((x_center, y_center), angulo, 1.0)
    h_img, w_img = img_original.shape[:2]
    img_rotacionada = cv2.warpAffine(img_original, M, (w_img, h_img))
    mask_rotacionada = cv2.warpAffine(mask_folha, M, (w_img, h_img))
    
    # Recalcula contorno na imagem rotacionada para medir altura/largura exatas
    contornos_rot, _ = cv2.findContours(mask_rotacionada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos_rot: return img_rotacionada, 0, 0
    
    cnt_rot = max(contornos_rot, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt_rot)
    
    # CALIBRAÇÃO (1 real = 27mm = 2.7cm)
    # Se tivéssemos a moeda detectada, usaríamos aqui. 
    # Como fallback, vamos assumir uma calibração fixa ou estimada se não tiver moeda.
    # Exemplo: Se a imagem for 4000px de altura, e a folha ocupou 2000px...
    # Ajuste este fator conforme sua câmera!
    PIXELS_POR_CM = 110.0 # <--- AJUSTE ISTO TESTANDO COM UMA RÉGUA
    
    altura_cm = h / PIXELS_POR_CM
    largura_cm = w / PIXELS_POR_CM
    
    # Desenhar linhas (Altura: Vermelho, Largura: Azul)
    # Altura (Centro X, de Y até Y+H)
    centro_x = x + w // 2
    cv2.line(img_rotacionada, (centro_x, y), (centro_x, y + h), (0, 0, 255), 5)
    
    # Largura (Centro Y, de X até X+W)
    centro_y = y + h // 2
    cv2.line(img_rotacionada, (x, centro_y), (x + w, centro_y), (255, 0, 0), 5)
    
    return img_rotacionada, altura_cm, largura_cm

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="FolhaScan", layout="centered")
st.title("🌿 FolhaScan")

if not MODELO_CARREGADO:
    st.error("ERRO: 'modelo_folhas.pkl' não encontrado! Rode o 'etapa3_treinamento.py' primeiro.")
else:
    opcao = st.radio("Entrada:", ("Câmera", "Upload"), horizontal=True)
    img_file = None

    if opcao == "Câmera":
        img_file = st.camera_input("Tirar foto")
    else:
        img_file = st.file_uploader("Escolher imagem", type=['jpg', 'png', 'jpeg'])

    if img_file:
        # Lê a imagem
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img_cv2 = cv2.imdecode(file_bytes, 1)
        
        with st.spinner('Processando...'):
            # 1. Segmentar
            mask, img_sem_fundo = segmentar_folha(img_cv2)
            
            # 2. Classificar
            feats = extrair_descritores(img_cv2)
            pred = clf.predict([feats])[0]
            proba = clf.predict_proba([feats])[0]
            
            nome_classe = "Syzygium malaccense" if pred == 1 else "Outra Espécie"
            confianca = proba[pred]
            
            # 3. Medir e Rotacionar
            img_final, alt, larg = alinhar_e_medir(img_cv2, mask)
            
            # Exibir Resultados
            st.divider()
            
            # Cor do status (Verde se for a planta, Vermelho se não for)
            cor_texto = "green" if pred == 1 else "red"
            st.markdown(f"<h2 style='color:{cor_texto}'>{nome_classe}</h2>", unsafe_allow_html=True)
            st.write(f"Confiança: **{confianca*100:.1f}%**")
            
            col1, col2 = st.columns(2)
            col1.metric("Altura", f"{alt:.1f} cm")
            col2.metric("Largura", f"{larg:.1f} cm")
            
            # Converter BGR -> RGB para mostrar no Streamlit
            img_show = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
            st.image(img_show, caption="Folha Analisada", use_container_width=True)
