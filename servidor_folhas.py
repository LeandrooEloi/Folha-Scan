from flask import Flask, request, jsonify
import cv2
import numpy as np

from SegmentadorFolhas import SegmentadorFolhas

app = Flask(__name__)
segmentador = SegmentadorFolhas()


@app.route("/analisar_folha", methods=["POST"])
def analisar_folha():
    if "imagem" not in request.files:
        return jsonify({"erro": "imagem não enviada"}), 400

    file = request.files["imagem"]
    bytes_imagem = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(bytes_imagem, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"erro": "imagem inválida"}), 400

    # 1) Medir com SegmentadorFolhas (pilha + folha)
    pilha = segmentador.detectar_pilha(img)
    folha = segmentador.segmentar_folha(img, pilha)

    if not folha.sucesso:
        return jsonify({"erro": folha.erro}), 400

    altura = float(folha.altura_cm)
    largura = float(folha.largura_cm)

    # 2) REGRA SIMPLES: folhas da espécie ~20–35 cm de altura e 7–13 cm de largura
    if 20 <= altura <= 35 and 7 <= largura <= 13:
        pertence_especie = True
        confianca = 0.8
    else:
        pertence_especie = False
        confianca = 0.5

    return jsonify({
        "especie_alvo": "Syzygium malaccense",
        "pertence_especie": bool(pertence_especie),
        "confianca": float(confianca),
        "altura_cm": altura,
        "largura_cm": largura,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
