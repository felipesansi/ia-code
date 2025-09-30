import os
import joblib
from flask import Flask, request, jsonify

MODEL_PATH = "best_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"

def load_artifacts():
    if not all(os.path.exists(p) for p in [MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH]):
        print("ERRO: Artefatos do modelo não encontrados.")
        return None, None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    return model, vectorizer, le

app = Flask(__name__)
model, vectorizer, le = load_artifacts()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    codigo = data.get("code", "")
    if not codigo:
        return jsonify({"error": "Nenhum código fornecido"}), 400
    try:
        codigo_transformado = vectorizer.transform([codigo.lower()])
        predicao = model.predict(codigo_transformado)
        linguagem_predita = le.inverse_transform(predicao)[0].capitalize()
        return jsonify({"best_prediction": linguagem_predita})
    except Exception as e:
        return jsonify({"error": "Erro interno na predição"}), 500
