import os
import joblib
from flask import Flask, request, jsonify, render_template

# --- Caminhos dos artefatos ---
MODEL_PATH = "best_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"

def load_artifacts():
    """Carrega os artefatos do modelo pré-treinado."""
    if not all(os.path.exists(p) for p in [MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH]):
        print("ERRO: Artefatos do modelo não encontrados.")
        return None, None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    return model, vectorizer, le

# --- Flask ---
app = Flask(__name__)
model, vectorizer, le = load_artifacts()

if model is not None:
    @app.route("/")
    def index():
        return render_template("index.html")

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
            print(f"Erro na predição: {e}")
            return jsonify({"error": "Erro interno na predição"}), 500
else:
    print("ERRO: Modelo não carregado. Verifique os artefatos.")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
