from flask import Flask, request, jsonify, render_template
import os, joblib

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

MODEL_PATH = "../best_model.joblib"
VECTORIZER_PATH = "../vectorizer.joblib"
LABEL_ENCODER_PATH = "../label_encoder.joblib"

def load_artifacts():
    if not all(os.path.exists(p) for p in [MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH]):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    return model, vectorizer, le

model, vectorizer, le = load_artifacts()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    code = data.get("code", "")
    if not code:
        return jsonify({"error": "Código vazio"}), 400
    try:
        x = vectorizer.transform([code.lower()])
        pred = model.predict(x)
        lang = le.inverse_transform(pred)[0].capitalize()
        return jsonify({"best_prediction": lang})
    except Exception as e:
        return jsonify({"error": "Erro interno na predição"}), 500
