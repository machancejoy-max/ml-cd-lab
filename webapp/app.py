print("RUNNING THIS FILE:", __file__)

import os
import torch
from flask import Flask, request, jsonify
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.serialization

app = Flask(__name__)

MODEL_PATH = "model_full.pt"
TOKENIZER_PATH = "tokenizer"

def load_model():
    print("--------------------------------------------------")
    print("Loading PyTorch model from:", MODEL_PATH)

    # Allowlist the Roberta class for safe unpickling (PyTorch 2.6 requirement)
    torch.serialization.add_safe_globals([RobertaForSequenceClassification])

    if not os.path.exists(MODEL_PATH):
        print("[ERROR] model_full.pt not found at:", MODEL_PATH)
        return None

    try:
        model = torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu"),
            weights_only=False  # IMPORTANT for PyTorch 2.6
        )
        model.eval()
        print("Model loaded successfully.")
        return model

    except Exception as e:
        print("[ERROR] Failed to load model:", e)
        return None


def load_tokenizer():
    print("Loading tokenizer from:", TOKENIZER_PATH)

    if not os.path.exists(TOKENIZER_PATH):
        print("[ERROR] tokenizer folder not found at:", TOKENIZER_PATH)
        return None

    try:
        tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH)
        print("Tokenizer loaded successfully.")
        return tokenizer

    except Exception as e:
        print("[ERROR] Failed to load tokenizer:", e)
        return None


# Load model + tokenizer
model = load_model()
tokenizer = load_tokenizer()

# If either fails, stop the app
if model is None or tokenizer is None:
    print("[ERROR] Application startup failed. Fix the errors above.")
    exit(1)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()

        labels = ["negative", "neutral", "positive"]
        result = {"label": labels[prediction]}

        print("Prediction:", result)  # Log to console
        return jsonify(result)

    except Exception as e:
        print("[ERROR] Prediction failed:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Flask app on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
