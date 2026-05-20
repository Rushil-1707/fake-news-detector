# ─────────────────────────────────────────────
# Fake News Detector - Flask Web App
# Author: Rushil Popat
# ─────────────────────────────────────────────

from flask import Flask, render_template, request
import pickle
import re
import string

app = Flask(__name__)

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.route("/")
def home():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news_text", "").strip()

    if not news_text:
        return render_template("index.html", result=None,
                               error="Please enter some news text.")

    if len(news_text.split()) < 5:
        return render_template("index.html", result=None,
                               error="Please enter at least 5 words for accurate prediction.")

    # Preprocess
    cleaned = clean_text(news_text)

    # Vectorize
    vectorized = tfidf.transform([cleaned])

    # Predict
    prediction  = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]

    label      = "REAL" if prediction == 1 else "FAKE"
    confidence = round(max(probability) * 100, 1)
    fake_prob  = round(probability[0] * 100, 1)
    real_prob  = round(probability[1] * 100, 1)

    result = {
        "label":      label,
        "confidence": confidence,
        "fake_prob":  fake_prob,
        "real_prob":  real_prob,
        "text":       news_text[:200] + ("..." if len(news_text) > 200 else ""),
    }

    return render_template("index.html", result=result, error=None)


if __name__ == "__main__":
    app.run(debug=True)
