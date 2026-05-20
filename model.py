# ─────────────────────────────────────────────
# Fake News Detector - Model Training
# NLP + TF-IDF + Logistic Regression
# Author: Rushil Popat
# ─────────────────────────────────────────────

import pandas as pd
import numpy as np
import pickle
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

# ── 1. Load Dataset ──────────────────────────
# Download from Kaggle: https://www.kaggle.com/c/fake-news/data
# Place Fake.csv and True.csv in the data/ folder
# OR use the generate_sample_data() function below for testing

def generate_sample_data():
    """
    Generates a small sample dataset for testing without Kaggle download.
    For production accuracy, replace with the real Kaggle dataset.
    """
    real_headlines = [
        "Scientists discover new treatment for Alzheimer's disease",
        "Government announces new budget plan for infrastructure",
        "Stock markets reach record highs amid economic recovery",
        "Climate summit concludes with major carbon reduction pledges",
        "NASA successfully launches new Mars exploration mission",
        "World Health Organization updates COVID-19 guidelines",
        "Tech giant reports strong quarterly earnings growth",
        "New renewable energy project creates thousands of jobs",
        "Supreme Court rules on landmark immigration case",
        "Global food prices rise due to supply chain disruptions",
        "Central bank raises interest rates to combat inflation",
        "Electric vehicle sales hit record levels worldwide",
        "International trade agreement signed by 30 nations",
        "Researchers develop new vaccine for tropical diseases",
        "City council approves major public transport expansion",
        "Olympic committee announces host city for 2032 games",
        "New study links air pollution to cognitive decline",
        "Government launches digital literacy program for seniors",
        "Hospital introduces AI system for early cancer detection",
        "Major airline announces new international routes",
        "Scientists confirm record-breaking ocean temperatures",
        "New education policy to improve rural school funding",
        "Diplomatic talks resume between rival nations",
        "Engineers complete construction of longest suspension bridge",
        "Health ministry reports decline in childhood obesity rates",
        "Technology companies commit to net-zero emissions by 2040",
        "Agricultural ministry reports bumper wheat harvest this year",
        "United Nations calls for immediate ceasefire in conflict zone",
        "University researchers publish breakthrough in quantum computing",
        "Finance minister presents balanced budget to parliament",
    ]

    fake_headlines = [
        "SHOCKING: Government secretly putting chemicals in drinking water",
        "EXPOSED: Celebrities are actually lizard people in disguise",
        "Doctor reveals miracle cure that big pharma doesn't want you to know",
        "BREAKING: Moon landing was filmed in a Hollywood studio confirmed",
        "Secret society controls all world governments, whistleblower says",
        "5G towers proven to cause cancer, government covers up evidence",
        "Scientist claims earth is actually flat, NASA hiding the truth",
        "Bill Gates microchip found inside COVID vaccine by researcher",
        "ALERT: Aliens living among us government confirms in leaked file",
        "Miracle weight loss pill doctors don't want you to know about",
        "EXPOSED: Election results manipulated by foreign supercomputer",
        "Secret underground city discovered beneath the White House",
        "Ancient scrolls prove historical figures were time travelers",
        "SHOCKING PROOF: Climate change is a hoax invented by globalists",
        "Mind control device found hidden inside popular smartphone model",
        "World leaders meet secretly to plan global population reduction",
        "New law will allow government to read all your private messages",
        "Famous actor reveals he faked his own death ten years ago",
        "Scientists admit they have been hiding cure for all cancers",
        "BREAKING: Major bank about to collapse, insiders confirm panic",
        "Leaked document shows moon is actually a giant alien spacecraft",
        "Government fluoride program linked to massive IQ reduction plot",
        "Celebrity admits to being part of secret illuminati organization",
        "Shocking study reveals smartphones cause brain tumors in children",
        "Whistleblower exposes top secret time travel program at Pentagon",
        "Ancient aliens built the pyramids, government finally admits it",
        "New world order planning to replace all cash with digital control",
        "Secret ingredient in fast food causes addiction by design always",
        "EXPOSED: Weather control machines used to create natural disasters",
        "Hidden camera footage proves voting machines are pre-programmed",
    ]

    real_df = pd.DataFrame({
        'text': real_headlines,
        'label': [1] * len(real_headlines)   # 1 = REAL
    })
    fake_df = pd.DataFrame({
        'text': fake_headlines,
        'label': [0] * len(fake_headlines)   # 0 = FAKE
    })

    df = pd.concat([real_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def load_kaggle_data():
    """Load real Kaggle dataset if available."""
    try:
        fake = pd.read_csv("data/Fake.csv")
        true = pd.read_csv("data/True.csv")
        fake['label'] = 0
        true['label'] = 1
        df = pd.concat([fake[['text', 'label']], true[['text', 'label']]])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Kaggle dataset loaded: {len(df)} records")
        return df
    except FileNotFoundError:
        print("Kaggle dataset not found. Using sample data for testing.")
        return generate_sample_data()


# ── 2. Text Preprocessing ────────────────────
def clean_text(text):
    """
    Clean raw text:
    - Lowercase
    - Remove punctuation
    - Remove extra whitespace
    """
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)          # remove text in brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # remove punctuation
    text = re.sub(r'\n', ' ', text)              # remove newlines
    text = re.sub(r'\s+', ' ', text).strip()     # remove extra spaces
    return text


# ── 3. Train Model ───────────────────────────
def train():
    # Load data
    df = load_kaggle_data()
    print(f"Dataset: {len(df)} articles | Real: {df['label'].sum()} | Fake: {(df['label']==0).sum()}")

    # Clean text
    df['text_clean'] = df['text'].apply(clean_text)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['label'],
        test_size=0.2, random_state=42
    )

    # TF-IDF Vectorization
    # max_features=5000: use top 5000 most frequent words
    # ngram_range=(1,2): use single words AND pairs of words
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    print(f"TF-IDF Matrix shape: {X_train_tfidf.shape}")

    # Logistic Regression Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred   = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n── Model Evaluation ──")
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

    # Save model and vectorizer
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    print("✅ model.pkl saved")
    print("✅ tfidf.pkl saved")
    print("\nTraining complete!")


if __name__ == "__main__":
    train()
