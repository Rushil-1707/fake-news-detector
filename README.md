# 📰 Fake News Detector

An NLP-based fake news detection web app using TF-IDF Vectorization and Logistic Regression.

## Tech Stack
- **Python** – Core language
- **Scikit-learn** – TF-IDF + Logistic Regression
- **Pandas & NumPy** – Data processing
- **Flask** – Web framework
- **HTML/CSS** – Frontend UI

## Dataset
- **Sample mode:** Built-in sample data (works out of the box)
- **Full mode:** Kaggle Fake News Dataset (40,000+ articles)
  - Download from: https://www.kaggle.com/c/fake-news/data
  - Place `Fake.csv` and `True.csv` in the `data/` folder

## ML Pipeline
1. Load and label dataset (0 = Fake, 1 = Real)
2. Clean text (lowercase, remove URLs, punctuation, stopwords)
3. TF-IDF Vectorization (top 5000 features, unigrams + bigrams)
4. Train Logistic Regression model
5. Evaluate with accuracy, precision, recall, F1
6. Save model with Pickle

## Model Performance (Kaggle dataset)
- Accuracy: ~93%
- Strong F1 score for both FAKE and REAL classes

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python model.py
```

### 3. Run the web app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

## Project Structure
```
fake-news-detector/
├── model.py              # Train and save ML model
├── app.py                # Flask web application
├── data/                 # Place Kaggle CSVs here
├── templates/
│   └── index.html        # Web UI
├── requirements.txt
└── README.md
```

## Points
- TF-IDF: assigns higher weight to rare but important words
- Logistic Regression: outputs probability (0-1) for binary classification
- Text preprocessing: lowercase, remove URLs, punctuation, stopwords
- predict_proba() used to show confidence percentage

## Author
Rushil Popat | 23IT102 | CSPIT, CHARUSAT
