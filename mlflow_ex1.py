# pip install mlflow scikit-learn
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import subprocess, requests, json

# ── SETUP ─────────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("imdb-sentiment-traditional")

# ── DATASET ───────────────────────────────────────────────────────────────────
# Using a small inline dataset to avoid needing to download IMDB.
# In production, replace with: datasets.load_dataset("imdb") via HuggingFace.
reviews = [
    "This movie was absolutely fantastic! I loved every minute.",
    "Brilliant performances and a gripping storyline. Highly recommend.",
    "One of the best films I have ever seen. A masterpiece.",
    "Great direction and superb acting. Will watch again.",
    "Incredible movie with a beautiful ending. Loved it.",
    "Terrible film. Boring, slow, and a complete waste of time.",
    "Awful acting and a nonsensical plot. I want my money back.",
    "One of the worst movies ever made. Do not watch.",
    "Dull and predictable. Fell asleep halfway through.",
    "Disappointing in every way. The script was a disaster.",
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1=positive, 0=negative

X_train, X_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.3, random_state=42
)

# ── CANDIDATE MODELS ──────────────────────────────────────────────────────────
# Each pipeline = TF-IDF vectorizer + classifier.
# TF-IDF converts raw text into numerical feature vectors.
# MLflow records every combination so we can compare them in the UI.
candidates = [
    {
        "run_name": "tfidf-logistic",
        "pipeline": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf",   LogisticRegression(max_iter=1000)),
        ]),
        "params": {"vectorizer": "tfidf", "model": "LogisticRegression",
                   "max_features": 5000, "ngram_range": "(1,2)"},
    },
    {
        "run_name": "tfidf-linearsvc",
        "pipeline": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf",   LinearSVC()),
        ]),
        "params": {"vectorizer": "tfidf", "model": "LinearSVC",
                   "max_features": 5000, "ngram_range": "(1,2)"},
    },
    {
        "run_name": "tfidf-randomforest",
        "pipeline": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=3000)),
            ("clf",   RandomForestClassifier(n_estimators=100, random_state=42)),
        ]),
        "params": {"vectorizer": "tfidf", "model": "RandomForest",
                   "max_features": 3000, "n_estimators": 100},
    },
]

# ── RUN ALL CANDIDATES ────────────────────────────────────────────────────────
run_results = []

for candidate in candidates:
    with mlflow.start_run(run_name=candidate["run_name"]) as run:

        candidate["pipeline"].fit(X_train, y_train)
        preds = candidate["pipeline"].predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_score": f1_score(y_test, preds, zero_division=0),
        }

        mlflow.log_params(candidate["params"])
        mlflow.log_metrics(metrics)

        # Log the full sklearn Pipeline (vectorizer + classifier together).
        # This ensures the same preprocessing is always applied at inference time.
        mlflow.sklearn.log_model(
            sk_model=candidate["pipeline"],
            artifact_path="model",
            registered_model_name="imdb-sentiment"
        )

        run_results.append({
            "run_id"  : run.info.run_id,
            "run_name": candidate["run_name"],
            **metrics
        })
        print(f"{candidate['run_name']:25s} | acc={metrics['accuracy']:.4f} | "
              f"f1={metrics['f1_score']:.4f}")

# ── FIND BEST & ASSIGN ALIAS ──────────────────────────────────────────────────
best = max(run_results, key=lambda x: x["f1_score"])
print(f"\nBest model: {best['run_name']} (F1={best['f1_score']:.4f})")

client = MlflowClient()
client.set_registered_model_alias("imdb-sentiment", "champion", version="1")
print("Alias 'champion' → version 1")


# ── STEP 5: SERVE AS REST API ─────────────────────────────────────────────────
# Launch a local REST API server backed by the champion model.
# Run this in a SEPARATE terminal:
#
#   mlflow models serve \
#     -m "models:/imdb-sentiment@champion" \
#     --host 0.0.0.0 --port 5002 --no-conda
#
# Then call it like this:

def predict_sentiment(texts: list[str]) -> dict:
    """Send review texts to the MLflow model server and get predictions."""
    response = requests.post(
        url="http://localhost:5001/invocations",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"inputs": texts})
    )
    return response.json()

# Example call (requires server to be running)
# results = predict_sentiment([
#     "This movie was amazing, I loved it!",
#     "Terrible film, complete waste of time.",
# ])
# print(results)  # {"predictions": [1, 0]}
