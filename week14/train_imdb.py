from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import ray

ray.init()

def main():
    dataset = load_dataset("stanfordnlp/imdb")

    train_data = dataset["train"].shuffle(seed=42)
    train_texts = train_data["text"][:5000]
    train_labels = train_data["label"][:5000]

    test_texts = train_data["text"][:2000]
    test_labels = train_data["label"][:2000]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english")),
        ("clf", LogisticRegression(max_iter=200))
    ])

    model.fit(train_texts, train_labels)
    preds = model.predict(test_texts)
    acc = accuracy_score(test_labels, preds)

    print(f"Test accuracy: {acc:.4f}")

    joblib.dump(model, "imdb_model.joblib")
    print("Saved model to imdb_model.joblib")

if __name__ == "__main__":
    main()
