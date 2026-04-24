from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from ray import tune
from ray.tune import Tuner
import joblib
import ray

ray.init()

dataset = load_dataset("stanfordnlp/imdb")
train_data = dataset["train"].shuffle(seed=42)

train_texts = train_data["text"][:5000]
train_labels = train_data["label"][:5000]
test_texts = train_data["text"][:2000]
test_labels = train_data["label"][:2000]

def train_fn(config):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=config["max_features"],
            ngram_range=(1, config["max_ngram"]),
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            C=config["C"],
            max_iter=300
        ))
    ])

    model.fit(train_texts, train_labels)
    preds = model.predict(test_texts)
    acc = accuracy_score(test_labels, preds)

    tune.report({"accuracy": acc})

def main():
    param_space = {
        "max_features": tune.choice([5000, 10000, 20000]),
        "max_ngram": tune.choice([1, 2]),
        "C": tune.choice([0.1, 1.0, 10.0]),
    }

    tuner = Tuner(
        train_fn,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            num_samples=3,           # This is added to overcome memory issues
            max_concurrent_trials=1  # This is added to overcome memory issues
        ),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="accuracy", mode="max")

    print("Best config:", best.config)
    print("Best accuracy:", best.metrics["accuracy"])

    best_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=best.config["max_features"],
            ngram_range=(1, best.config["max_ngram"]),
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            C=best.config["C"],
            max_iter=300
        ))
    ])

    best_model.fit(train_texts, train_labels)
    joblib.dump(best_model, "imdb_best_model.joblib")
    print("Saved best model to imdb_best_model.joblib")

if __name__ == "__main__":
    main()
