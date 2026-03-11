# pip install mlflow google-generativeai
# Get free API key at: https://aistudio.google.com
import mlflow
import mlflow.gemini          # MLflow 3.0 native Gemini integration
import google.generativeai as genai
import os

# ── SETUP ─────────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("imdb-sentiment-llm-gemini")

# Configure Gemini with your free API key from aistudio.google.com
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# ── ENABLE TRACING ────────────────────────────────────────────────────────────
# MLflow 3.0 automatically captures every Gemini call:
# - full prompt (system + user messages)
# - full response text
# - token usage (input / output / total)
# - latency per call
# - model name and generation config
# Visible under the "Traces" tab in the MLflow UI.
mlflow.gemini.autolog()

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
# The system prompt is the "model" in an LLM pipeline.
# Logging it as an artifact lets you version and compare prompts
# across runs — just like you'd compare hyperparameters in traditional ML.
SYSTEM_PROMPT = """You are a sentiment classifier for movie reviews.

Classify the review as exactly one of:
  POSITIVE — the reviewer liked the movie
  NEGATIVE — the reviewer disliked the movie

Rules:
- Reply with only the label: POSITIVE or NEGATIVE
- No explanations, no punctuation, nothing else."""

# Few-shot examples are included in the user message to improve accuracy.
FEW_SHOT_EXAMPLES = """Examples:
Review: "A breathtaking cinematic experience."
Label: POSITIVE

Review: "Painfully boring and poorly written."
Label: NEGATIVE"""

# ── CLASSIFIER FUNCTION ───────────────────────────────────────────────────────
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash", # free tier model
    system_instruction=SYSTEM_PROMPT
)

@mlflow.trace  # marks this function as a named span in the trace
def classify_review(review: str) -> str:
    """Classify a single review as POSITIVE or NEGATIVE using Gemini."""
    prompt = f"{FEW_SHOT_EXAMPLES}\n\nReview: \"{review}\"\nLabel:"

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.0,   # deterministic output — critical for classifiers
            max_output_tokens=10,
        )
    )
    return response.text.strip()

@mlflow.trace
def run_batch(reviews: list[str]) -> list[str]:
    """Classify a batch of reviews and return predictions."""
    results = []
    for review in reviews:
        results.append(classify_review(review))
        time.sleep(60)
    return results

# ── EVALUATION DATASET ────────────────────────────────────────────────────────
test_reviews = [
    ("I walked out after 20 minutes. Absolutely dreadful.",              "NEGATIVE"),
    ("Funny, clever, and thoroughly entertaining.",                       "POSITIVE"),
]

# ── RUN & LOG ─────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="gemini-flash-fewshot"):

    # Log the prompt as a versioned artifact.
    # Changing the prompt = new run = easy A/B comparison in the UI.
    mlflow.log_text(SYSTEM_PROMPT,     artifact_file="system_prompt.txt")
    mlflow.log_text(FEW_SHOT_EXAMPLES, artifact_file="few_shot_examples.txt")

    mlflow.log_params({
        "model"      : "gemini-2.0-flash",
        "temperature": 0.0,
        "n_shots"    : 4,
        "strategy"   : "few-shot",
    })

    # Run batch classification — each Gemini call is traced automatically
    reviews    = [r for r, _ in test_reviews]
    gold_labels = [l for _, l in test_reviews]
    predictions = run_batch(reviews)

    # Compute and log accuracy
    correct  = sum(p == g for p, g in zip(predictions, gold_labels))
    accuracy = correct / len(gold_labels)
    mlflow.log_metric("accuracy", accuracy)

    # Log per-review results for detailed inspection
    for i, (review, gold, pred) in enumerate(zip(reviews, gold_labels, predictions)):
        match = "✓" if pred == gold else "✗"
        print(f"{match} [{gold:8s} → {pred:8s}] {review[:60]}...")

    print(f"\nAccuracy: {accuracy:.2f} ({correct}/{len(gold_labels)})")
    print("Check traces at http://localhost:5001 → Traces tab")
