import time
import ray
from ray import serve
import joblib

ray.init()

serve.start(
    http_options={
        "host": "0.0.0.0",
        "port": 9000,
    }
)

@serve.deployment
class SentimentClassifier:
    def __init__(self):
        self.model = joblib.load("imdb_best_model.joblib")

    async def __call__(self, request):
        text = request.query_params.get("text", "")
        pred = self.model.predict([text])[0]
        label = "positive" if pred == 1 else "negative"
        return {"prediction": label}

app = SentimentClassifier.bind()
serve.run(app)

print("Serve is running at http://127.0.0.1:9000")
while True:
    time.sleep(3600)
