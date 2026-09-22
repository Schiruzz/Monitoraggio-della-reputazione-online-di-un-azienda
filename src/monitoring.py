from datasets import load_dataset
from model_setup import predict_batch
import numpy as np
import json
from pathlib import Path
import mlflow
import sys
from datetime import date

WINDOW_SIZE = 500          # tweets per monitoring window, one "day" of traffic
DRIFT_THRESHOLD = 0.10     # alert when the negative share grows by more than 10 points
BASELINE_PATH = Path("baseline.json")
BASELINE_WINDOWS = 4       # first windows of the stream define normal behaviour
MLFLOW_URI = "sqlite:///mlflow.db"



def get_window(index, size=WINDOW_SIZE):
    """Return one window of incoming texts. Swap this to read from a live source."""
    test = load_dataset("cardiffnlp/tweet_eval", "sentiment")["test"]
    start = index * size
    return test["text"][start:start + size]


def sentiment_distribution(texts):
    """Share of negative, neutral and positive predictions over a list of texts."""
    predictions = predict_batch(texts)
    counts = np.bincount(predictions, minlength=3)
    return (counts / len(predictions)).tolist()

def build_baseline():
    """Compute the reference distribution on the first windows of the stream and store it."""
    texts = []
    for index in range(BASELINE_WINDOWS):
        texts += get_window(index)

    distribution = sentiment_distribution(texts)
    BASELINE_PATH.write_text(json.dumps(distribution))
    return distribution

def load_baseline():
    """Read the stored baseline, building it the first time."""
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text())
    return build_baseline()


def check_drift(current, baseline, threshold=DRIFT_THRESHOLD):
    """Compare the negative share against the baseline; return the shift and whether it alerts."""
    shift = current[0] - baseline[0]  # index 0 is the negative class
    return shift, shift > threshold


def monitor(first_window, n_windows):
    """Run the monitoring loop over consecutive windows; return the alerting ones."""
    baseline = load_baseline()
    print(f"{'baseline negative share:':<26}{baseline[0]:>8.2%}\n")

    header = f"{'window':<10}{'negative':>10}{'neutral':>10}{'positive':>10}{'shift':>10}"
    print(header)
    print("-" * len(header))

    alerts = []
    for index in range(first_window, first_window + n_windows):
        distribution = sentiment_distribution(get_window(index))
        shift, alerting = check_drift(distribution, baseline)

        mlflow.log_metrics({
            "negative": distribution[0],
            "neutral": distribution[1],
            "positive": distribution[2],
            "negative_shift": shift,
            "alert": int(alerting),
        }, step=index)

        flag = "  ALERT" if alerting else ""
        print(f"{index:<10}{distribution[0]:>10.2%}{distribution[1]:>10.2%}"
              f"{distribution[2]:>10.2%}{shift:>+10.2%}{flag}")
        if alerting:
            alerts.append(index)

    return alerts


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_URI)

    # walk the stream with the calendar, so a scheduled run sees new data each day
    total_windows = 24
    available = total_windows - BASELINE_WINDOWS
    first = BASELINE_WINDOWS + (date.today().toordinal() % available)

    with mlflow.start_run():
        mlflow.log_params({"window_size": WINDOW_SIZE, "threshold": DRIFT_THRESHOLD})
        alerts = monitor(first_window=first, n_windows=1)