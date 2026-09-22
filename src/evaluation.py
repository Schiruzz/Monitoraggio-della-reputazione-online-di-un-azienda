from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from datasets import load_dataset
from model_setup import predict

SAMPLE_SIZE = 100


def evaluate_model(sample_size=SAMPLE_SIZE):
    """Score the model on a slice of the test set, return accuracy and weighted F1."""
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    test_dataset = dataset["test"]

    texts = test_dataset["text"][:sample_size]
    true_labels = np.array(test_dataset["label"][:sample_size], dtype=int)
    pred_labels = np.array([predict(text) for text in texts], dtype=int)

    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="weighted")
    return accuracy, f1


if __name__ == "__main__":
    accuracy, f1 = evaluate_model()
    print(f"{'Accuracy:':<12}{accuracy:>8.4f}")
    print(f"{'F1 score:':<12}{f1:>8.4f}")