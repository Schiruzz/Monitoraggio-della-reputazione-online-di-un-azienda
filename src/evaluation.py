from datasets import load_dataset
from model_setup import predict_batch, config
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
import numpy as np


def evaluate_model(sample_size=None):
    """Predict on the test set (or its first sample_size tweets); return true and predicted labels."""
    test = load_dataset("cardiffnlp/tweet_eval", "sentiment")["test"]
    if sample_size is not None:
        test = test.select(range(sample_size))

    true_labels = np.array(test["label"])
    pred_labels = np.array(predict_batch(test["text"]))
    return true_labels, pred_labels


if __name__ == "__main__":
    true_labels, pred_labels = evaluate_model()
    names = [config.id2label[i] for i in range(3)]

    print(f"{'tweets:':<12}{len(true_labels):>8}")
    print(f"{'accuracy:':<12}{accuracy_score(true_labels, pred_labels):>8.4f}")
    print(f"{'macro F1:':<12}{f1_score(true_labels, pred_labels, average='macro'):>8.4f}\n")

    print(classification_report(true_labels, pred_labels, target_names=names, digits=3))

    # rows = true class, columns = predicted class
    matrix = confusion_matrix(true_labels, pred_labels)
    print(f"{'true / pred':<14}" + "".join(f"{n:>10}" for n in names))
    for name, row in zip(names, matrix):
        print(f"{name:<14}" + "".join(f"{v:>10}" for v in row))