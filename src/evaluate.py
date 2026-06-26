from model_setup import dataset, predict
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# --- EVALUATION ---
print("Loading test set...")
test_dataset = dataset["test"]

'''
it takes a long time to predict on the entire test set,
so we will use a  sample of 500 tweets to evaluate the model
'''
sample_size = 500
texts = test_dataset["text"][:sample_size]
true_labels = np.array(test_dataset["label"][:sample_size], dtype=int)
pred_labels = np.array([int(predict(text)) for text in texts], dtype=int)

accuracy = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels, average="weighted")

print(f"\n--- Evaluation Results ---")
print(f"Accuracy: {np.round(accuracy, 4)}")
print(f"F1 Score: {np.round(f1, 4)}")


# Examples — only when file run directly
if __name__ == "__main__":
    print("\n--- Tweet 1 ---")
    predict("Covid cases are increasing fast!")

    print("\n--- Tweet 2 ---")
    predict("I love this product, it's amazing!")

    print("\n--- Tweet 3 ---")
    predict("The update was released yesterday.")