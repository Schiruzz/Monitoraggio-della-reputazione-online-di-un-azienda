from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from scipy.special import softmax
from datasets import load_dataset

'''
Load the dataset tweet_eval with sentiment task
train / validation / test already Splited
'''

dataset = load_dataset("tweet_eval", "sentiment")

# Preprocess text as required by the model
# - @username -> @user
# - https://... -> http

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Load model, tokenizer and config from HuggingFace
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
config    = AutoConfig.from_pretrained(MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Predict sentiment of a text
def predict(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)[::-1]
    top_idx = int(ranking[0])
    top_label = config.id2label[top_idx]
    top_score = np.round(float(scores[top_idx]), 4)

    print(f"prediction: {top_label} ({top_score})")

    return top_idx

# --- EVALUATION ---
print("Loading test set...")
dataset = load_dataset("tweet_eval", "sentiment")
test_dataset = dataset["test"]

print("Predicting on test set... (this may take a while)")

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



# Test with some examples
print("--- Tweet 1 ---")
predict("Covid cases are increasing fast!")

print("\n--- Tweet 2 ---")
predict("I love this product, it's amazing!")

print("\n--- Tweet 3 ---")
predict("The update was released yesterday.")
