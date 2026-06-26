from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from datasets import load_dataset

'''
Load the dataset tweet_eval with sentiment task
train / validation / test already Splitted
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
    print(f"Prediction: {top_label} ({top_score})")

    return top_idx
