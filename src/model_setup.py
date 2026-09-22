from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import torch

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


# Predict sentiment of a list of texts, in batches; returns class indices
def predict_batch(texts, batch_size=32):
    predictions = []
    for start in range(0, len(texts), batch_size):
        batch = [preprocess(t) for t in texts[start:start + batch_size]]
        encoded = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():  # inference only, no gradients to track
            logits = model(**encoded).logits
        predictions.extend(logits.argmax(dim=1).tolist())
    return predictions


# Predict sentiment of a single text; returns the class index
def predict(text):
    return predict_batch([text])[0]