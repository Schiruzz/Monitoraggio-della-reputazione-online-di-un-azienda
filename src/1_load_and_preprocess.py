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


# Applying preprocessing to the dataset
dataset = dataset.map(lambda x: {"text": preprocess(x["text"])})

print(dataset)
print("\nExample preprocessed tweet:")
print(dataset["train"][0])
print("Dataset completed loading!")
