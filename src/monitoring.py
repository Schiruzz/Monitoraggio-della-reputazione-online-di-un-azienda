from model_setup import dataset, predict
import numpy as np
import random

# Take a sample of random tweets to monitor
test_dataset = dataset["test"]
sample_size  = 100
indices = random.sample(range(len(test_dataset["text"])), sample_size)
texts   = [test_dataset["text"][i] for i in indices]


# Predict sentiment for each tweet
print("Predicting sentiment on sample...")
pred_labels = [predict(text) for text in texts]

# Count sentiment distribution
# 0=Negative, 1=Neutral, 2=Positive
distribution = np.bincount(pred_labels, minlength=3) / sample_size

print(f"\n--- Sentiment Distribution ---")
print(f"Negative: {distribution[0]:.2%}")
print(f"Neutral:  {distribution[1]:.2%}")
print(f"Positive: {distribution[2]:.2%}")