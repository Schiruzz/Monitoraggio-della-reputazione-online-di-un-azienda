import gradio as gr
from model_setup import predict, config

def analyze_sentiment(text):
    idx = predict(text)
    label = config.id2label[idx]
    return label

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(placeholder="Enter a tweet to analyze..."),
    outputs=gr.Label(),
    title="MachineInnovators - Sentiment Monitor",
    description="Analyze the sentiment of social media texts: Positive, Neutral or Negative",
    examples=[
        ["Covid cases are increasing fast!"],
        ["I love this product, it's amazing!"],
        ["The update was released yesterday."]
    ]
)

demo.launch(share= True)