import gradio as gr
from model_setup import predict_scores, config

def analyze_sentiment(text):
    scores = predict_scores(text)
    return {config.id2label[i]: scores[i] for i in range(3)}

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

demo.launch(server_name="0.0.0.0", server_port=7860)