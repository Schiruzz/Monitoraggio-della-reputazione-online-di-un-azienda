# Monitoraggio-della-reputazione-online-di-un-azienda

Questo progetto implementa un sistema MLOps per l'analisi automatica del sentiment sui social media, sviluppato per MachineInnovators Inc. L'obiettivo è monitorare la reputazione aziendale attraverso l'analisi di testi provenienti dai social media, classificandoli in sentiment positivo, neutro o negativo.

**L'applicazione è accessibile su HuggingFace Spaces:** [MachineInnovators - Sentiment Monitor](https://huggingface.co/spaces/Schiro/Monitoraggio_Della_Reputazione_Online)

**Tecnologie Utilizzate**

Modello: cardiffnlp/twitter-roberta-base-sentiment-latest (RoBERTa fine-tuned su 124M tweet)
Dataset: cardiffnlp/tweet_eval (sentiment task)
Framework ML: PyTorch, Transformers (HuggingFace)
Interfaccia Web: Gradio
CI/CD: GitHub Actions
Deploy: HuggingFace Spaces
Containerizzazione: Docker
Testing: pytest, flake8
Metriche: scikit-learn (accuracy, F1 score)


**Come Eseguire il Progetto Localmente**

bash# Clona il repository
git clone https://github.com/Schiruzz/Monitoraggio-della-reputazione-online-di-un-azienda.git
cd Monitoraggio-della-reputazione-online-di-un-azienda

# Installa le dipendenze
pip install -r requirements.txt

# Esegui la valutazione del modello
cd src
python evaluate.py

# Esegui il monitoraggio
python monitoring.py

# Avvia l'app Gradio
python app.py

**#Come Eseguire con Docker**

bashdocker build -t sentiment-app .
docker run -p 7860:7860 sentiment-app
