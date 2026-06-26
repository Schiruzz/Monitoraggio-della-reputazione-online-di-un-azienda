# Monitoraggio-della-reputazione-online-di-un-azienda

Questo progetto implementa un sistema MLOps per l'analisi automatica del sentiment sui social media, sviluppato per MachineInnovators Inc. L'obiettivo è monitorare la reputazione aziendale attraverso l'analisi di testi provenienti dai social media, classificandoli in sentiment positivo, neutro o negativo.

**L'applicazione è accessibile su HuggingFace Spaces:** [MachineInnovators - Sentiment Monitor](https://huggingface.co/spaces/Schiro/Monitoraggio_Della_Reputazione_Online)

**Struttura del Progetto**

Monitoraggio-della-reputazione-online-di-un-azienda/
├── src/
│   ├── model_setup.py       # Caricamento dataset, modello, preprocessing e predizione
│   ├── evaluate.py          # Valutazione delle performance del modello (accuracy, F1)
│   ├── monitoring.py        # Monitoraggio distribuzione sentiment e performance
│   └── app.py               # Interfaccia web Gradio per l'analisi del sentiment
├── tests/
│   └── test_model.py        # Test di integrazione con pytest
├── .github/
│   └── workflows/
│       ├── ci.yml           # Pipeline CI: linting, test, performance check
│       └── cd.yml           # Pipeline CD: deploy automatico su HuggingFace Spaces
├── deploy.py                # Script di deploy su HuggingFace Spaces
├── Dockerfile               # Containerizzazione dell'applicazione
├── requirements.txt         # Dipendenze del progetto
└── README.md

Percorso di Sviluppo

**Fase 1: Implementazione del Modello di Analisi del Sentiment**
Scelta del Modello e Studio della Documentazione

Il primo passo è stato studiare la documentazione ufficiale del modello su HuggingFace. Analizzando la pagina del modello cardiffnlp/twitter-roberta-base-sentiment-latest ho notato due elementi fondamentali che differivano da un approccio generico:

Il Preprocessing specifico: il modello richiede che i testi vengano preprocessati in un modo particolare prima di essere passati al tokenizer. Ogni @nomeutente viene sostituito con @user e ogni URL con http, perché il modello è stato addestrato su tweet dove username e URL erano già sostituiti in questo modo. Senza questo preprocessing, il modello non riconosce correttamente questi elementi.

Organizzazione del Codice
Inizialmente avevo creato file separati per ogni piccola funzionalità (load_dataset.py, preprocess.py), ma mi sono reso conto che per tre righe di codice non aveva senso avere file separati. Ho deciso di unire tutto in model_setup.py, che contiene:

Caricamento del dataset cardiffnlp/tweet_eval (sentiment) con i suoi split già pronti (train: 45.615, test: 12.284, validation: 2.000)
La funzione preprocess() fedele alla documentazione ufficiale
Il caricamento del modello, tokenizer e config da HuggingFace
La funzione predict() che esegue il preprocessing, la tokenizzazione, l'inferenza e il ranking


Un problema iniziale è stato la duplicazione del preprocessing: lo avevo sia nel caricamento del dataset che nella funzione predict. Ho deciso di tenerlo solo in model_setup.py come funzione importabile, evitando ridondanze.

Dataset
Ho utilizzato il dataset cardiffnlp/tweet_eval con il task "sentiment", che contiene tweet etichettati con tre classi:

0 → Negative
1 → Neutral
2 → Positive


Inizialmente il caricamento usava load_dataset("tweet_eval", "sentiment"), ma una versione più recente di huggingface_hub ha richiesto il namespace completo: load_dataset("cardiffnlp/tweet_eval", "sentiment").

Comprensione del Funzionamento

Ogni riga del codice di inferenza ha un ruolo specifico che ho dovuto comprendere:

tokenizer(text, return_tensors='pt'): il tokenizer trasforma il testo in numeri comprensibili dal modello. return_tensors='pt' specifica il formato PyTorch, necessario perché il modello è stato costruito con questa libreria.
model(**encoded_input): passa i numeri al modello, che restituisce i logits (punteggi grezzi).
.detach().numpy(): stacca il risultato dal calcolo del gradiente di PyTorch (non necessario per l'inferenza) e converte in array NumPy.
softmax(scores): converte i logits grezzi in probabilità che sommano a 1.
np.argsort(scores)[::-1]: restituisce gli indici ordinati dal valore più alto al più basso, permettendo di identificare la label più probabile.


**Fase 1.5: Valutazione del Modello**
Ho separato la valutazione in evaluate.py per mantenere il codice ordinato e permetterne l'importazione dalla pipeline CI/CD. La valutazione calcola accuracy e F1 score su un sample di 1000 tweet dal test set.

Risultati ottenuti con 2000 sample:


Accuracy: 0.715 (il modello indovina il sentiment corretto 7 volte su 10)
F1 Score: 0.7146 (conferma che il modello è bilanciato tra le classi)


Un aspetto importante: ho usato if __name__ == "__main__" per separare il codice che deve essere eseguito solo quando il file viene lanciato direttamente, non quando viene importato da altri file (come nel performance check della CI/CD).

**Fase 2: Creazione della Pipeline CI/CD**

CI (Continuous Integration) - ci.yml

La pipeline CI viene eseguita automaticamente ad ogni push sul branch main e comprende:


Linting con flake8: controlla la sintassi del codice Python senza eseguirlo, segnalando errori di sintassi, variabili non definite e problemi di stile. Due passaggi: il primo blocca la pipeline per errori gravi, il secondo segnala warning senza bloccare.
Test di integrazione con pytest: esegue tests/test_model.py che verifica:

Il preprocessing sostituisce correttamente @username con @user
Il preprocessing sostituisce correttamente gli URL con http
La funzione predict restituisce un valore valido (0, 1 o 2)



Performance gate: importa le metriche da evaluate.py e verifica che accuracy e F1 siano sopra la soglia di 0.65. Se scendono sotto, la pipeline fallisce come segnale di allarme.


CD (Continuous Deployment) - cd.yml

La pipeline CD parte automaticamente solo quando il CI passa con successo (workflow_run con condizione success). Esegue deploy.py che:


Carica la cartella src/ su HuggingFace Spaces
Carica il requirements.txt separatamente


Ho dovuto risolvere diversi problemi con il deploy:


Il token HuggingFace doveva essere configurato come repository secret su GitHub (nome: HF_TOKEN, valore: il token senza virgolette o spazi)
L'indentazione YAML è critica: ogni step deve essere allineato con gli altri
Ho separato il codice di deploy in un file Python dedicato (deploy.py) per evitare problemi di formattazione YAML con codice Python inline


**Fase 3: Deploy e Monitoraggio Continuo**

Deploy su HuggingFace Spaces

L'applicazione è deployata come Gradio Space su HuggingFace. L'interfaccia web permette di:


Inserire un testo da analizzare
Visualizzare il sentiment predetto (Positive, Neutral, Negative)
Provare esempi pre-configurati


Un problema incontrato: le versioni fisse nel requirements.txt non erano compatibili con il Python 3.13 di HuggingFace Spaces. La soluzione è stata rimuovere le versioni fisse, lasciando che pip installi automaticamente quelle compatibili.

**App Gradio (app.py)**

L'interfaccia è costruita con Gradio, che crea automaticamente una pagina web a partire da una funzione Python. I componenti principali:


gr.Interface: il wrapper principale che collega input, funzione e output
gr.Textbox: la casella di input per il tweet
gr.Label: il componente che mostra il risultato
examples: tweet di esempio cliccabili per testare rapidamente


Durante lo sviluppo locale su Codespaces, ho dovuto usare share=True in demo.launch() perché Codespaces non permette di accedere a localhost direttamente. Questo crea un link pubblico temporaneo (72 ore) tramite i server di Gradio.

**Monitoraggio (monitoring.py)**

Il sistema di monitoraggio simula un controllo continuo delle performance del modello:


Prende un sample casuale di 100 tweet dal test set (usando random.sample per simulare nuovi dati in arrivo ogni volta)
Predice il sentiment per ogni tweet
Calcola la distribuzione del sentiment (percentuale di positivi, neutri, negativi)
Calcola accuracy e F1 score confrontando le predizioni con le label reali


**Risultati tipici del monitoraggio:**


Negative: ~37%
Neutral: ~45%
Positive: ~18%


Docker

Il progetto include un Dockerfile per la containerizzazione:

Usa Python 3.10 come immagine base
Installa le dipendenze dal requirements.txt
Copia il codice sorgente
Avvia l'applicazione Gradio


Il Dockerfile garantisce che l'ambiente sia sempre identico indipendentemente da dove viene eseguito.

**Problemi Incontrati e Soluzioni**
ProblemaSoluzioneModuleNotFoundError: No module named 'datasets'Installazione dei requirements con pip install -r requirements.txt
Import tra file in cartelle diverse
sys.path.append("src") nel file di testFile con numeri nel nome non importabili
Rinominati da 1_load_and_preprocess.py a load_and_preprocess.py
Dataset tweet_eval non trovatoAggiornato a cardiffnlp/tweet_eval con namespace completo
Gradio non funziona su CodespacesAggiunto share=True al demo.launch()
Token HuggingFace vuoto nel deployUsato os.environ['HF_TOKEN'] e configurato correttamente il secret 
su GitHubtorch==2.2.0 non compatibile con Python 3.13
Rimossi i vincoli di versione rigidi dal requirements.txt

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
