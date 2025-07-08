# 📰 Fake News Detection Chatbot (Transformer-based)

This project implements a Transformer-based chatbot that classifies whether a news article is **fake or real** using BERT. It combines fine-tuning, prediction, and an interactive Streamlit interface.

---

##  Dataset

Download `true.csv` and `fake.csv` as instructed in the `final_project.pdf`.

---

##  Model Training

1. Open the notebook: `fake_news_detection.ipynb`.
2. Run **all cells one by one** (ensure you’re in Google Colab or a GPU-enabled environment).
3. The final cell will generate and download a file named : fake_news_model.zip

## Project Structure

Your project folder should now look like:
your_project_directory/
├── fake_news_chatbot.py
├── requirements.txt
└── fake_news_model/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    ├── vocab.txt

Make sure your requirements.txt includes:

streamlit
transformers
torch


##  Running the Chatbot Locally

### Install dependencies:
```bash
pip install -r requirements.txt
```
### Launch the Streamlit chatbot:
```bash
streamlit run fake_news_chatbot.py
```
### Access the chatbot:
```bash
open: http://localhost:8501
```
