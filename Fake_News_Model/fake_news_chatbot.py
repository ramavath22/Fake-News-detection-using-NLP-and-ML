import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("fake_news_model")
    tokenizer = BertTokenizer.from_pretrained("fake_news_model")
    return model, tokenizer

model, tokenizer = load_model()

# Prediction function
def predict_news(text):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs).item()
        confidence = torch.max(probs).item()
    return prediction, confidence

# Streamlit UI
st.set_page_config(page_title="Fake News Detection Chatbot", layout="centered")
st.title("📰 Fake News Detection Chatbot")
st.write("Enter a news article or snippet to check if it's likely real or fake.")

# Text input
user_input = st.text_area("Paste your news article here:", height=200)

# Detect button
if st.button("Detect"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        label, conf = predict_news(user_input)
        if label == 1:
            st.success(f"🟢 This news is likely **REAL** with a confidence of {conf * 100:.2f}%.")
        else:
            st.error(f"🔴 This news is likely **FAKE** with a confidence of {conf * 100:.2f}%.")
