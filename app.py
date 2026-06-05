import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import requests
from bs4 import BeautifulSoup
import numpy as np

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[\d]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'<.*?>', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Extract text from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        article_text = " ".join([para.get_text() for para in paragraphs])
        return article_text if article_text else None
    except Exception:
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="Fake News Detector", page_icon="", layout="centered")

# Title
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'> Fake News Detector</h1>
    <p style='text-align: center; font-size:18px;'>Enter an article or URL to check whether it's <b style="color:green;">REAL</b> or <b style="color:red;">FAKE</b></p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Input options
option = st.radio("Choose input type:", [" Paste article text", " Enter article link"])

# Prediction function
def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    confidence = np.max(probs) * 100
    return prediction, confidence

# Paste text option
if option == " Paste article text":
    text_input = st.text_area("Paste the news article text here:", height=200, placeholder="Enter the article content...")
    if st.button("🔍 Check"):
        if text_input.strip() == "":
            st.warning(" Please enter some text.")
        else:
            pred, conf = predict_news(text_input)
            if pred == "REAL":
                st.success(f" Prediction: **REAL NEWS** ({conf:.2f}% confidence)")
            else:
                st.error(f" Prediction: **FAKE NEWS** ({conf:.2f}% confidence)")

# URL option
elif option == " Enter article link":
    url_input = st.text_input("Enter the article URL:", placeholder="https://example.com/article")
    if st.button(" Fetch & Check"):
        if url_input.strip() == "":
            st.warning("Please enter a URL.")
        else:
            article_text = extract_text_from_url(url_input)
            if article_text:
                pred, conf = predict_news(article_text)
                if pred == "REAL":
                    st.success(f" Prediction: **REAL NEWS** ({conf:.2f}% confidence)")
                else:
                    st.error(f" Prediction: **FAKE NEWS** ({conf:.2f}% confidence)")
            else:
                st.error(" Could not extract text from the provided URL.")

# Footer
st.markdown("---")

