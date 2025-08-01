import streamlit as st
import string
import emoji
import joblib
import nltk
import os
import pandas as pd
import numpy as np
import re

from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Define NLTK data path and download required resources
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)

for pkg in ['stopwords', 'wordnet']:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_path)

# Load saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Use Treebank tokenizer to avoid punkt errors
tokenizer = TreebankWordTokenizer()

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = emoji.replace_emoji(text, replace='')
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.title("📖 Sentiment Analysis App")
st.write("Enter a review below and see if it's Positive or Negative:")

user_input = st.text_area("✍️ Type your review here")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review before submitting.")
    else:
        cleaned = preprocess_text(user_input)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]
        st.success(f"🎯 **Predicted Sentiment:** {prediction.capitalize()}")
