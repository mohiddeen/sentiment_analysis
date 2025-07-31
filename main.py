import streamlit as st
import string
import emoji
import joblib
import nltk
import os

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download NLTK requirements

# Load saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = emoji.replace_emoji(text, replace='')
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
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
