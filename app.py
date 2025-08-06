import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("logreg_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# App title
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below and get its predicted sentiment!")

# User input
user_input = st.text_area("Your Movie Review:", "")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a review before predicting.")
    else:
        # Transform input
        input_vector = vectorizer.transform([user_input])
        
        # Predict
        prediction = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector)[0]

        # Display result
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.success(f"Predicted Sentiment: **{sentiment}**")
        st.info(f"Confidence: Positive = {proba[1]:.2f}, Negative = {proba[0]:.2f}")
