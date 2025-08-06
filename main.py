import joblib
from src.data_preprocessing import clean_text

def predict_sentiment(text: str):
    model = joblib.load('logreg_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')

    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment

if __name__ == "__main__":
    user_input = input("Enter a movie review: ")
    result = predict_sentiment(user_input)
    print(f"Sentiment: {result}")
