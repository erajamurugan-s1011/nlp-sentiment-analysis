import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def predict_sentiment(text: str):
    model = joblib.load('logreg_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')

    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    return "Positive" if pred == 1 else "Negative"

if __name__ == "__main__":
    while True:
        text = input("Enter text (or type 'exit'): ")
        if text.lower() == 'exit':
            break
        print("Sentiment:", predict_sentiment(text))
