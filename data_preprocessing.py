import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Clean input text by removing URLs, mentions, hashtags, punctuations, stopwords and lowering the case."""
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # remove URLs
    text = re.sub(r'\@\w+|\#','', text)  # remove mentions and hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # remove punctuation and numbers
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocess_dataframe(df, text_column: str, label_column: str):
    """Apply text cleaning to dataframe text column, encode labels, and return X, y."""
    df['cleaned_text'] = df[text_column].apply(clean_text)
    df[label_column] = df[label_column].map({'positive': 1, 'negative': 0})  # label encoding
    X = df['cleaned_text']
    y = df[label_column]
    return X, y
