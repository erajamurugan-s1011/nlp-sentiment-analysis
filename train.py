import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import joblib
from src.data_preprocessing import preprocess_dataframe

def train_model(dataset_path: str, text_col: str, label_col: str):
    # Load data
    df = pd.read_csv(dataset_path)

    # Preprocess
    X, y = preprocess_dataframe(df, text_col, label_col)

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vect = vectorizer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

    # Define models to compare
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100)
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {name}: {acc:.4f}")
        print(f"Classification Report for {name}:\n{classification_report(y_test, y_pred)}")

        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{name.replace(' ', '_').lower()}.png")
        plt.close()

    # Save best model (example: Logistic Regression)
    best_model = models['Logistic Regression']
    joblib.dump(best_model, 'logreg_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

if __name__ == "__main__":
    train_model('data/imdb_reviews.csv', text_col='review', label_col='sentiment')