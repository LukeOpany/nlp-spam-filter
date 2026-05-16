"""Train and evaluate an SMS spam detection model."""

from __future__ import annotations

import string
from pathlib import Path
from typing import Any

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


DATA_PATH = Path(__file__).with_name("SMSSpamCollection.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.3
LABEL_ORDER = ["ham", "spam"]
ENGLISH_STOPWORDS: set[str] | None = None


def ensure_stopwords() -> None:
    """Download the NLTK stopwords corpus if it is not already available."""
    global ENGLISH_STOPWORDS

    try:
        ENGLISH_STOPWORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        ENGLISH_STOPWORDS = set(stopwords.words("english"))


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the SMS spam dataset."""
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"],
    )


def text_process(message: str) -> list[str]:
    """Remove punctuation and common English stopwords from a message."""
    if ENGLISH_STOPWORDS is None:
        ensure_stopwords()

    no_punctuation = "".join(
        character for character in message if character not in string.punctuation
    )

    return [
        word
        for word in no_punctuation.lower().split()
        if word not in ENGLISH_STOPWORDS
    ]


def build_pipeline() -> Pipeline:
    """Build the text preprocessing and classification pipeline."""
    return Pipeline(
        steps=[
            ("bow", CountVectorizer(analyzer=text_process)),
            ("tfidf", TfidfTransformer()),
            ("classifier", MultinomialNB()),
        ]
    )


def train_and_evaluate(df: pd.DataFrame) -> dict[str, Any]:
    """Train the model and return test-set metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=LABEL_ORDER,
        ),
        "classification_report": classification_report(y_test, predictions),
    }


def main() -> None:
    ensure_stopwords()
    df = load_data()
    metrics = train_and_evaluate(df)

    print("SMS Spam Detection Model")
    print("========================")
    print(f"Rows: {len(df):,}")
    print("Class counts:")
    print(df["label"].value_counts().rename_axis(None).to_string())
    print()
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print("Confusion matrix (labels: ham, spam):")
    print(metrics["confusion_matrix"])
    print()
    print("Classification report:")
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
