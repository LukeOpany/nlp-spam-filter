"""
NLP Spam Filter - Core Functions

This module contains the main functions used in the NLP spam filter project.
It provides text preprocessing, model training, and prediction capabilities.

Author: Luke Opany
License: MIT
"""

import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(filepath='SMSSpamCollection.csv'):
    """
    Load and preprocess the SMS Spam Collection dataset.
    
    Args:
        filepath (str): Path to the SMS spam collection CSV file
        
    Returns:
        pd.DataFrame: DataFrame with columns ['label', 'message', 'length']
        
    Example:
        >>> messages = load_dataset('SMSSpamCollection.csv')
        >>> print(messages.head())
    """
    # Load dataset with tab separation and no header
    messages = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
    
    # Add message length feature
    messages['length'] = messages['message'].apply(len)
    
    # Display basic dataset information
    print(f"Dataset loaded successfully!")
    print(f"Total messages: {len(messages)}")
    print(f"Ham messages: {len(messages[messages['label'] == 'ham'])}")
    print(f"Spam messages: {len(messages[messages['label'] == 'spam'])}")
    
    return messages


def text_process(mess):
    """
    Process text message for spam classification.
    
    This function performs comprehensive text preprocessing including:
    1. Punctuation removal
    2. Case normalization
    3. Stopword removal
    4. Tokenization
    
    Args:
        mess (str): Input text message to process
        
    Returns:
        list: List of cleaned and processed tokens
        
    Example:
        >>> text_process("Free entry to win £1000! Call now!!!")
        ['free', 'entry', 'win', 'call']
    """
    # Step 1: Remove punctuation
    no_punc = [char for char in mess if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    
    # Step 2: Convert to lowercase and split into words
    words = no_punc.lower().split()
    
    # Step 3: Remove stopwords
    clean_tokens = [word for word in words if word not in stopwords.words('english')]
    
    return clean_tokens


def create_spam_pipeline():
    """
    Create a machine learning pipeline for spam classification.
    
    The pipeline consists of:
    1. CountVectorizer with custom text processing
    2. TF-IDF transformation
    3. Multinomial Naive Bayes classifier
    
    Returns:
        sklearn.pipeline.Pipeline: Configured ML pipeline
        
    Example:
        >>> pipeline = create_spam_pipeline()
        >>> pipeline.fit(X_train, y_train)
        >>> predictions = pipeline.predict(X_test)
    """
    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=text_process)),  # Bag of words with custom preprocessing
        ('tfidf', TfidfTransformer()),                    # TF-IDF transformation
        ('classifier', MultinomialNB())                   # Naive Bayes classifier
    ])
    
    print("Spam classification pipeline created successfully!")
    print("Pipeline steps:")
    for step_name, step_obj in pipeline.steps:
        print(f"  - {step_name}: {type(step_obj).__name__}")
    
    return pipeline


def train_test_split_data(messages, test_size=0.3, random_state=101):
    """
    Split the dataset into training and testing sets with stratification.
    
    Args:
        messages (pd.DataFrame): Dataset with 'message' and 'label' columns
        test_size (float): Proportion of data for testing (default: 0.3)
        random_state (int): Random seed for reproducibility (default: 101)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Training and testing data
        
    Example:
        >>> X_train, X_test, y_train, y_test = train_test_split_data(messages)
        >>> print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    """
    # Extract features and labels
    X = messages['message']
    y = messages['label']
    
    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Display split information
    print(f"Data split completed:")
    print(f"  Training samples: {len(X_train)} ({(1-test_size)*100:.1f}%)")
    print(f"  Testing samples: {len(X_test)} ({test_size*100:.1f}%)")
    print(f"  Training ham: {len(y_train[y_train == 'ham'])}")
    print(f"  Training spam: {len(y_train[y_train == 'spam'])}")
    print(f"  Testing ham: {len(y_test[y_test == 'ham'])}")
    print(f"  Testing spam: {len(y_test[y_test == 'spam'])}")
    
    return X_train, X_test, y_train, y_test


def evaluate_model(pipeline, X_test, y_test, show_details=True):
    """
    Evaluate the trained spam classification model.
    
    Args:
        pipeline: Trained sklearn pipeline
        X_test: Test feature data
        y_test: Test labels
        show_details (bool): Whether to show detailed metrics (default: True)
        
    Returns:
        dict: Dictionary containing evaluation metrics
        
    Example:
        >>> metrics = evaluate_model(pipeline, X_test, y_test)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    # Make predictions
    predictions = pipeline.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    
    if show_details:
        print("Model Evaluation Results:")
        print("=" * 50)
        print(f"Overall Accuracy: {accuracy:.3f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, predictions))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, predictions)
        print(cm)
        
        # Class-specific metrics
        ham_precision = cm[0, 0] / (cm[0, 0] + cm[1, 0])
        spam_precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        ham_recall = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        spam_recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        
        print(f"\nAdditional Metrics:")
        print(f"  Ham Precision: {ham_precision:.3f}")
        print(f"  Spam Precision: {spam_precision:.3f}")
        print(f"  Ham Recall: {ham_recall:.3f}")
        print(f"  Spam Recall: {spam_recall:.3f}")
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'confusion_matrix': confusion_matrix(y_test, predictions)
    }


def visualize_message_lengths(messages, figsize=(12, 6)):
    """
    Visualize message length distributions for ham and spam messages.
    
    Args:
        messages (pd.DataFrame): Dataset with 'label' and 'length' columns
        figsize (tuple): Figure size for the plot (default: (12, 6))
        
    Example:
        >>> visualize_message_lengths(messages)
    """
    plt.figure(figsize=figsize)
    
    # Create subplots for ham and spam distributions
    plt.subplot(1, 2, 1)
    messages[messages['label'] == 'ham']['length'].hist(bins=50, alpha=0.7, color='green')
    plt.title('Ham Messages Length Distribution')
    plt.xlabel('Message Length (characters)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    messages[messages['label'] == 'spam']['length'].hist(bins=30, alpha=0.7, color='red')
    plt.title('Spam Messages Length Distribution')
    plt.xlabel('Message Length (characters)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("Message Length Statistics:")
    print("=" * 40)
    print(messages.groupby('label')['length'].describe())


def predict_message(pipeline, message):
    """
    Predict whether a single message is spam or ham.
    
    Args:
        pipeline: Trained sklearn pipeline
        message (str): Text message to classify
        
    Returns:
        tuple: (prediction, probability) - Classification result and confidence
        
    Example:
        >>> prediction, prob = predict_message(pipeline, "Free money! Call now!")
        >>> print(f"Prediction: {prediction}, Confidence: {prob:.3f}")
    """
    # Make prediction
    prediction = pipeline.predict([message])[0]
    
    # Get probability scores
    probabilities = pipeline.predict_proba([message])[0]
    
    # Get confidence (probability of predicted class)
    if prediction == 'ham':
        confidence = probabilities[0]  # Probability of ham
    else:
        confidence = probabilities[1]  # Probability of spam
    
    print(f"Message: '{message}'")
    print(f"Prediction: {prediction.upper()}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Ham probability: {probabilities[0]:.3f}")
    print(f"Spam probability: {probabilities[1]:.3f}")
    
    return prediction, confidence


def setup_nltk_data():
    """
    Download required NLTK data for text processing.
    
    Downloads stopwords corpus needed for text preprocessing.
    """
    try:
        # Try to access stopwords to see if already downloaded
        stopwords.words('english')
        print("NLTK stopwords already available.")
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
        print("NLTK stopwords downloaded successfully!")


# Example usage and main workflow
if __name__ == "__main__":
    print("NLP Spam Filter - Main Workflow")
    print("=" * 40)
    
    # Setup NLTK data
    setup_nltk_data()
    
    # Load dataset
    messages = load_dataset('SMSSpamCollection.csv')
    
    # Visualize data
    visualize_message_lengths(messages)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split_data(messages)
    
    # Create and train pipeline
    pipeline = create_spam_pipeline()
    print("\nTraining model...")
    pipeline.fit(X_train, y_train)
    print("Model training completed!")
    
    # Evaluate model
    metrics = evaluate_model(pipeline, X_test, y_test)
    
    # Example predictions
    print("\nExample Predictions:")
    print("-" * 30)
    
    test_messages = [
        "Hi, how are you doing today?",
        "Free entry to win £1000! Call now!",
        "Can you pick up milk on your way home?",
        "URGENT! You have won a prize. Text CLAIM to 12345"
    ]
    
    for msg in test_messages:
        predict_message(pipeline, msg)
        print()