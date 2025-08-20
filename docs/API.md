# API Documentation

This document provides detailed information about the functions and classes available in the NLP Spam Filter project.

## Core Functions

### Text Processing Functions

#### `text_process(mess)`

Processes raw text messages for spam classification by removing punctuation, converting to lowercase, removing stopwords, and tokenizing.

**Parameters:**
- `mess` (str): Input text message to process

**Returns:**
- `list`: List of cleaned and processed tokens

**Example:**
```python
message = "Free entry to win £1000! Call now!!!"
tokens = text_process(message)
print(tokens)  # ['free', 'entry', 'win', 'call']
```

**Implementation Details:**
- Removes all punctuation using `string.punctuation`
- Converts text to lowercase
- Splits into individual words
- Removes NLTK English stopwords
- Returns list of meaningful tokens

## Machine Learning Pipeline Components

### CountVectorizer Configuration

The project uses scikit-learn's `CountVectorizer` with custom configuration:

**Parameters:**
- `analyzer`: Custom `text_process` function
- Creates bag-of-words representation
- Converts text documents to token count matrix

### TfidfTransformer

Transforms count matrix to TF-IDF representation:
- **TF (Term Frequency)**: How often a term appears in a document
- **IDF (Inverse Document Frequency)**: How rare or common a term is across all documents
- Helps reduce impact of common words and emphasize distinctive terms

### MultinomialNB Classifier

Naive Bayes classifier specifically designed for discrete features:
- Assumes features are independent (naive assumption)
- Works well with text classification tasks
- Handles spam/ham binary classification effectively

## Pipeline Architecture

### Complete Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])
```

**Pipeline Steps:**
1. **Bag of Words (`bow`)**: Converts text to numerical features
2. **TF-IDF (`tfidf`)**: Applies term frequency weighting
3. **Classifier (`classifier`)**: Performs spam/ham classification

## Model Training and Evaluation

### Training Process

```python
# Split data
from sklearn.model_selection import train_test_split
msg_train, msg_test, label_train, label_test = train_test_split(
    messages['message'], messages['label'], test_size=0.3, random_state=101
)

# Train pipeline
pipeline.fit(msg_train, label_train)

# Make predictions
predictions = pipeline.predict(msg_test)
```

### Evaluation Metrics

The model provides comprehensive performance metrics:

```python
from sklearn.metrics import classification_report
print(classification_report(label_test, predictions))
```

**Available Metrics:**
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of actual occurrences of each class
- **Accuracy**: Overall correctness percentage

## Data Structures

### Input Data Format

Expected CSV format with tab separation:
```
label	message
ham	Go until jurong point, crazy.. Available only in bugis
spam	Free entry in 2 a wkly comp to win FA Cup final tkts
```

### Data Processing

```python
import pandas as pd
messages = pd.read_csv('SMSSpamCollection.csv', sep='\t', 
                      header=None, names=['label', 'message'])
```

**DataFrame Structure:**
- `label`: Classification label ('ham' or 'spam')
- `message`: Raw text content of SMS message
- `length`: Optional message length feature

## Visualization Functions

### Distribution Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Message length distribution
messages['length'] = messages['message'].apply(len)
plt.figure(figsize=(12, 5))

# Ham messages
plt.subplot(1, 2, 1)
messages[messages['label'] == 'ham']['length'].hist(bins=50)
plt.title('Ham Messages Length Distribution')

# Spam messages
plt.subplot(1, 2, 2)
messages[messages['label'] == 'spam']['length'].hist(bins=50)
plt.title('Spam Messages Length Distribution')
```

### Performance Visualization

The notebook includes confusion matrix and performance metric visualizations to assess model effectiveness.

## Error Handling

### Common Issues

1. **Missing NLTK Data**:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

2. **Empty Messages**:
   - The `text_process` function handles empty strings gracefully
   - Returns empty list for messages with no meaningful content

3. **Encoding Issues**:
   - CSV file should be read with appropriate encoding
   - Default UTF-8 encoding works for the SMS Spam Collection dataset

## Performance Considerations

### Memory Usage
- TF-IDF transformation can be memory-intensive for large datasets
- Consider using sparse matrices for very large text corpora

### Training Time
- Model training is relatively fast with Naive Bayes
- Most computational time spent in text preprocessing and vectorization

### Prediction Speed
- Real-time classification is very fast once model is trained
- Pipeline allows for efficient batch predictions

## Extension Points

### Custom Preprocessing
Override the `text_process` function to implement custom preprocessing:
```python
def custom_text_process(mess):
    # Add your custom preprocessing logic
    return processed_tokens
```

### Feature Engineering
Extend the pipeline with additional features:
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer

# Add numerical features alongside text features
preprocessor = ColumnTransformer([
    ('text', CountVectorizer(analyzer=text_process), 'message'),
    ('length', 'passthrough', ['length'])
])
```

### Model Alternatives
Replace MultinomialNB with other classifiers:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Alternative classifiers
pipeline_lr = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', LogisticRegression())
])
```