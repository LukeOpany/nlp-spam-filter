# NLP Spam Filter

A machine learning-powered SMS spam detection system built with Python and scikit-learn. This project uses Natural Language Processing techniques to classify SMS messages as either spam or legitimate (ham) messages with 96% accuracy.

## 🚀 Features

- **High Accuracy**: Achieves 96% overall accuracy on SMS classification
- **Robust Pipeline**: Complete ML pipeline with text preprocessing, feature extraction, and classification
- **Text Preprocessing**: Comprehensive text cleaning including punctuation removal and stopword filtering
- **TF-IDF Vectorization**: Advanced feature extraction using Term Frequency-Inverse Document Frequency
- **Naive Bayes Classification**: Efficient MultinomialNB classifier optimized for text classification
- **Easy to Use**: Simple Jupyter notebook interface for training and prediction
- **Comprehensive Dataset**: Trained on 5,572 SMS messages (4,825 ham, 747 spam)

## 📊 Model Performance

Our spam detection model achieves excellent performance metrics:

- **Overall Accuracy**: 96%
- **Ham (Legitimate) Messages**:
  - Precision: 95%
  - Recall: 100%
  - F1-Score: 98%
- **Spam Messages**:
  - Precision: 100%
  - Recall: 69%
  - F1-Score: 81%

## 🛠 Technology Stack

- **Python 3.x** - Core programming language
- **Jupyter Notebook** - Interactive development environment
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning library
- **matplotlib & seaborn** - Data visualization
- **nltk** - Natural Language Toolkit for text processing
- **numpy** - Numerical computing

## 📁 Project Structure

```
nlp-spam-filter/
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── nlp.ipynb                  # Main Jupyter notebook with ML pipeline
├── SMSSpamCollection.csv      # SMS dataset (5,572 messages)
├── CONTRIBUTING.md            # Contribution guidelines
└── docs/                      # Additional documentation
    ├── METHODOLOGY.md         # ML approach and methodology
    ├── DATASET.md            # Dataset information and structure
    └── API.md                # Code components documentation
```

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/LukeOpany/nlp-spam-filter.git
   cd nlp-spam-filter
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (required for stopwords):
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## 🚀 Quick Start

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook nlp.ipynb
   ```

2. **Run all cells** to:
   - Load and explore the SMS dataset
   - Preprocess text data (remove punctuation, stopwords)
   - Create TF-IDF feature vectors
   - Train the Naive Bayes classifier
   - Evaluate model performance

3. **Predict new messages**:
   ```python
   # Example prediction
   new_message = ["Congratulations! You've won a $1000 gift card. Click here to claim!"]
   prediction = pipeline.predict(new_message)
   print(f"Prediction: {prediction[0]}")  # Output: 'spam'
   ```

## 📈 Dataset Information

The project uses the **SMS Spam Collection Dataset**:
- **Total Messages**: 5,572
- **Legitimate Messages (Ham)**: 4,825 (86.6%)
- **Spam Messages**: 747 (13.4%)
- **Source**: UCI Machine Learning Repository
- **Format**: Tab-separated values with label and message columns

### Sample Messages:
- **Ham**: "Ok lar... Joking wif u oni..."
- **Spam**: "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!"

## 🔧 ML Pipeline Overview

1. **Data Loading**: Load SMS messages from CSV file
2. **Text Preprocessing**: 
   - Remove punctuation
   - Convert to lowercase
   - Remove stopwords (common words like 'the', 'and', etc.)
3. **Feature Extraction**: Convert text to TF-IDF vectors
4. **Model Training**: Train MultinomialNB classifier
5. **Evaluation**: Assess performance on test set (30% split)

## 📊 Usage Examples

### Training the Model
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# Create and train pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(msg_train, label_train)
```

### Making Predictions
```python
# Predict on new messages
test_messages = [
    "Free entry in 2 a wkly comp to win FA Cup final tkts",
    "Hey, are we still meeting for lunch today?"
]

predictions = pipeline.predict(test_messages)
print(predictions)  # ['spam', 'ham']
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- How to submit bug reports and feature requests
- Development setup and workflow
- Code style standards
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SMS Spam Collection Dataset from UCI Machine Learning Repository
- scikit-learn community for excellent ML tools
- NLTK team for natural language processing capabilities

## 📞 Contact

**Luke Opany** - [GitHub Profile](https://github.com/LukeOpany)

---

⭐ If you find this project helpful, please consider giving it a star!