# NLP Spam Filter

A machine learning project that uses Natural Language Processing techniques to classify SMS messages as spam or ham (legitimate messages). This project implements a spam detection system using scikit-learn's Multinomial Naive Bayes classifier with TF-IDF vectorization.

## Features

- **High Accuracy**: Achieves 96% overall accuracy on test data
- **Robust Text Processing**: Implements comprehensive text preprocessing including punctuation removal and stopword filtering
- **Visualization**: Includes data exploration and performance visualization
- **Pipeline Architecture**: Uses scikit-learn pipelines for efficient model training and prediction
- **Real-time Classification**: Can classify new SMS messages instantly

## Dataset Information

The project uses the SMS Spam Collection dataset containing:
- **Total Messages**: 5,574 SMS messages
- **Ham Messages**: 4,827 legitimate messages (86.6%)
- **Spam Messages**: 747 spam messages (13.4%)
- **Source**: UCI Machine Learning Repository

## Model Performance

| Metric | Ham | Spam | Overall |
|--------|-----|------|---------|
| Precision | 0.95 | 1.00 | 0.96 |
| Recall | 1.00 | 0.69 | 0.96 |
| F1-Score | 0.98 | 0.81 | 0.95 |
| Support | 1,446 | 226 | 1,672 |

## Technology Stack

- **Python 3.7+**: Core programming language
- **Jupyter Notebook**: Interactive development environment
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and pipelines
- **matplotlib & seaborn**: Data visualization
- **nltk**: Natural language processing toolkit
- **numpy**: Numerical computing

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/LukeOpany/nlp-spam-filter.git
cd nlp-spam-filter
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (required for text processing):
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

4. Start Jupyter Notebook:
```bash
jupyter notebook
```

5. Open `nlp.ipynb` to run the analysis

## Usage Examples

### Basic Classification

```python
# Load and preprocess the dataset
import pandas as pd
messages = pd.read_csv('SMSSpamCollection.csv', sep='\t', 
                      header=None, names=['label', 'message'])

# Train the model using the pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

# Fit the model
pipeline.fit(msg_train, label_train)

# Make predictions
predictions = pipeline.predict(['Free entry to win £1000!'])
print(predictions)  # Output: ['spam']
```

### Text Preprocessing

The project includes a comprehensive text processing function that:
- Removes punctuation
- Converts to lowercase
- Removes stopwords
- Tokenizes text

## File Structure

```
nlp-spam-filter/
├── nlp.ipynb              # Main analysis notebook
├── SMSSpamCollection.csv  # Dataset file
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── CONTRIBUTING.md       # Contribution guidelines
├── LICENSE               # MIT license
└── docs/                 # Detailed documentation
    ├── API.md           # Function documentation
    ├── METHODOLOGY.md   # Technical approach
    └── DATASET.md       # Dataset details
```

## Machine Learning Pipeline

The spam filter uses a three-stage pipeline:

1. **Bag of Words (CountVectorizer)**: Converts text into numerical features
2. **TF-IDF Transformation**: Weighs features by term frequency and inverse document frequency
3. **Multinomial Naive Bayes**: Classifies messages based on probabilistic features

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Setting up the development environment
- Code style standards
- Submitting issues and pull requests
- Testing procedures

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SMS Spam Collection dataset from UCI Machine Learning Repository
- scikit-learn library for machine learning tools
- NLTK library for natural language processing capabilities

## Contact

For questions or suggestions, please open an issue on GitHub.