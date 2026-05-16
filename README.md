# SMS Spam Detection

This project classifies SMS messages as either legitimate (`ham`) or unwanted (`spam`) using natural language processing and machine learning.

The main goal is to show a clear machine learning workflow:

1. Load the SMS message dataset.
2. Explore the target classes and message text.
3. Clean and tokenize the text.
4. Convert messages into numeric features.
5. Train a text classification model.
6. Evaluate how well the model separates ham from spam.

## Dataset

The dataset is [SMSSpamCollection.csv](SMSSpamCollection.csv). It contains 5,572 SMS messages from the SMS Spam Collection dataset.

| Column | Meaning |
| --- | --- |
| `label` | Message class: `ham` or `spam` |
| `message` | Raw SMS message text |

Class balance:

| Class | Messages |
| --- | ---: |
| `ham` | 4,825 |
| `spam` | 747 |

The dataset is imbalanced: most messages are legitimate. That matters because a high accuracy score can hide weak spam detection, so the project also reports precision, recall, F1-score, and a confusion matrix.

## Project Files

| File | Purpose |
| --- | --- |
| [nlp.ipynb](nlp.ipynb) | Guided notebook with exploration, preprocessing, training, and evaluation |
| [train_model.py](train_model.py) | Clean reproducible training script |
| [SMSSpamCollection.csv](SMSSpamCollection.csv) | SMS dataset used for modeling |
| [requirements.txt](requirements.txt) | Python dependencies |

## Notebook Workflow

The notebook is organized as a guided walkthrough:

1. Import libraries.
2. Load the dataset with a portable relative path.
3. Inspect rows, shape, labels, missing values, and duplicates.
4. Understand the target balance between ham and spam messages.
5. Explore message length patterns by class.
6. Define text preprocessing for punctuation and stopwords.
7. Split the data into training and test sets.
8. Build a scikit-learn text classification pipeline.
9. Train and evaluate the model.
10. Try sample predictions on new SMS messages.

## Quick Start

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the reproducible training script:

```bash
python train_model.py
```

Or open the notebook:

```bash
jupyter notebook nlp.ipynb
```

When using VS Code, select the project virtual environment as the notebook kernel before running cells.

## Modeling Approach

The training script uses a scikit-learn pipeline:

- Removes punctuation from each message.
- Converts text to lowercase tokens.
- Removes common English stopwords.
- Uses `CountVectorizer` to build bag-of-words features.
- Uses `TfidfTransformer` to weight words by importance.
- Trains a `MultinomialNB` classifier.
- Evaluates on a held-out stratified test set.

The test split uses `random_state=42`, so the output is reproducible.

Example output:

```text
SMS Spam Detection Model
========================
Rows: 5,572
Class counts:
ham     4825
spam     747

Accuracy: 0.963
Confusion matrix (labels: ham, spam):
[[1448    0]
 [  62  162]]
```

In beginner-friendly terms, the model is correct about 96% of the time. It is especially conservative about marking messages as spam: when it predicts `spam`, it is very precise, but it still lets some spam messages through. That is a reasonable tradeoff for a spam filter because incorrectly hiding a real message can be more costly than allowing some spam into the inbox.

## Acknowledgments

- SMS Spam Collection Dataset from the UCI Machine Learning Repository.
- scikit-learn for the machine learning pipeline tools.
- NLTK for English stopword handling.
