"""Working with text data in sklearn.

Setup
-----
The examples in this file use a dataset of movie reviews from the IMDb database
collected by Standord researcher Andrew Maas. The dataset contains the text
review and a "positive" or "negative" label. A review of 7+ is "pos" and <= 4 is
"neg". Neutral reviews are not in the dataset.

To download and unpack the training data for these examples (380MB total w/o
`unsup`):

wget -nc http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz -P data
tar xzf data/aclImdb_v1.tar.gz --skip-old-files -C data

# Remove unlabeled data (not used in these examples) rm -r
data/aclImdb/train/unsup

sklearn's `load_files` loads folders with the same folder structre as the test
data

/
    train/
        neg/ pos/
    test/
        neg/ pos/

Terminology
-----------

* "Document": A single text object (tweet / article / comment / writing).

* "Corpus": A set of documents.

* "Information Retrieval": The science of searching information in a document or
  corpus. (i.e., Search) * https://en.wikipedia.org/wiki/Information_retrieval

* "Natural Language Processing" (NLP): Programming computers to understand text.


"""
import os

import numpy as np
import pandas as pd
from sklearn import datasets, feature_extraction, linear_model, model_selection


def load_data(dir: str) -> tuple[list[str], list[str], list[str], list[str]]:
    reviews_train = datasets.load_files(os.path.join(dir, "train"))
    reviews_test = datasets.load_files(os.path.join(dir, "test"))

    return (
        reviews_train.data,
        reviews_test.data,
        reviews_train.target,
        reviews_test.target,
    )


def test_inspect_dataset() -> None:
    X_train, X_test, y_train, y_test = load_data("./data/aclImdb")

    # Simple analysis of text_train
    assert isinstance(X_train, list)
    print(f"len of text_train: {len(X_train)}")

    idx = np.random.randint(0, len(X_train))
    print(f"Example review (idx={idx}): {X_train[idx]}")
    print(f"Samples per class: {np.bincount(y_train)}")

    # Simple preprocessing
    X_train = [doc.replace(b"<br />", b" ") for doc in X_train]
    X_test = [doc.replace(b"<br />", b" ") for doc in X_test]


def test_bag_of_words() -> None:
    """Bag-of-words (BoW) counts how often each word appears in a document.

    1. Tokenization. Tokenize each document.
    2. Vocabulary. Build a vocabulary of all words in the corpus. Each word is
       given a number.
    3. Encoding. For each document, count how often each vocabulary word exists
       in each document.

    Our numeric representation has one feature for each unique word in the
    dataset. With BoW, word ordering is completely irrelevant.
    """

    corpus = [
        "The fool doth think he is wise",
        "but the wise man knows himself to be a fool",
    ]

    # CountVectorizer tokenizes the training data and builds the vocabulary.
    vect = feature_extraction.text.CountVectorizer()
    vect.fit(corpus)

    v = vect.vocabulary_
    print(f"Vocab size: {len(v)}")
    print(f"Vocab: {v}")
    print(f"Feature names: {vect.get_feature_names()}")

    # Stored in a scipy sparse matrix (since most values will be 0)
    bow = vect.transform(corpus)
    print(f"Dense BoW: {bow.toarray()}")

    # Would we want to build a dense df from bow data w/ large vocabularies?
    df = pd.DataFrame(bow.toarray(), columns=vect.get_feature_names())
    print(df)

    # Words that appear in a single document are unlikely to be in our test set.
    # To eliminate these words, only include words that appear in `n` or more
    # documents using the `min_df` parameter.
    vect = feature_extraction.text.CountVectorizer(min_df=2)
    vect.fit(corpus)

    v = vect.vocabulary_
    print(f"Vocab size: {len(v)}")
    print(f"Vocab: {v}")


def test_bow_movie_reviews() -> None:
    X_train, X_test, y_train, y_test = load_data("./data/aclImdb")

    vect = feature_extraction.text.CountVectorizer(min_df=5)
    vect.fit(X_train)
    X_train = vect.transform(X_train)

    # Determine baseline performance before improving feature extraction
    scores = model_selection.cross_val_score(
        linear_model.LogisticRegression(), X_train, y_train, cv=5
    )
    baseline_cv_mean = np.mean(scores)
    print(f"Mean cv accuracy: {baseline_cv_mean:.2f}")

    # Let's tune C to determine if we can improve on the baseline
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10]}
    grid = model_selection.GridSearchCV(
        linear_model.LogisticRegression(), param_grid, cv=5
    )
    grid.fit(X_train, y_train)
    print(f"Best cv score: {grid.best_score_}")
    print(f"Best params: {grid.best_params_}")

    X_test = vect.transform(X_test)
    print(f"Test score: {grid.score(X_test, y_test):.2f}")
