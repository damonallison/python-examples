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
from sklearn import (
    datasets,
    feature_extraction,
    linear_model,
    model_selection,
    pipeline,
)


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

    #
    # CountVectorizer tokenizes the training data and builds the vocabulary.
    #
    # Each word becomes a key, it's ordinal value is the value
    #
    # {
    #   "the": 10,
    #   "fool": 4,
    #   ...
    # }
    #
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

    #
    # Words that appear in a single document are unlikely to be in our test set.
    # To eliminate these words, only include words that appear in `n` or more
    # documents using the `min_df` parameter.
    #
    vect = feature_extraction.text.CountVectorizer(min_df=2)
    vect.fit(corpus)

    v = vect.vocabulary_
    print(f"Vocab size: {len(v)}")
    print(f"Vocab: {v}")


def test_bow_movie_reviews() -> None:
    text_train, text_test, y_train, y_test = load_data("./data/aclImdb")

    vect = feature_extraction.text.CountVectorizer()
    vect.fit(text_train)
    X_train = vect.transform(text_train)

    print(f"X_train.shape: {X_train.shape}")
    print(f"Vocab length: {len(vect.vocabulary_)}")
    print(f"X_train: {repr(X_train)}")

    #
    # Since we have not done any preprocessing, CountVectorizer picks up all
    # kinds of "words", including numerics, multiple versions of the same word
    # (draw, drawing, draws). The default regex used to split words is
    # "\b\w\w+\b" where \b is a word boundary. \b is *not* included in the
    # match.
    #
    # \b = word boundary
    # \w = [A-Za-z0-9_]
    #

    #
    # Determine baseline performance before improving feature extraction
    #
    scores = model_selection.cross_val_score(
        linear_model.LogisticRegression(), X_train, y_train, cv=5
    )
    baseline_cv_mean = np.mean(scores)
    print(f"Default mean cv accuracy: {baseline_cv_mean:.2f}")

    #
    # Let's tune C to determine if we can improve on the baseline
    #
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10]}
    grid = model_selection.GridSearchCV(
        linear_model.LogisticRegression(), param_grid, cv=5
    )
    grid.fit(X_train, y_train)
    print(f"Tuned best cv score: {grid.best_score_}")
    print(f"Tuned best params: {grid.best_params_}")

    X_test = vect.transform(text_test)
    print(f"Tuned test score: {grid.score(X_test, y_test):.2f}")

    #
    # Let's try throwing away words that are in less than 5 documents, resulting
    # in a much smaller vocabulary (less features), to determine if model
    # performance suffers.
    #

    vect = feature_extraction.text.CountVectorizer(min_df=5)
    vect.fit(text_train)
    X_train = vect.transform(text_train)

    print(f"Trimmed X_train.shape: {X_train.shape}")
    print(f"Trimmed vocab length: {len(vect.vocabulary_)}")
    print(f"Trimmed X_train: {repr(X_train)}")

    grid = model_selection.GridSearchCV(
        linear_model.LogisticRegression(), param_grid, cv=5
    )
    grid.fit(X_train, y_train)
    print(f"Trimmed mean cv accuracy: {grid.best_score_}")

    #
    # Notice we didn't reduce accuracy while cutting the vocabulary count from
    # 3445861 to 27272. That proves that "unique" (or infrequent) words didn't
    # matter.
    #


def test_feature_engineering_remove_stop_words() -> None:
    #
    # Stop words are too frequent to be informative. (`the`, `is`, `a`, `this`)
    #
    # By removing the stop words (300+ words by default), we remove *some*
    # features but the overall test performance dropped a bit. It doesn't make
    # sense to omit stop words with this dataset.
    #
    # Stop words are helpful for small data sets, which don't contain a lot of
    # information for the model to determine which words are important.
    #

    stop_words = feature_extraction.text.ENGLISH_STOP_WORDS

    text_train, text_test, y_train, y_test = load_data("./data/aclImdb")

    vect = feature_extraction.text.CountVectorizer(min_df=5, stop_words=stop_words)
    vect.fit(text_train)
    X_train = vect.transform(text_train)

    print(f"X_train.shape: {X_train.shape}")
    print(f"Vocab length: {len(vect.vocabulary_)}")
    print(f"X_train: {repr(X_train)}")

    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10]}
    grid = model_selection.GridSearchCV(
        linear_model.LogisticRegression(), param_grid, cv=5
    )
    grid.fit(X_train, y_train)
    print(f"Tuned best cv score: {grid.best_score_}")
    print(f"Tuned best params: {grid.best_params_}")

    X_test = vect.transform(text_test)
    print(f"Tuned test score: {grid.score(X_test, y_test):.2f}")


def test_feature_engineering_tfidf() -> None:
    #
    # tf-idf is a way to rank features by importance (or weight).
    #
    # "The intuition of this method is to give high weight to any term that
    # appears often in a particular document, but not in many documents in the
    # corpus. If a word appears often in a particular document, but not in very
    # many documents, it is likely to be very descriptive of the content of that
    # document." - Introduction to Machine Learning - Chapter 7
    #
    # Words with a higher tf-idf score are considered more important.
    #
    # Because tf-idf uses statistical properties of the data, we will use a
    # pipeline to ensure the results of our grid search are valid.

    text_train, text_test, y_train, y_test = load_data("./data/aclImdb")

    pipe = pipeline.Pipeline(
        steps=[
            ("tfidf", feature_extraction.text.TfidfVectorizer(min_df=5)),
            ("lr", linear_model.LogisticRegression()),
        ]
    )
    param_grid = {"lr__C": [0.001, 0.01, 0.1, 1, 10]}
    grid = model_selection.GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
    grid.fit(text_train, y_train)
    print(f"Best CV score: {grid.best_score_:.2f}")

    #
    # Which words did tf-idf deem important?
    #
    vect: feature_extraction.text.TfidfVectorizer = grid.best_estimator_.named_steps[
        "tfidf"
    ]
    X_train = vect.transform(text_train)

    # Find the maximum value for each of the features over the dataset.
    max_value = X_train.max(axis=0).toarray().ravel()
    sorted_by_tfidf = max_value.argsort()
    feature_names = np.array(vect.get_feature_names())

    #
    # Features with low tf-idf are those that are very common across documents
    # or are used sparingly only in longer documents.
    #
    # Features with high df-idf appear often in particular reviews. They contain
    # specific information about the review.
    #
    print(f"Lowest tf-idf: {feature_names[sorted_by_tfidf[:20]]}")
    print(f"Highest tf-idf: {feature_names[sorted_by_tfidf[-20:]]}")

    # Which features have low idf? They appear frequently and are therefore
    # deemed less important.
    #
    # As expected, low idf words are mostly stop words. However even words you
    # think would be important for sentiment analysis like "good", "bad",
    # "movie" are among the most frequent and less relevant.
    sorted_by_idf = np.argsort(vect.idf_)
    print(f"Lowest idf: {feature_names[sorted_by_idf[:100]]}")
    print(f"Highest idf: {feature_names[sorted_by_idf[-100:]]}")

    #
    # Model coefficients: which features had the highest (most positive)
    # influence and lowest (most negative) influence?
    #
    lr: linear_model.LinearRegression = grid.best_estimator_.named_steps["lr"]
    coefs = np.array(lr.coef_)
    # argsort returns indices that index the data in sorted order.
    positive_coefs = np.argsort(coefs)[-20:]
    negative_coefs = np.argsort(coefs)[:20]
    print(f"Most positive features: {feature_names[positive_coefs]}")
    print(f"Most negative features: {feature_names[negative_coefs]}")


def test_bow_with_ngrams() -> None:
    #
    # n-grams add features for groups of words
    #
    # The main limitation of BoW is word order is completely discarded. "it's
    # bad, it's not good at all" and "it's good, it's not bad at all" have hte
    # same represntation but opposite meanings. Since words lose surrounding
    # context.
    #
    # Important movie bigram examples are:
    # 'not worth', 'well worth', 'definitely worth'
    #
    # You don't get that context from unigram (single word) BoW features.
    #
    # Note the feature space explodes with ngrams. bigrams == words^2, for
    # example. This is going to be expensive.
    #
    text_train, text_test, y_train, y_test = load_data("./data/aclImdb")

    vect = feature_extraction.text.CountVectorizer(ngram_range=(1, 3)).fit(text_train)
    print(f"Vect vocab: {len(vect.vocabulary_)}")

    pipe = pipeline.Pipeline(
        steps=[
            ("tfidf", feature_extraction.text.TfidfVectorizer(min_df=5)),
            ("lr", linear_model.LogisticRegression()),
        ]
    )
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "lr__C": [0.001, 0.01, 0.1, 1, 10],
    }
    grid = model_selection.GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
    grid.fit(text_train, y_train)
    print(f"Best CV score: {grid.best_score_:.2f}")
    print(f"Best params {grid.best_params_}")
