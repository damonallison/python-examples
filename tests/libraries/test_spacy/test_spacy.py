"""
Natural Language Processing is the discipline of finding meaning in text data.
Examples include topic identification (what's the main point?), chat bots, text
classification, and sentiment analysis.

Before you can analyze text, you probably need to preproces it - including
tokenization. Regular expressions are a powerful form of text search. A regular
expression allows you to find patterns in text. Tokenization is the process of
converting a piece of text into tokens. For example, into sentences, words,
email addresses, hashtags, or any other token you'd like to identify.

Other text processing include lemmatization (finding base words, or lemmas),
Named Entity Recognition - identifying an object (entity) within text. A person,
place, organization, date, hashtag, email address, etc, finding sentiment in
text, etc.

There are multiple NLP libraries. The `nltk` library (Natural Language Toolkit)
has multiple tokenizers (i.e., regex, word, sentence) for tokenizing text.
`gensym` is another document vectorizer. spacy is a collections of pipelines and
uses pre-trained models.

---

Tf-idf (term frequency * inverse document frequency) is a simple formula for
determining which words are important in a document. The more the term is used
in the document, and less it's used across all documents, makes it more
important. Common words or words used across all documents are less unique and
therefore less important.

Tfidf weights make great feature inputs for modeling.

Named Entity Recognition (NER) is the process of identifying entities (people,
places, things) and the parts of speech (adjective, adverb, etc). `nltk`,
`spacy` and `polyglot` can be used for NER.

Once you have features identified, you are ready for modeling. Naive Bayes is an
excellent algorithm for text classification using "bag of words" since it
expects integer value inputs. Using NB allows us to classify text. For example,
determining what category an article relates to.


spacy
-----

A spacy pipeline has the following steps:

text -> tokenizer -> tagger (pos) -> parser -> ner -> [more ...] -> doc

"""

"""
IMPORTANT: Before you can get started with spacy, you need to download a trained
pipeline.

> python -m spacy download en_core_web_em

> python -m spacy download en_core_web_md


"""

import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest
import spacy
import spacy.tokens
from sklearn.decomposition import PCA
from spacy import displacy  # document visualizer

NLP = spacy.load("en_core_web_sm")
NLP_MD = spacy.load("en_core_web_md")


def test_tokenizer() -> None:
    doc = NLP("My daughter, Lily, is in France.")

    words = ["My", "daughter", "Lily", "is", "in", "France"]
    assert words == [token.text for token in doc if token.is_alpha]

    puncs = [",", ",", "."]
    assert puncs == [token.text for token in doc if token.is_punct]

    assert len(list(doc.sents)) == 1

    s: spacy.tokens.Span = list(doc.sents)[0]
    assert s.text == "My daughter, Lily, is in France."


def test_linguistic_features() -> None:
    """Linguistc features provide meaning to text.

    It's not enough to break down a document into words. We want to extract
    *meaning*.

    Part of speech (PoS)
    --------------------
    Classifies each word into a part of speech - like noun, verb, adjective,
    conjuction. Each token has a `pos_` property which contains it's pos.

    Named Entity Recognition (NER)
    ------------------------------
    A word or phrase that finds "entities" (i.e., people, places, things) and
    categorizes them. Each doc has a `.ents` property which holds it's entities.

    """
    # pos - part of speech - english has 9 (noun, verb, conjunction, etc) - confirms the meaning of a
    # ner
    verb_sent = "I watch TV."
    noun_sent = "I left without my watch."

    for sent in [verb_sent, noun_sent]:
        print(
            [
                (token.text, token.pos_, token.ent_type_, spacy.explain(token.pos_))
                for token in NLP(sent)
            ]
        )

    # Examine entities
    doc = NLP("My daughter, Lily Allison, is in France.")
    print([(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents])

    # Visualization
    with tempfile.NamedTemporaryFile("w", delete=True, delete_on_close=False) as f:
        f.write(displacy.render(doc))
        print(f"saved svg to {f.name}")

    # displacy.serve(doc, style="dep", auto_select_port=True)
    # displacy.serve(doc, style="ent", auto_select_port=True)


def test_word_sense_disambiguation() -> None:
    """
    Words mean different things. Depending on the sense of the word (the POS),
    we will treat the word differently. For example, the verb "to fish" is
    translated differently in spanish than the noun "the fish".
    """
    verb_sent = "I will fish tomorrow."
    noun_sent = "I ate fish."

    for sent in [verb_sent, noun_sent]:
        print(
            [
                (token.text, token.pos_, token.ent_type_, spacy.explain(token.pos_))
                for token in NLP(sent)
                if "fish" in token.text.lower()
            ]
        )
        # displacy.serve(NLP(sent), style="dep", auto_select_port=True)


def test_dependency_parsing() -> None:
    """
    Dependnecy parsing - relationships between words (dependencies / links
    between token). The result is a tree.

    Common depenencies:
    ---------------------
    * nsub: nominal subject
    * root: root
    * det: determiner
    * dobj: direct object
    * aux:auxiliary
    """
    doc = NLP("My daughter, Lily Allison, is in France.")

    # entities
    print([(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents])

    # dependencies
    print([(token.text, token.dep_, spacy.explain(token.dep_)) for token in doc])

    # Visualization
    with tempfile.NamedTemporaryFile("w", delete=True, delete_on_close=False) as f:
        f.write(displacy.render(doc))
        print(f"saved svg to {f.name}")

    # display the dependency tree
    # displacy.serve(doc, style="dep", auto_select_port=True)
    # displacy.serve(doc, style="ent", auto_select_port=True)


def test_word_vectors() -> None:
    """
    Word vectors (word embeddings) are numerical representations of words. They
    provide more meaning by acting on words in multiple dimensions. They consder
    word frequences, and the presence of other words in similar contexts.

    Bag of words: {"I": 1, "got": 2} ...

    BoW does not give the word *meaning*.


    How do you produce word vectors?

    * word2vec
    * glove
    * fasttext
    * transformer based architectures

    *Some* spacy models have word vectors. `en_core_web_sm` does *not* have word
    vectors, while `en_core_web_md` has 300 dim vectors for 20k words.

    * nlp.vocab (access the vocab)
    * nlp.vocab.strings
    """
    print(NLP_MD.meta["vectors"])  # determine vector information
    print(NLP_MD.meta["vectors"]["width"])  # vector size for each word

    # find the id
    like_id = NLP_MD.vocab.strings["like"]

    # print the vector
    print(NLP_MD.vocab.vectors[like_id])
    print(len(NLP_MD.vocab.vectors[like_id]))


@pytest.mark.plot
def test_visualizing_vectors() -> None:
    """
    Use PCA to project words into a 2D space. You'll see how words relate.

    Word vectors allow us to find relationships between two words. For example,
    there is a relationship between "king" and "man" in a similar fashion that
    "queen" is to "woman". Relationships are called "analogies".
    """

    words = [
        "wonderful",
        "horrible",
        "apple",
        "banana",
        "orange",
        "watermelon",
        "dog",
        "cat",
    ]
    word_vectors = np.vstack(
        [NLP_MD.vocab.vectors[NLP_MD.vocab.strings[w]] for w in words]
    )

    pca = PCA(n_components=2)
    word_vectors_transformed = pca.fit_transform(word_vectors)

    plt.scatter(word_vectors_transformed[:, 0], word_vectors_transformed[:, 1])
    for word, coord in zip(words, word_vectors_transformed):
        x, y = coord
        plt.text(x, y, word, size=10)
    plt.show()


def test_find_similar_words() -> None:
    """
    Find semantically similar words.
    """

    word = "covid"
    word_vec = NLP_MD.vocab.vectors[NLP_MD.vocab.strings[word]]

    # finds the n most similar vectors to the target word
    similar_word_vecs = NLP_MD.vocab.vectors.most_similar(np.asarray([word_vec]), n=5)

    # [0][0] is the list of word_ids
    similar_words = [NLP_MD.vocab.strings[w] for w in similar_word_vecs[0][0]]
    print(similar_words)


def test_measuring_similarity() -> None:
    """
    Now that we have similar words, how similar are they?

    Why would you want to do this?

    * Find relevant content in a document
    * Detect duplicates (cheating)
    * Categorize like texts

    In order to find similarity, we calculate similarity scores using cosine
    similarity between two word vectors.
    """

    doc1 = NLP_MD("We eat pizza")
    doc2 = NLP_MD("We like to eat pasta")

    token1 = doc1[2]  # pizza
    token2 = doc2[4]  # pasta

    # token similarity - higher the score, the more related
    print(
        f"Similarity between {token1} and {token2} = {round(token1.similarity(token2), 3)}"
    )

    # span similarity = 0.588
    span1 = doc1[1:]  # eat pizza
    span2 = doc2[1:]  # like to eat pasta
    print(
        f"Similarity between {span1} and {span2} = {round(span1.similarity(span2), 3)}"
    )

    # 0.936
    span1 = doc1[1:]  # eat pizza
    span2 = doc2[3:]  # eat pasta
    print(
        f"Similarity between {span1} and {span2} = {round(span1.similarity(span2), 3)}"
    )

    # document similarity
    doc1 = NLP_MD("I like to play basketball")
    doc2 = NLP_MD("I love to play basketball")
    print(f"Similarity between {doc1} and {doc2} = {round(doc1.similarity(doc2), 3)}")

    # pretty similar scores, but radically different meaning (positive vs. negative)
    doc1 = NLP_MD("I like to play basketball")
    doc2 = NLP_MD("I don't like to play basketball")
    print(f"Similarity between {doc1} and {doc2} = {round(doc1.similarity(doc2), 3)}")

    # word / sentence similarity (finding similar questions to the word price)
    sentences = NLP_MD(
        "What is the cheapest flight from Boston to Seattle? \
        Which airline serves Denver, Pittsburgh, and Atlanta? \
        What kinds of planes are used by American Airlines?"
    )
    keyword = NLP_MD("price")

    for i, sentence in enumerate(sentences.sents):
        print(
            f"Similarity score with sentence {i+1} = {round(sentence.similarity(keyword), 5)}"
        )


def test_pipelines() -> None:
    """
    How to create and add pipeline components to a spacy pipeline.

    Many times an off the shelf pipeline isn't flexible enough or we want to
    expand it to include additional steps.

    For example, if you only want to use a piece of a pipeline, you need to
    construct a blank pipeline and add only the necessary steps. This is much
    more efficient than running the entire pipeline.
    """
    text = " ".join(["This is a sentence."] * 10000)
    blank_nlp = spacy.blank("en")
    blank_nlp.add_pipe("sentencizer")
    doc = blank_nlp(text)

    assert len(list(doc.sents)) == 10000

    # analyzes the pipeline to determine what the pipeline assigns, requires,
    # produces, any problems with the pipeline (i.e., one step doesn't set
    # required values for another step)

    # ============================= Pipeline Overview =============================

    # #   Component     Assigns               Requires   Scores    Retokenizes
    # -   -----------   -------------------   --------   -------   -----------
    # 0   sentencizer   token.is_sent_start              sents_f   False
    #                   doc.sents                        sents_p
    #                                                    sents_r

    # ✔ No problems found.

    print(blank_nlp.analyze_pipes(pretty=True))


def test_entity_ruler() -> None:
    pass
