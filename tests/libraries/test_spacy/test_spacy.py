"""
Natural Language Processing is the discipline of finding meaning in text data.
Examples include topic identification (what's the main point?), chat bots, text
classification, and sentiment analysis.

Before you can understand text, you need to preproces it by breaking it down
into tokens (tokenization) and finding entities like people, places, or things.

spacy is a pipeline based language processing library which includes prebuilt
pipelines trained on large amounts of data. It provides the ability to extend it
by writing custom pipelines.

Pattern Matching
-----------------
Regular expressions are a powerful form of text search. A regular expression
allows you to find patterns in text. Tokenization is the process of converting a
piece of text into tokens. For example, into sentences, words, email addresses,
hashtags, or any other token you'd like to identify.

Text Processing
----------------
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

import random
import re
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest
import spacy
import spacy.tokens
import spacy.training
from spacy import displacy  # document visualizer
from spacy.matcher import Matcher, PhraseMatcher

from sklearn.decomposition import PCA

try:
    NLP = spacy.load("en_core_web_sm")
    NLP_MD = spacy.load("en_core_web_md")
except:
    print(
        """spacy models are not downloaded.

          Run the following manually:

          python -m spacy download en_core_web_sm
          python -m spacy download en_core_web_md
          """
    )


pytestmark = pytest.mark.ml


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
    How to build a custom spacy pipeline. Why? Peformance, simplicity,
    customization.

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
    """
    EntityRuler adds named entities to a `Doc` container. It can be used on it's
    own or combined with EntityRecognizer.

    EntityPattern: assigns entities to matched patterns.

    There are two types of patterns:

    * Token pattern
        {"label": "GPE", "pattern": [{"LOWER": "san"}, {"LOWER": "francisco"}]}

    * Phrase entity patterns (exact string)
        {"label": "ORG", "pattern": "Microsoft"}
    """
    text = " ".join(
        ["Manhattan associates was started by Damon Allison in Redmond, Washington."]
    )

    # don't mess with the global, default NLP instance as we are modifying it
    nlp = spacy.load("en_core_web_sm")

    # Add our ruler *before* ner - our ruler should take priority over the
    # built-in NER ruler. Any token found by our ruler will not be considered
    # for NER.
    #
    # In our case, "Manhattan associates" will be found as an ORG and
    # "Manhattan" (GPE) will *not* be considered by the default NER pipeline
    # step since it's already used as a token in a previous step.
    #
    # Had we added our pipeline *after* NER, we would not have found "Manhattan
    # associates" as "Manhattan" would have already been used in a token.
    ruler = nlp.add_pipe("entity_ruler", before="ner")

    patterns = [
        # you can use whatever label you want - but you probably want to stick to
        # the labels spacy recognizes by default (i.e., PERSON, ORG, GPE, DATE, TIME, MONEY)
        # {"label": "PERSON", "pattern": [{"LOWER": "damon"}, {"LOWER": "allison"}]},
        {"label": "ORG", "pattern": [{"LOWER": "manhattan"}, {"LOWER": "associates"}]},
        {
            "label": "GPE",
            "pattern": [{"LOWER": "redmond"}, {"LOWER": ","}, {"LOWER": "washington"}],
        },
    ]
    ruler.add_patterns(patterns)

    doc = nlp(text)

    def contains(token_text: str, token_label: str) -> bool:
        return (token_text, token_label) in [(ent.text, ent.label_) for ent in doc.ents]

    assert len(doc.ents) == 3
    assert contains("Damon Allison", "PERSON")
    assert contains("Manhattan associates", "ORG")
    assert contains("Redmond, Washington", "GPE")


def test_regex() -> None:
    pattern = r"((\d{3})-(\d{3})-(\d{4}))"
    text = "My cell number is 612-912-1211 and my work number is 612-887-2392"

    # Returns all groups. The outer group is first, thin
    matches = re.findall(pattern, text)
    assert len(matches) == 2

    assert isinstance(matches[0][0], str)
    assert matches[0][0] == "612-912-1211"
    assert matches[0][1] == "612"
    assert matches[0][2] == "912"
    assert matches[0][3] == "1211"

    assert matches[1][0] == "612-887-2392"
    assert matches[1][1] == "612"
    assert matches[1][2] == "887"
    assert matches[1][3] == "2392"


def test_phone_regex() -> None:
    pattern = r"(\((\d){3}\)-(\d){3}-(\d){4})"
    text = "The number is (612)-765-4321"
    for match in re.finditer(pattern, text):
        print(match)
        print(type(match.start()))
        print(type(match.end()))


def test_regex() -> None:
    """
    Use regex to search / replace (links / emails). Regex is fast, well
    supported, but complex as it requires understanding the DSL.

    We can use regex patterns with EntityRuler
    """

    pattern = r"((\d{3})-(\d{3})-(\d{4}))"
    text = "My cell number is 612-912-1211 and my work number is 612-887-2392"

    # using finditer will return re.Match objects
    for match in re.finditer(pattern, text):
        assert isinstance(match, re.Match)
        # you would think that ".string" would return the matched string, but you'd be wrong.
        print(f"match: {match.group()} start: {match.start()} end: {match.end()}")
        # `.string` returns the string passed into find.
        assert match.string == text

    # using EntityRuler to add regex pattern search to a pipeline
    patterns = [
        {
            "label": "PHONE_NUMBER",
            "pattern": [
                {"SHAPE": "ddd"},  # regex?
                {"ORTH": "-"},  # literal
                {"SHAPE": "ddd"},
                {"ORTH": "-"},
                {"SHAPE": "dddd"},
            ],
        }
    ]
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)
    doc = nlp(text)

    ents = [ent.text for ent in doc.ents]
    assert ents[0] == "612-912-1211"
    assert ents[1] == "612-887-2392"

    # using a "REGEX" pattern
    text = "The number is 1234567890"
    pattern = r"(\d){10}"
    patterns = [{"label": "PHONE_NUMBERS", "pattern": [{"TEXT": {"REGEX": pattern}}]}]
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)
    doc = nlp(text)

    ents = [(ent.text, ent.label_) for ent in doc.ents]
    assert len(ents) == 1
    assert ents[0][0] == "1234567890"
    assert ents[0][1] == "PHONE_NUMBERS"


def test_matcher() -> None:
    """
    Regex is complex an difficult to debug.

    Matcher is a readable and production-level alternative to regex.
    """

    doc = NLP("Good morning, this is our first day on campus")
    matcher = Matcher(NLP.vocab)

    # include start and end token indices of the matched pattern
    pattern = [{"LOWER": "good"}, {"LOWER": "morning"}]
    matcher.add("morning_greeting", [pattern])

    # [(id, start_token, end_token)]
    matches = matcher(doc)

    assert len(matches) == 1
    start = matches[0][1]  # good
    end = matches[0][2]  # token after "morning"
    assert doc[start].text == "Good"
    assert doc[end - 1].text == "morning"
    assert doc[start:end].text == "Good morning"


def test_matcher_operators() -> None:
    """
    Regular expressions are difficult to debug. Spacy provides a higher level
    alternative with "Matcher" which is more explicit, easier to debug, but also
    more limited.

    Matcher allows you to include *some* operators similar to python's "in"
    statement:

    IN, NOT_IN, ==, >=, <, >
    """
    doc = NLP("Good morning and good evening")
    matcher = Matcher(NLP.vocab)
    pattern = [{"LOWER": "good"}, {"LOWER": {"IN": ["morning", "evening"]}}]
    matcher.add("morning_greeting", [pattern])

    matches = matcher(doc)
    assert len(matches) == 2


def test_phrase_matcher() -> None:
    """
    PhraseMatcher matches a long list of phrases in a given text.
    """

    terms = ["Bill Gates", "John Smith"]
    patterns = [NLP.make_doc(term) for term in terms]
    matcher = PhraseMatcher(NLP.vocab, attr="LOWER")
    matcher.add("PeopleOfInterest", patterns)
    doc = NLP("I'm sitting here with Bill Gates and John Smith")

    matches = matcher(doc)
    ents = [doc[match[1] : match[2]] for match in matches]
    assert [ent.text for ent in ents] == ["Bill Gates", "John Smith"]

    # You can also use the "SHAPE" attr class to find patterns by a shape.
    matcher = PhraseMatcher(NLP.vocab, attr="SHAPE")
    terms = ["123.0.0.0", "101.123.0.0"]
    patterns = [NLP.make_doc(term) for term in terms]
    matcher.add("IPAddress", patterns)
    doc = NLP("The IP address is 111.222.0.0")

    matches = matcher(doc)
    assert len(matches) == 1
    assert doc[matches[0][1] : matches[0][2]].text == "111.222.0.0"


#
# Customizing (training) spacy models.
#
# Most spacy models are used out of the box, however it is possible to train
# custom models (like twitter (hashtags) or custom domains (like medical data)).
#
# Because existing spacy NER models are generic, custom domains require special
# vocabularies, grammar, or entity recognition pipelines.
#
# Before buidling a custom model:
#
# 1. Determine if training is needed. Do the default models perform well enough
#    on our data?
#
# 2. Does our domain contain labels which spacy doesn't recgnize (i.e., drug
#    types in medical data)
#
# If we determine we need custom model training, we have to create a training
# data set and determine if we want to use a completely new model or update an
# existing spacy model.
#


def test_entity_recognition() -> None:
    """
    Assume here we are trying to find product entities. Notice the default spacy
    model assumes "Jumbo Salted Peanuts" and "Jumbo" are recognized as "PERSON"
    entities. Thus, we need to train a new model or augment the default spacy
    model to give it information about "PRODUCT"s.
    """
    texts = [
        "Product arrived labeled as Jumbo Salted Peanuts.",
        "Not sure if the product was labeled as Jumbo.",
    ]
    ents = []
    for d in [NLP(text) for text in texts]:
        ents.extend([(ent.text, ent.label_) for ent in d.ents])

    assert all([ent[1] == "PERSON" for ent in ents])

    # Training steps (general steps):
    # ---------------------------------------------------
    # 1. Annotate and prepare input data.
    # 2. Initialize model weights.
    # 3. Predict a few examples with the current weights.
    # 4. Compare prediction with correct answers.
    # 5. Use optimizer to optimize weights.

    training_data = [
        (
            "I will visit you in Austin.",
            {"entities": [(20, 26, "GPE")]},
        ),
        (
            "I'm going to Sam's house.",
            {"entities": [(13, 18, "PERSON"), (19, 24, "GPE")]},
        ),
    ]

    # We need to create an Example object for each training sample
    for text, annotations in training_data:
        doc = NLP(text)
        example_sent = spacy.training.Example.from_dict(doc, annotations)
        print(example_sent.to_dict())


def test_custom_pipeline_component_training() -> None:
    """
    Here, we build a custom NER pipeline component.
    """
    # Training a pipeline component
    #
    # 1. Annotate input data
    # 2. Disable other pipeline components except NER (to isolate training)
    #

    nlp = spacy.blank("en")
    nlp.add_pipe("ner")

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    nlp.disable_pipes(*other_pipes)

    # train model over multiple epochs w/ weight adjustments using an optimizer
    training_data = [
        ("I will visit you in Austin.", {"entities": [(20, 26, "GPE")]}),
    ]

    optmizier = nlp.begin_training()
    epochs = 10
    # losses are the predictions, a number representing error
    losses = {}

    for i in range(epochs):
        random.shuffle(training_data)
        for text, annotations in training_data:
            doc = nlp.make_doc(text)
            example = spacy.training.Example.from_dict(doc, annotations)
            nlp.update([example], sgd=optmizier, losses=losses)
            print(losses)

    # save the model
    ner = nlp.get_pipe("ner")
    ner.to_disk("model.mdl")

    # load the model
    nlp = spacy.blank("en")
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    nlp.disable_pipes(*other_pipes)

    ner = nlp.create_pipe("ner")
    ner.from_disk("model.mdl")

    # nlp.add_pipe(ner, "model.mdl")

    # # use the model
    # doc = nlp(text)
    # entities = [(ent.text, ent.label_) for ent in doc.ents]
    # print(entities)
