"""
NLP: Natural Language Processing: Extracting meaning from text.

Transformers
------------
A model architecture for solving NLP problems including text generation.

A pipeline performs multiple steps:
* Preprocessing
* Inference
* Postprocessing


Todo:

* summarization
* text-generation
* question-answering

"""

import faulthandler

import sys
from typing import Any
import logging

import torch
import torch.nn.functional as F
import transformers
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

DEVICE = "cpu"

faulthandler.enable()


def env() -> None:
    logger.info("System version: %s", sys.version)
    logger.info("Torch version: %s", torch.__version__)
    logger.info("Transformers version: %s", transformers.__version__)


def test_torch() -> None:
    print(torch.rand(2, 2))


def sent_2(sents: list[str]) -> list[dict[str, Any]]:
    # Load model and tokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    results: list[dict[str, Any]] = []
    for sent in sents:
        inputs = tokenizer(sent, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the logits and apply softmax to obtain probabilities
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)

        # Get label names and their corresponding scores
        labels = model.config.id2label

        results.append(
            {labels[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        )

    return results


def sentiment_analysis(sents: list[str]) -> list[dict[str, Any]]:
    # a popular model used for sentiment analysis
    classifier = pipeline(
        "sentiment-analysis",
        device=DEVICE,
    )
    print(classifier(sents))
    return classifier(sents)


def zero_shot_classification(sent: str, labels: list[str]) -> dict[str, Any]:
    classifier = pipeline("zero-shot-classification")
    return classifier(sent, candidate_labels=labels)


if __name__ == "__main__":
    env()
    test_torch()
    print(
        sent_2(
            [
                "today is going to be a great day",
                "i really don't like this",
            ]
        )
    )

    result = sentiment_analysis(
        [
            "hello world",
        ]
    )
    print(result)
    # logger.info("%s: %s", type(result), result)
    # result = zero_shot_classification(
    #     sent="this is a course about machine learning and data science",
    #     labels=["education", "science", "math"],
    # )

    # print(result)
    # logger.info("%s: %s", type(result), result)
