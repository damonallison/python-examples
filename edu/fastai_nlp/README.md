# fast.ai: Practical Deep Learning for Coder

> One thing is clear -- it's important that we all do our best to understand this
 technology, because otherwise we'll get left behind!

> What does DL require? Not advanced mathematical or years of study. It requires ommon sense and tenacity.

# # TODO

* PyTorch vs. JAX
* `fast.ai` libraries (used / popularoty)
* `pytorch` deep dive (`DataLoader`)
* `PILImage`  (pillow?)
* RISE: jupyter -> slides
* graphviz (textual graphical)

## Lesson 1: Getting Started

### fast.ai background

* Jeremy Howard. Inventor of ULMFit: key to the Transformers book.
* All residents at Tesla and OpenAI use this course for onboarding.

## Deep Learning

2012: Breast cancer research: 1000s of features into logistic regression to predict survival.

The key difference between neural networks and traditional ML models (trees, regression) is **feature creation**. Neural networks automatically learn progressively more complex features.

Colors -> edges -> corners -> circles -> letters -> words -> sentences

Wave forms and data can be turned into images

Tensorflow popularity is dying in 2021. - Ryan O'Connor

fast.ai is a library built on top of `pytorch`.

Most NNs used in problems have the same structure. The data is what's important. `fast.ai` uses `DataBlocks` to determine what type of NN should be used.

`fast.ai` is highly functional. It requires functions.

`DataBlocks` / `DataLoaders` are iterators that `pytorch` can use

`Learner` combines a model (NN function) and data.

`pth`: pytorch model (?)

Topics covered in this course:

* Model training
  * Vision
    * Image classification
    * Segmentation: separate pieces of an image into categories or "segments"
  * Tabular analysis: 2D data w/ label.
  * Collaborative filtering (recommendation system): similarity
  * NLP

If a human can do it reasonably quickly, DL will work. If the problem would require a lot of thought (winning an election), DL probably can't do it.

An NN is a mathematical function which combines inputs with weights. Weights initially are random.

Training loop

* forward propogation: inputs -> model -> results -> loss
* back propogation: loss -> gradients -> update weights

We can solve anything given enough time and data.

