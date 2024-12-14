# fast.ai: Practical Deep Learning for Coder

> One thing is clear -- it's important that we all do our best to understand
 this technology, because otherwise we'll get left behind!

> What does DL require? Not advanced mathematical or years of study. It requires
> ommon sense and tenacity.

> We now have what Rosenblatt promised: “a machine capable of perceiving,
> recognizing, and identifying its surroundings without any human training or
> control.”

> You should assume that whatever libraries and software you learn today will be
> obsolete in a year or two.

## Lesson 1: Getting Started

### fast.ai background

* Jeremy Howard. Inventor of ULMFit: key to the Transformers book.
* All residents at Tesla and OpenAI use this course for onboarding.

The course is taught top down. Starting with problems and diving into theory when you have context and motivation.

## Deep Learning

### A brief history

Patterned after neurons in the brain, neural networks are made up of layers of neurons.

* A set of processing units (neurons)
* Each neuron has
  * A state of activation
  * Output function
  * An activation rule
* A pattern of connectivity among units
* Propagation rule
* Learning rule
* Environment


### Automatic Feature Creation

The key difference between neural networks and traditional ML models (trees, regression) is **feature creation**. Neural networks automatically learn progressively more complex features.

2012: Breast cancer research: 1000s of features into logistic regression to predict survival.

Colors -> edges -> corners -> circles -> letters -> words -> sentences

### pytorch

fast.ai is a library built on top of `pytorch`. Why `pytorch`?

* Tensorflow popularity is dying in 2021. - Ryan O'Connor.
* `torch` dominates research, which translates into industry.

Most NNs  have the same structure. The data is what's important.

* `fast.ai` uses `DataBlocks` to determine what type of NN should be used.
* `fast.ai` is highly functional. It requires functions.
* `DataBlocks` / `DataLoaders` are iterators that `pytorch` can use
* `Learner` combines a model (NN function) and data.

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

NN training loop:

* forward propogation: inputs -> model -> results -> loss
* back propogation: loss -> gradients -> update weights

> We can solve anything given enough time and data.

### Machine Learning

What is `Machine Learning`? The ability for a machine to learn from data how to solve problems.

* Assign weights
* Assess performance
* Update weights (learning rate)

A trained model can be treated just like a regular computer program.

> inputs -> model -> output

#### What is a Neural Network?

A neural network is a mathematical function. A mathematical proof called the
`universal approximation theorem` shows that this function can solve any problem
to any level of accuracy, in theory.

Stochastic Gradient Descent (`SGD`) is the process of updating weights
automatically by "descending" to the lowest error.

* Architecture: The model template (layers). The mathematical function applied to parameters.
* Weights (parameters)
* Model: The combination of the architecture and parameters (weights).
* Fit (Train): Updating weights to get accurate predictions.
* Data (independent variables)
* Labels (dependent variable / answer)
* Predictions (model results)
* Performance (loss)

Limitations of Machine Learning:

* Requires `labeled data` to learn.
* The model only creates predictions, not recommended actions

Watch out for `feedback loops`. A feedback loop can create bias and over
emphasize certain scenarios. The model may be used to make predictions, causing
the model to be even more biased.

* Classification: Predicts one class out of a number of discrete possibilities.
* Regression: Predicts a numeric quantity.

```python
def is_cat(x):
  """The dataset is created using the following pattern:

  [species]_[n].jpg

  Where species is the species of cat or dog. If cat, the species is capitalized.

  Examples:
    Scottish_Fold_1.jpg         # cat
    American_Shorthair_12.jpg   # cat
    goldern_retriever_100.jpg   # dog
  """
  return x[0].isupper()

# fast.ai will always show accuracy using the validation set
dls = ImageDataLoaders.from_name_func(
    path,
    get_image_files(path),
    valid_pct=0.2,  # hold out 20% for the validation set / keep 80% for training
    seed=42,
    label_func=is_cat,
    item_tfms=Resize(224)
)

# a fast.ai vision learner creates a CNN and specifies what architecture to use
# and what metric to use. CNNs are the state of the art for vision.
#
#
# Architecture
# ------------
# The model architecture in practice is unlikely to be something you spend much time on.
# resnet34 uses 34 layers. The more layers the more potential accuracy but also the
# more prone to overfitting.
#
# Metric
# ------
# Measures the quality of predictions using the validation set (printed @ each epoch)
#
# error_rate: % of images classified incorrectly
# accuracy: (1 - error_rate): % of images classified correctly
#
# loss != metric: metrics are meant for human consumption. loss is meant to use with SDG.
# You might decide that loss is a suitable metric, but not always
#
# resnet34 is a "pretrained model". In general, you should use pretrained models.
#
# vision_learner will remove the last layer which is specifically customized to the
# original training task, replacing it with one or more new layers with randomized weights,
# with an appropriate size for the dataset you are working with. The last part of the
# model is known as the `head`.
#
learn = vision_learner(dls, resnet34, metrics=error_rate)

# tells fastai how to `fit` the model.
# 1 == number of epochs
#
# fine_tune: A transfer learning technique where the parameters of the pretrained model
# are updated by training for additional epochs. The `head` layers are updated faster
# than the earlier layers.
learn.fine_tune(1)
```
#### Overfitting

You can determine overfitting if the validation error increases during training.

#### Transfer Learning

Using a pretrained model for a different task than it was originally trained for
is called `transfer learning`.



### Types of Deep Learning

#### Segmentation

Segmentation "segments" an image into known pieces. For example, being able to
identify an object.

#### NLP

Text generation, translate, summarize.

#### Tabular Data

```python
from fastai.tabular.all import *
path = untar_data(URLs.ADULT_SAMPLE)

dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
    cat_names = ['workclass', 'education', 'marital-status', 'occupation',
                 'relationship', 'race'],
    cont_names = ['age', 'fnlwgt', 'education-num'],
    procs = [Categorify, FillMissing, Normalize])

learn = tabular_learner(dls, metrics=accuracy)

# pretrained models are rarely available for tabular data
# fit_one_cycle trains a model from scratch
learn.fit_one_cycle(3)
```

### The importance of data

Most practioners use subsets of data to experiment and prototype, using the full
dataset once they are confident in their approach.

### Training | Validation | Test sets

* Training set: training data.
* Validation set: used for evaluating hyperparameters.
* Test set: completely unseen data used to validate model quality.

When doing hyperparameter tuning, we are tuning based on the validation set. We
don't want to overfit the validation data when hyperparameter tuning.

Test data is fully hidden from our eyes and is the ultimate judge of model
performance. A test set is not required if you have a little data, but it's
generally better to have one

IMPORTANT: Your validation and test sets must be representative of your
real-world data.

* For time series, use the earlier data as training data / latest data as
  validation.
* For pictures with the same person, ensure the person is either in the training
  or validation sets, but not both


### Questionnaire

1. Do you need these for deep learning? (no)
2. Name 5 areas where DL is now the best in the world
   1. NLP
   2. Computer visiystems
   3. Medicine (x-ray)
   4. Biology (genes)
   5. Image generation
   6. Recommendation systems
   7. Games
3. What was the name of the first device that was based on the principle of the artificial neuron?
   1. Mark I Perceptron by Frank Rosenblatt
4. Based on the book of the same name, what are the requirements for parallel distributed processing (PDP)?
   1. A set of processing units
   2. A state of activation
   3. An output function for each unit
   4. A pattern of connectivity among units
   5. A propagation rule for propagating patterns of activites through the network of connectivities
   6. An activation rule for combining the inputs impinging on a unit with the current state of that unit to produce an output for the unit
   7. A learning rule whereby patterns of connectivity are modified by experience
   8. An environment in which the system must operate
5. What were the two theoretical misunderstandings that held back the field of neural networks?
   1. Early researchers didn't appreciate or apply the principal that more layers produce better results.
6. What is a GPU?
   1. A highly parallel processor which can handle 1000s of single tasks at a time.
7. Open a notebook and execute a cell containing: 1+1. What happens?
   1. The cell is executed in a `kernel` and the result (2) is printed.
8. Follow through each cell of the stripped version of the notebook for this chapter. Before executing each cell, guess what will happen.
9. Complete the Jupyter Notebook online appendix.
10. Why is it hard to use a traditional computer program to recognize images in a photo?
    1.  Hard to define rules which classify an image from all different shapes / textures / colors.
11. What did Samuel mean by "weight assignment"?
    1.  The current state of values for model parameters
    2.  A means for testing weight effectiveness.
    3.  A mechanism for altering weights to maximize performance.
12. What term do we normally use in deep learning for what Samuel called "weights"?
    1.  Model `parameters`
13. Draw a picture that summarizes Samuel's view of a machine learning model.
14. Why is it hard to understand why a deep learning model makes a particular prediction?
    1.  They can have many layers and many neurons which interact.
15. What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?
    1.  Universal approximation theorem
16. What do you need in order to train a model?
    1.  Architecture / labeled training and validation data / metric / SGD (a way to update weights)
17. How could a feedback loop impact the rollout of a predictive policing model?
    1.  The feedback loop could reinforce bias on certain data.
18. Do we always have to use 224×224-pixel images with the cat recognition model?
    1.  No
19. What is the difference between classification and regression?
    1.  Classification predicts 1 of n classes. Regression is continuous.
20. What is a validation set? What is a test set? Why do we need them?
    1.  A validation set is used to determine how accurate the model is during training.
    2.  A test set is used to determine how accurate the model is *after* training.
21. What will fastai do if you don't provide a validation set?
    1.  Create one with `0.2`.
22. Can we always use a random sample for a validation set? Why or why not?
    1.  No. The validation set should represent the training set.
    2.  Different data sets (time series / people) will require different validation set strategies
23. What is overfitting? Provide an example.
    1.  Where the model memorizes the training data.
    2.  Example: Perfectly predicting every element in the training data. Doing poorly on new data.
24. What is a metric? How does it differ from "loss"?
    1.  A metric is how humans determine how good a model is. Loss is how a model determines how good a model is.
25. How can pretrained models help?
    1.  A head start
26. What is the "head" of a model?
    1.  Layers which are added to a pretrained model
27. What kinds of features do the early layers of a CNN find? How about the later layers?
    1.  Primitive features (like lines / gradients). Later layers find more complex entities.
28. Are image models only useful for photos?
    1.  No. Other data can be represented as images.
29. What is an "architecture"?
    1.  A model blueprint
30. What is segmentation?
    1.  Classifying parts of an image as a segment (i.e., car, tree)
31. What is y_range used for? When do we need it?
    1.  The range the target values have.
32. What are "hyperparameters"?
    1.  Parameters which control the training process itself.
33. What's the best way to avoid failures when using AI in an organization?
    1.  Define training, validation, and test sets properly.
    2.  Define a baseline of what success looks like on a truly unseen test set.


--

## Lesson 2: Deployment

x