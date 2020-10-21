# Data Science From Scratch - Joel Grus

Data is growing unbounded. The goal of DS is to turn that data into insight.

The most important traits to becoming a data scientist are:

* Inquisitive
* Growth mindset
* Hard work

Foundational concepts:

* Probability
* Statistics
* Linear Algebra

Libraries:

* NumPy
* scikit-learn
* pandas

Python is not the best language, but it's great for quick prototyping and bootstrapping.

## Chapter 1: Introduction

An example of simple data analysis to spot trends and relationships in data.

## Chapter 2: A Crash Course in Python

An elementary introduction to python.

## Chapter 3: Visualizing Data

A tour of `matplotlib` with line / bar chart / scatterplot examples.

## Chapter 4: Linear Algebra

Linear algebra is the branch of mathematics that deals with vector spaces. Vectors are lists.j

## Chapter 5: Statistics

### Centrality

Where is the data centered?

* `mean` : average
* `median` : the middle value (if odd) or mean of the two middle-most values (if even)
* `mode`: the value occurring most often
* `quantile` : the value less than which a certain percentile of the data lies (The median is 50%)

The mean is sensitive to outliers, leading to misrepresentation.

### Dispersion

How spread out is the data?

* `range` : the difference between the largest and smallest element
* `variance` : the average squared deviation from the mean
* `standard deviation` : the square root of the variance
* `interquartile range` : the middle 50% (the difference between Q1 (25%) and Q3 (75%))

### Correlation

* `covariance` - how close two variables are from their mean.
  * Large covariance means x and y are large together or small together (strongly related)
  * Small covariance means no relationship exists.

* `correlation` - how close two variables are in terms of standard deviations

Know your data! Beware of Simpson's Paradox, which states correlations can be
misleading when confounding variables are ignored. A confounding variable is one
which, if included in the analysis, drastically changes the correlation outcome.

For example, people from the east coast may have more friends than people on the
west coast. However, if you break up the data set by gender, you may find that
people on the west coast actually have more friends. Gender is the confounding
variable that needs to be accounted for in the analysis.

## Chapter 6: Probability

Probability is a way of quantifying uncertainty.

### Dependence

Two events are *dependent* if knowing the outcome of one event gives us
information about another event.

Mathematically, we say two events E and F are independent if the probability
they both happen is the product of the probabilities each one happens:

P(E, F) = P(E) * P(F)

> The probability of events E and F happening equals the probability of E
> happening by itself times the probability of F happening by itself.

### Conditional probability

P(E | F) = P(E, F) / P(F)
P(E, F) = P(E | F) * P(F)

> The probability of E occurring given F has occurred equals the probability of
> both E and F occurring / the probability that F occurs.

### Distribution

Distribution is the probability of a value falling within a range of possible
values. A cumulative distribution function is the probability that a random
variable is less than or equal to a certain value.

For example, given uniform distribution of values between 0 and 1, the
cumulative distribution function is:

```python

def uniform_cdf(x):
  """Returns the probability that a uniform random variable is <= x"""
  if x < 0: return 0     # uniform random is never < 0
  elif x < 1: return x   # P(X <= 0.4) == 0.4
  else return 1          # uniform random is never > 1
```

## Chapter 7: Hypothesis and Inference

* Significance: How willing are you to make a type 1 error (false positive) (1%? 5%?)
* Power: How willing are you to make a type 2 error (false negative)

* Bernoulli Trial: The result is a randome variable that can be approximated using the normal distribution

Example: Coin Flipping

* Null (default) hypothesis: The coin is fair (p = .5)
* Alternative hypothesis: The coin is *not* fair (p != .5)


## Chapter 8: Gradient Descent

Gradient descent attempts to find the minimum value (low point) of a function by
iteratively moving in the direction of the steepest descent as defined by the
negative of the gradient.

Loss (Cost) function: tells us "how good" our model is at making predictions for a
given set of parameters. The cost function has it's own curve and gradients. The
slope of this curve tell sus how to update our parameters to make the model more
accurate.


## Chpater 9: Getting Data

I/O, CSV, HTML, JSON, HTTP (Github / Twitter APIs)

## Chapter 10: Working with Data

### Exploration

Before building models, *explore* the data.

* Summary statistics, smallest, largest, min, max, mean, median, mode
* Histogram (grouping data into buckets)
* Scatter plots

### Cleaning

* Remove invalid values (`null`, `0`, `year == 3000`, etc)
* Check for outliers
* Reformat data: put into a form condusive to analysis (example: group data by stock ticker)

### Rescaling

Analyzing data with the wrong scale can produce different results.

When dimensions aren't compatible with each other (e.g., height, weight) we'll
*rescale* the data so that each dimension has a mean 0 and standard deviation of
1.0. This effectively gets rid of the units, converting each dimension to
"standard deviations from the mean".

### Dimensionality Reduction

Remove dimensions to focus on the data that captures the most variation. Some
dimensions can be redundant. Removing redundant dimensions allows you to focus
on only those which cause variance.

## Chapter 11: Maching Learning


What is a model? It's a specification of a mathematical relationship between different variables.

What is ML? Creating models that are *learned from data* to predict outcomes.

* Supervised: model created with a known-good labeled data set (training data)
* Unsupervised: model created without training data
* Online: model consistently ajusting in real time

### Overfitting and Underfitting

* Overfitting is creating a model which performs well on training data / poor in real world.
* Underfitting is producing a model that doesn't perform well on training data (usually not used in production)

Models that are too complex lead to overfitting.

* True positive - true prediction is correct
* False positive (type 1 error) - predicted true, actually false
* False negative (type 2 error) - predicted false, actually true
* True negative - false prediction is correct

Precison - how accurate were our true predictions?
Recall - of all true values, what percent did the model predict?

There is usually a tradeoff between precision and recall. A model that predicts
"yes" even if not very confident will have great recall, terrible precision (a
lot of false positives). A model that predicts "yes" only when very confident (a
lot of false negatives)

```python
def precision(true_pos: float, false_pos: float) -> float:
  """Measures how accurate our model identified positives"""
  return true_pos / (true_pos + false_pos)

def recall(true_pos: false, false_neg: float) -> float:
  """Measures what fraction of positives identified"""
  return true_pos / (true_pos + false_neg)
```

### Bias and Variance

* Bias - the degree to which a model doesn't fit the training data. A model that
  doesn't predict well has high bias.

* Variance - the degree to which a model's results vary across training data
  sets. A model that predicts similarily across training sets has low variance.

* High bias and low variance typically indicates the model is underfitting.
* Low bias and high variance typically indicates the model is overfitting.

* What should you do if your model has high bias? Try adding more features.

* Holding model complexity constant, the more data you have the harder it is to overfit.
* More data will *not* help with bias. If your model doesn't use enough features
  to capture regularities in the data, throwing more data at it won't help.

### Feature Extraction and Selection

* Features are model inputs. They fall into a few categories: bool, float, enum

* The type of features (bool, float, enum) constrain which model types you can use.

How do you choose features? Experience and domain expertise. Know your data.



### Chapter 12: kNN

Prediction based on neighboring elements. It makes no mathematical assumptions,
it simply looks at the relationship between two elements.

Where most models use the entire data set to help predict a result, kNN simply
needs the `k` closest data elements.

The "Curse of Dimensionality" means that as you increase dimensions, kNN will
not be as accurate. Points in high dimensional spaces tend not to be close to
one another at all. Points may be very close in some dimensions, wildly
different in others, making their overall distance "average".

In general, keep the number of dimensions small and meaningful.

`scikit-learn` has many kNN models.

### Chapter 13: Naive Bayes

Multiplies the probabilities of each distinct event. This algorithm is "naive"
because it assumes that each element occurring is independent of each other. For
example, a spam filter would use Naive Bayes to classify messages as spam using
known keywords like "cheap", "viagra", ...

* Spam = P(viagra) * P(cheap) * P(...)

### Chapter 14: Linear Regression

Finding the line of best fit. The goal is to make the "sum of squared error" as
low as possible. Gradient descent (consistently predicting and adjusting to find
the best estimate) can be used to create better results.

### Chapter 15: Multiple Regression

Using multiple variables in linear regression. The model gives us coefficients
for each variable - values representing how each variable impacts the result.

For example:

minutes spent online = 30.63 + 0.972 friends - 1.68 work hours + 0.911 phd

Means that on average, each extra friend corresponds to one more minute spent
online. Each hour spent working results in -1.68 minutes spent online, and
having a phd results in .911 more minutes online.

The variables are considered independent. If one variable impacts another (i.e.,
work hours impacts people with more friends more than with less friends), this
model won't tell us that.

You must pick features that are independent of each other.



