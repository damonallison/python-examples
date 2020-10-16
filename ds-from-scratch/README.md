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



