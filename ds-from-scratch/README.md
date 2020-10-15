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
