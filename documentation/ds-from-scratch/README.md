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

Linear algebra is the branch of mathematics that deals with vector spaces. Vectors are lists.

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

* Bernoulli Trial: The result is a random variable that can be approximated using the normal distribution

Example: Coin Flipping

* Null (default) hypothesis: The coin is fair (p = .5)
* Alternative hypothesis: The coin is *not* fair (p != .5)


## Chapter 8: Gradient Descent

Gradient descent attempts to find the minimum value (low point) of a function by
iteratively moving in the direction of the steepest descent as defined by the
negative of the gradient.

Loss (Cost) function: tells us "how good" our model is at making predictions for a
given set of parameters. The cost function has it's own curve and gradients. The
slope of this curve tells us how to update our parameters to make the model more
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

Regression is the process of finding the relationship between a dependent
variable (the result) and one or more independent variables (the 'predictors',
or 'features').

For example: Who is more likely to subscribe to a website given the state they
live in, their age, marital status, and number of kids.

* Subscription is the dependent variable.
* State, age, marital status, and kids are independent variables.

Linear regression is finding the line of best fit. The goal is to make the "sum
of squared error" as low as possible. Gradient descent (consistently predicting
and adjusting to find the best estimate) can be used to create better results.

### Chapter 15: Multiple Regression

Using multiple variables in linear regression. The model gives us coefficients
for each variable - values representing how each variable impacts the result.

For example:

minutes spent online = 30.63 + 0.972 friends - 1.68 work hours + 0.911 phd

Means that on average, each extra friend corresponds to one more minute spent
online. Each hour spent working results in -1.68 minutes spent online, and
having a phd results in .911 more minutes online. The larger the coefficients
are for a term, the more impact it has on the result.

The variables are considered independent. If one variable impacts another (i.e.,
work hours impacts people with more friends more than with less friends), this
model won't tell us that.

You must pick features that are independent of each other.

### Chapter 16: Logistic Regression

Logistic regression is used to model certain classes of event - like "win/lose",
"pass/fail", or "dog/cat/lion/tiger".

Attempts to fit the data to a logarithmic function rather than a linear function.

#### Support Vector Machines (SVM)

Used for classification, an SVM is the line which best separates the data. The
line which *maximizes* the distance betweeen the two classifications of data.
New points are classified depending on which side of the line they fall.

### Chapter 17: Decision Trees

The goal is to build an optimal decision tree which predicts an outcome for a
set of inputs.

Note: it's easy to build a tree which overfits to the training data and doesn't
generalize well.

When building a tree, you want to pick steps you are confident in. An ideal step
would accurately predict the outcome for an input. i.e., if height > 6.4, they
are always a basketball player.

Each point is measured by entropy (uncertainty). High uncertainty = high
entropy. Generally, you want to partition data into buckets with large numbers
of values. Partitioning data based on SSN for example gives us no information.

ID3 Algorithm for building a decision tree:

* If all the data has the same label, create a leaf node that predicts that
  label, then stop.

* If the list of attributes is empty (no more questions to ask), create a leaf
  node that predicts the most common label then stop.

* Partition the data by each attribute. Create a decision node using the
  partition with the least entropy.

* Recurse on each partitioned subset.

#### Random Forests

Decision trees are closely tied to their training data and have the tendency to
overfit. A "random forest" consists of running muliple decision trees and take
the most common result.

Multiple trees can be built using multiple training sets.

To introduce randomness, rather than always choosing the "best" (lowest
entrophy), we choose a random attribute to split on.

Random Forests are examples of "ensemble learning", which combines several weak
models to produce a strong model (strenth in numbers).

### Chapter 18: Neural Networks

Neural networks are similar to brain networks. Each neuron takes an input and
either fires or doesn't fire.

Neurons are chained together:

input -> hidden neurons (many layers) -> output

#### Backpropogation

How do you decide which neurons to build and how many layers a network should
have? By training.

Similar to gradient descent, backpropogation finds the "best fit" neural network.

Backpropogation calculates the error result for a given set of inputs and
"propogates" the errors backward thru the hidden layers to adjust their weights.


### Chapter 19: Clustering

Clustering is an example of unsupervised learning. Unsupervised learning does
*not* use labeled training data when model building. Models are built just from
the data itself.

Clustering assumes that like elements will generally cluster together.

k-means is a simple custering methods which the number of clusters is chosen in
advance. Once k is chosen, the goal is to cluster the inputs into k clusters
which minimizes the total sum of squares distances from each point to the mean
of it's assigned cluster.

Algorithm:

1. Start with a set of k clusters at ramdom points.
2. Assign each data item to the point which it's closest.
3. Determine if any assignment has changed since the last iteration. If no, you are done.
4. Compute new cluster points based on previous results.
5. Repeat step 2.

How do you choose `k`?

Run the algorithm for k=1 -> k=n. Determine at what point the sum of squared
error begins to level out. You'll reach a `k` where the delta in squared error
starts to level out.

Bottom up clustering: starting with a cluster for each element, merge close
clusters until you have a single cluster. By keeping track of the merge order,
we will have a list of clusters for k=1 to k=n.

### Chapter 20: Natural Language Processing

NLP analyzes text and language (grammars).

n-grams : n words that appear together. A bi-gram (2) is a pair of words
together. A tri-gram is a set of three words together. Stringing together a set
of n-grams together can produce coherent documents. The larger n, the more
realisitc the documents become.

A grammar is a set of language rules.

Topic modeling: identifying the topic of a document given it's contents.

### Chapter 21: Network Analysis

Networks are graphs. Series of nodes and edges.

Edges can be undirected (bidirectional - like facebook friends) or directed (one
way - like html links).

#### Centrality

[Centrality](https://en.wikipedia.org/wiki/Centrality)

Degree Centrality: nodes who are highly connected.

Betweeness Centraility: nodes frequently on the shortest path.

How to find betweenness centrality? Write a function that finds the shortest
path from one node to all other nodes. Then find out how many of those shortest
paths pass thru the node.

Closeness Centrality: How close is a node to other nodes?

Eigenvector Centrality: The measure of a node's influence on the network
(PageRank). High eigenvector centrality means a node has a lot of connections to
other highly connected nodes.

Directed Graphs / PageRank

The number of links to a page isn't by itself very important. Links from "more
important" pages (highly linked to) are more important than links from less
important pages (less linked to).

### Chapter 22: Recommender Systems

What makes a great recommendation?

What's popular.

User-Based Collaborative Filtering: Find other nodes whose profile look like the
current node, making recommendations based off that node.

Item-Based Collaborative Filtering: Recommend based on the characteristics of
the node, not other nodes. For example, rather than looking for "other nodes
like the current node", base the recommmendation off of one of the node's
attributes. Think "user's who have bought an iPhone also buy...". The
recommendation is based on the iPhone, not the profile of the user who bought
it.

### Chapter 23: Databases and SQL

GROUP BY: Every column *NOT* in the group by needs to be an aggregate function.
HAVING: The filtering is applied to the aggregates.

### Chapter 24: MapReduce

* Break work into independent chunks (map).
* Execute each chunk in parallel.
* Reduce the chunks to produce a result (reduce).

MapReduce allows you to move processing closer to the data.

### Chapter 25: Go Forth and Data Science

* ipython: a better shell (with notebooks)
* numpy: scientific data structures
* pandas: `DataFrame` / data set manipulation
* scikit-learn: ML implementations
* d3.js (data driven documents): web visualization
