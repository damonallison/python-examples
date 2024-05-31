# Causal ML

> Association (correlation) does not equal causation.

Using predictive inference (ML) to perform causal inference.

## Questions

* Example experiments which used ML to find confounding variables?

* How do you find confounding variables?
  * Use ML to determine which variables impact price?


## Causal Inference

Causal inference: Determining whether an observed association reflects a
cause-effect relationship.

Correlation does not equal causation. For example, people who consume olive oil
may live longer, but consuming olive oil may not *cause* longer life. For
example, olive oil is expensive and those who are wealthy generally have better
medical care, which is the real cause of longer life.


## Statistical Inference

Statistical inference observes relationships (i.e., correlations) between
variables. It does not imply causation, causal inference is the process for
determining the underlying cause.


## Terminology

### Confounder

A counfounder is an alternate variable which influences a relationship. For
example, if we observe a correlation between eating ice cream and sunburn, we
don't believe eating ice cream causes sunburns. The confounding variable is the
weather (sunny).

In causal inference we want to "eliminate" every possible counfounder to get the
real causal effect of one variable on another.

In order to eliminate confounders, we need to *determine* confounders.


### Interventional (treatment group?)

What is the impact of an intervention on final outcomes?

### Counterfactual (control group?)

What is the impact if I acted differently? This necessitates retrospective
reasoning.


### Causal Hierahchy (Pearl, 2016)

#### Level 1: Association: P(y|x)

Association between two variables. What is the relationship between vitamin C
and health?

#### Level 2: Intervention P(y|do(x), z)

At this level, we intervene.

What if we take aspirin, will my headache be cured?

#### Level 3: Counterfactuals:

Reasoning about a different action. What if I had acted differently (or not at all)?


## Preface

* The goal of causal ML is to use ML to determine which variables are confounders.
* We are ultimately bound by the data we have. If the data does not capture all
  of the confounding effects, ML cannot help.
* We need to use *all* the data we have, along with ML, to find all confounders.
  Not just numeric data, but text, images, videos, and more.

* It's impossible to definitively define a causal effect, as it will inevitably
  rest on fundamentally untestable assumptions. Still, ML allows us to be as
  confident as possible given the data we have.

## Chapter 1: Predictive Inference with Linear Regression in Moderately High Dimensions

What is inference?

> Infer: to form an opinion or guess that something is true based on the
> information you have.

### Foundations of Linear Regression

The goal is to find the "line of best fit" for all features. The line of best
fit minimizes error between predictions and actuals.

Features may include "constructed features" like polynomial / interaction /
computed features. Constructed features are nonlinear and may capture nonlinear,
more complex, patterns. Constructed (non-linear) features generally provides
better predictions than linear only.

### Statistical Properties of Least Squares

* The number of samples (`n`) should be large when compared to the number of
  features (`p`).
* The more data, the better.


* Analysis of Variance
    * How much variance can be explained by each feature (explained variation)?
    * How much variance is unexplained (residual variation)?
    * `MSE` = total unexplained error
    * `R^2` = explained variation / unexplained variation. Higher is better.

* Partialling-Out
    * Separating out the explained and residual (unexplained) variances.

* Overfitting
    * More parameters generally increase overfitting.
    * OLS does not work when `p / n` is large (we overfit).

* Train / test split
    * Split into train / test allows us to verify the trained model on new data.
      and iterate coefficients in attempt to reduce error.
    * Prefer stratified splitting, especially w/ smaller data sets.

## Chapter 2: Causal Inference via Randomized Experiments

* RCT = Randomized Controlled Trial (treatment / non-treatment groups)
* ATE = Average Treatment Effect (causal effect)

RCTs are also called A/B tests.

Random assignment to treatment or control groups elimninates selection bias.
RCTs are used everywhere. You need to make sure your selection is truly random.
For example, dropping data with missing values may impact randomness (a certain
popluation is impacted more than another).

For example, maybe women don't typically answer a survey question. Removing
samples missing that answer makes the sample more male.


### Limitations to RCTs

* Externalities - the outcomes of one group should not be impacted by treating
  another group. For example, if a large group takes a COVID vaccine, the
  control group may show less COVID due to the fact the vaccine was effective
  and the control group was less exposed to COVID.


