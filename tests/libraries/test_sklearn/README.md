# scikit-learn

## Linear Models

### Feature Engineering

There are a few feature engineering options we have to give linear models
more power.

* Binning data into discrete features (one hot encoding each bin into a feature)
  gives the model more power (not sure why / when to use binning)

* Polynomial features allow for a smooth fit on data.

* Non-linear features can make the data more gaussian and remove outliers.

Note that kernel SVM or more complex models (NNs) can learn a similar regression
curve without having to tranform features.

Polynomial features can actually decrease performance of more complex models
like trees or RandomForestRegressor.

Non-linear tranformations like log, exp, sin. Linear models and neural
networks are tied to the scale and distribution of each feature. If there is
a non-linear feature, it becomes hard to model.

sin, cos, and log (non-linear) makes the distribution of data more gaussian
and reduces outliers. Not all features should hvae nonlinear transformations
applied. Try plotting the feature before / after the non-linear
transformation and see if it improves the distribution and reduces outliers.

The implications of binning, interaction features, and polynomials can work
well for simple models and potentially NNs, but you have to experiment.

Binning, interaction features, and polynomials can discover interactions
themselves and don't need data transformations most of the time.

### Feature Selection

Feature selection is the process of selecting the most important features.
The best features are those which predict the target the best.

Always use domain experts if possible. They have intuition on the data to
use.

Practical advice:

* Linear models: The higher the correlation between X(i) and Y, the better
    X(i) is.
* Tree models: The higher the feature_importance, the better.
* Use LASSO or Ridge. The higher the coefficients, the higher the importance.
