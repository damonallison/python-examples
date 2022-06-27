import pandas as pd
import pytest
from sklearn import datasets, linear_model, model_selection, preprocessing

# Mark all tests this module as 'ml'. These tests will be skipped with
# `make test` since they are slow.
pytestmark = pytest.mark.ml


def test_linear_regresssion() -> None:
    """A simple linear regression workflow.

    Feature Engineering
    -------------------
    There are a few feature engineering options we have to give linear models
    more power.

    * Binning data into discrete features (one hot encoding each bin into a
      feature) gives the model more power (not sure why / when to use binning)

    * Polynomial features allow for a smooth fit on data.

    * Non-linear features can make the data more gaussian and remove outliers.

    Note that kernel SVM or more complex models can learn s similar regression
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

    Feature Selection
    -----------------

    """
    X, y = datasets.make_regression(
        n_samples=10000,
        n_features=10,
        n_informative=8,
        noise=0.2,
    )

    # Add polynomial features
    # include polynomials up to x ** 10:
    # the default "include_bias=True" adds a feature that's constantly 1
    # This will add all combinations of (feature * feature) and feature^2 for a total of 65 features
    #
    # x_n == 10
    # x_n^1 == 10 (polynomial features)
    # x * x+1 == 9+8+7+6+5+4+3+2+1 == 45 (interaction features)
    # Total == 65
    #
    poly = preprocessing.PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(X)
    X_poly = poly.transform(X)

    assert X_poly.shape == (10000, 65)

    X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out())

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X_poly_df, y, test_size=0.2
    )

    # TODO(@damon): Examime features. Add sin / cos / log if the features are
    # skewed at all.

    # TODO(@damon): Automated feaature selection
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)

    r2 = lr.score(X_test, y_test)
    print(f"r2 == {r2}")
