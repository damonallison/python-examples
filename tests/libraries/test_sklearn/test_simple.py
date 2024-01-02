"""Creates simple models."""

from typing import Any

import joblib
import os
import pandas as pd
from sklearn import datasets, linear_model, model_selection, preprocessing
import tempfile


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

    Note that kernel SVM or more complex models can learn a similar regression
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

    Feature selection is the process of selecting the most important features.
    The best features are those which predict the target the best.

    Always use domain experts if possible. They have intuition on the data to
    use.

    Practical advice:

        * Linear models: The higher the correlation between X(i) and Y, the better
          X(i) is.
        * Tree models: The higher the feature_importance, the better.
        * Use LASSO or Ridge. The higher the coefficients, the better.
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

    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)

    r2 = lr.score(X_test, y_test)

    print(f"n_reatures_in_: {lr.n_features_in_}")
    print(f"feature_names_in_: {lr.feature_names_in_}")

    print(f"r2 == {r2}")


def save_model(m: Any) -> str:
    model_name = "model.joblib"
    tmpdir = tempfile.gettempdir()
    model_path = os.path.join(tmpdir, model_name)
    print(tmpdir)

    if os.path.exists(model_path):
        os.remove(model_path)
    joblib.dump(m, model_path)
    return model_path


def test_linear_regression_simple() -> None:
    """Creates and persists a simple linear model with 2 features.

    This simple model can be used for testing (i.e., testing model hosting)
    """
    X, y = datasets.make_regression(
        n_samples=100000,
        n_features=2,
        n_informative=8,
        noise=0.2,
    )
    X = pd.DataFrame(X, columns=["x1", "x2"])
    y = pd.DataFrame(y)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
    )
    lr = linear_model.LinearRegression(n_jobs=-1)
    lr.fit(X_train, y_train)

    assert len(lr.feature_names_in_) == lr.n_features_in_ == 2

    # Evaluate
    r2 = lr.score(X_test, y_test)
    print(f"r2 == {r2}")

    model_path = save_model(lr)
    print(f"model saved to: {model_path}")
