"""
scikit-learn has a common workflow and API for all predictors.

* Data Preparation
    * Analysis
    * Cleaning
    * Feature Engineering
* Model training
    * "fit"
* Model evaluation
    * Cross validation
    * Hyperparameter tuning

These tests show that basic workflow.
."""

from typing import Any, cast

import joblib
import math
import os
import re

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets, linear_model, model_selection, preprocessing
import tempfile

# import matplotlib.pyplot as plt


def test_linear_regression_simple() -> None:
    """Creates and persists a simple linear model with 2 features.

    This simple model can be used for testing (i.e., testing model hosting)
    """

    def save_model(m: Any) -> str:
        """
        A utility function for persisting a model to a temp directory.
        """
        model_name = "model.joblib"
        tmpdir = tempfile.gettempdir()
        model_path = os.path.join(tmpdir, model_name)

        print(f"Saving model to: {model_path}")

        if os.path.exists(model_path):
            os.remove(model_path)
        joblib.dump(m, model_path)
        return model_path

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

    # pickle / unpickle using joblib
    model_path = save_model(lr)
    lr2: linear_model.LinearRegression = joblib.load(model_path)

    r22 = lr2.score(X_test, y_test)
    assert math.isclose(r2, r22, abs_tol=0.01)


def test_linear_regresssion_polynomial_features() -> None:
    """
    Shows adding polynominal features to a linear regression model.

    Often, input features interact in non-linear ways. Adding polynomial and
    interaction features (i.e., x1 * x2) features expose these interactions
    which may improve model performance, at the cost of model complexity and
    potential overfitting.
    """
    X, y = datasets.make_regression(
        n_samples=10000,
        n_features=10,
        n_informative=8,
        noise=0.2,
    )
    assert isinstance(X, np.ndarray)
    assert np.issubdtype(X.dtype, np.floating)
    assert np.issubdtype(X.dtype, np.float64)
    assert X.shape == (10000, 10)

    assert isinstance(y, np.ndarray)
    assert np.issubdtype(y.dtype, np.floating)
    assert np.issubdtype(y.dtype, np.float64)
    assert y.shape == (10000,)

    # Add polynomial features
    # include polynomials up to x ** 10:
    # the default "include_bias=True" adds a feature that's constantly 1
    # This will add all combinations of (feature * feature) and feature^2 for a total of 65 features
    #
    # x_n == 10
    # x_n^2 == 10 (polynomial features)
    # x * x+1 == 9+8+7+6+5+4+3+2+1 == 45 (interaction features)
    # Total == 65
    #
    poly = preprocessing.PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    assert X_poly.shape == (10000, 65)

    # Convert X_poly into a dataframe using the feature names created by
    # PolynomialFeatures
    X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out())
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X_poly_df, y, test_size=0.2
    )

    # View histograms of all features
    # X_poly_df.hist()
    # plt.show()

    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)

    assert lr.n_features_in_ == X_poly_df.shape[1]
    assert np.array_equal(lr.feature_names_in_, X_train.columns)

    r2 = lr.score(X_test, y_test)
    assert r2 > 0.0


def test_feature_order() -> None:
    """
    Features sent to predict must be in the same order as when the model was
    `fit`.

    Note that some ML frameworks (lgb) will replace spaces in feature names with
    `_`.
    """

    X, y = datasets.load_iris(return_X_y=True, as_frame=True)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    X = cast(pd.DataFrame, X)
    y = cast(pd.Series, y)

    feature_names = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    assert np.array_equal(X.columns, feature_names)
    assert y.name == "target"

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=0
    )

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)

    assert X_train.columns.to_list() == feature_names
    assert X_test.columns.to_list() == feature_names
    assert y_train.name == "target"
    assert y_test.name == "target"

    lr = linear_model.LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)

    assert isinstance(lr.feature_names_in_, np.ndarray)
    assert lr.feature_names_in_.tolist() == feature_names
    assert lr.n_features_in_ == len(feature_names)

    assert lr.score(X_test, y_test) > 0

    X_test_altered = X_test[
        [
            "petal length (cm)",
            "petal width (cm)",
            "sepal length (cm)",
            "sepal width (cm)",
        ]
    ]

    pattern = re.compile(r"feature names.*same order", re.DOTALL)
    with pytest.raises(ValueError, match=pattern):
        lr.score(X_test_altered, y_test)
