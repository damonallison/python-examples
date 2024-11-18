from typing import Tuple

import numpy as np
import pandas as pd

from sklearn import (
    compose,
    datasets,
    ensemble,
    feature_selection,
    linear_model,
    preprocessing,
)
from sklearn.model_selection import train_test_split


def test_one_hot_encoding() -> None:
    """Using sklearn to one-hot encode columns.

    There are two ways to OHE features:

    * Pandas: pd.get_dummies
    * scikit-learn: preprocessing.OneHotEncoder

    The advantage to using sklearn is you can embed emcoding into your sklearn
    pipleine.
    """

    df = pd.DataFrame.from_dict(
        {
            "int_feature": [0, 1, 2, 1],
            "categorical_feature": ["yellow", "green", "yellow", "red"],
        }
    )

    #
    # OneHotEncoder will encode *all* colunns it's given.
    #
    # Setting sparse=False tells OneHotEncoder to return a numpy array, not a
    # sparse matrix.
    #
    ohe = preprocessing.OneHotEncoder(sparse_output=False)
    ohe.fit_transform(df)

    assert set(ohe.get_feature_names_out()) == set(
        [
            "int_feature_0",
            "int_feature_1",
            "int_feature_2",
            "categorical_feature_yellow",
            "categorical_feature_green",
            "categorical_feature_red",
        ]
    )

    #
    # ColumnTransformer allows you to apply different transformations to
    # different columns.
    #
    ct = compose.ColumnTransformer(
        [
            (
                "scaling",
                preprocessing.StandardScaler(),
                ["int_feature"],
            ),
            (
                "onehot",
                preprocessing.OneHotEncoder(sparse_output=False),
                ["categorical_feature"],
            ),
        ]
    )

    #
    # Unlike pandas, which returns a new DataFrame, sklearn will
    # return an np.ndarray.
    #
    vals: np.ndarray = ct.fit_transform(df)
    df = pd.DataFrame(vals, columns=ct.get_feature_names_out())
    assert set(
        [
            "scaling__int_feature",
            "onehot__categorical_feature_green",
            "onehot__categorical_feature_red",
            "onehot__categorical_feature_yellow",
        ]
    ) == set(df.columns)
    df.notna()


def test_outlier_detection() -> None:
    """
    Outliers are often considered values outside a certain multiple of the IQR.
    """

    f1 = np.linspace(0, 1, 9)
    # append an f1 outlier at the end
    f1 = np.append(f1, 5)

    f2 = np.linspace(1, 100, 9)
    # append an f3 outlier at the beginning
    f2 = np.append([-200], f2)

    df = pd.DataFrame({"f1": f1, "f2": f2})

    # quantile will return the given quantile value for each column
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    factor = 1.5

    lower_limit = q1 - (factor * iqr)
    upper_limit = q3 + (factor * iqr)

    # assume outlier detection is 1.5x or more outside IQR
    # returns a boolean mask if any column is outide the IQR multiplier range
    outliers = ((df < lower_limit) | (df > (upper_limit))).any(axis=1)
    assert outliers.sum() == 2

    outliers_df = df[outliers]
    assert np.array_equal(outliers_df.index.values, np.array([0, 9]))


def test_binning() -> None:
    """Binning allows you to treat a continuous set of data into multiple
    discrete features.

    For example: turning a numeric "num_in_family" into "family_small",
    "family_medium", "family_large" features.

    There are multiple ways to define bins. Uniform or quantiles are two
    examples.

    Uniform: bins are equidistant. Quantiles: bins are set based on quantiles
    (e.g., 10% of the data in each bin). Bins will be different sizes (smaller
    ranges when there is more data, larger when less data), but contain the same
    number of points.

    Why bin? Binning gives linear models more flexability. A linear model will
    give a different value for each bin, which ideally is closer to the data.
    When the data is not truly linear, binning allows the linear model to
    predict closer to actuals for each bin (ideally).

    When bin? Bin features when their data is not truly linear. Understand the
    error both before and after binning (build a linear model with just one
    feature) to see how binning lowers or raises the overall error.

    Do *NOT* bin data when working with trees. Trees choose their own bins
    (sometimes in combinations with other features). Binning actually decreases
    predictive power when working with trees.
    """

    X = np.random.default_rng(42).integers(0, 101, size=(100,)).reshape(-1, 1)

    # By default, KBinsDiscretizer will return a sparse array. "onehot-dense"
    # will transform into a dense array.
    binner = preprocessing.KBinsDiscretizer(
        n_bins=10, strategy="quantile", encode="onehot-dense"
    )
    binner.fit(X)

    X_binned = binner.transform(X)

    # Each feature is given an array of bins. Flatten them out to receive a list
    # of all bins for all features.
    cols: list = []
    for feature_bins in binner.bin_edges_:  # Each feature has a list of bins
        for idx in range(len(feature_bins) - 1):
            cols.append(f"col_{feature_bins[idx]}_{feature_bins[idx+1]}")

    df = pd.DataFrame(X_binned, columns=cols)

    # Each bin now has a column name which reflects its bin range.
    assert df.columns[0] == "col_4.0_12.9"


def test_interaction_features() -> None:
    """Interaction and polynomial features are features which are created based
    off the combination of multiple features or multiples of a single feature.

    Interaction features are created by combining features together. For
    example, assume you have `miles` and `duration` features. Combining the two,
    create a `miles_per_hour` feature which reduces the feature space without
    losing any modeling power.

    Polynominal features are x^^2 or x^^3. They provide curvature to the line of
    best fit.

    Why create interaction and/or polynominal features?

    Using polynomial features makes the line of best fit to better curve with
    the data. Note that polynomials of high degree behave in extreme ways on
    boundaries or regions with little data.

    Note that using a more complex model (SVM), we can learn a similarly complex
    prediction without transforming features.

    As always, test your model with and without interaction / polynomial
    features. Performance of tree based models (or any model for that matter)
    may DECREASE with interaction features.
    """

    X = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [2, 4, 6, 8],
        }
    )

    poly = preprocessing.PolynomialFeatures(degree=3, interaction_only=False)
    X_poly = poly.fit_transform(X)

    assert np.array_equal(poly.feature_names_in_, ["x", "y"])

    df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out())
    assert np.array_equal(
        df.columns,
        ["1", "x", "y", "x^2", "x y", "y^2", "x^3", "x^2 y", "x y^2", "y^3"],
    )
    # Verify one polynomial feature
    assert np.array_equal(df["x y"], [2, 8, 18, 32])


def test_nonlinear_transformations() -> None:
    """Nonlinear transformations.

    Tree based models only care about the ordering of the features. Linear
    models are tied to the scale and distribution of data. If there is a
    non-linear relation between the feature and the target, it's hard to model.

    Applying non-linear functions to features (log, exp) can make feature values
    Gaussian distributed.

    This helps with features like counts, which are never negative and tend to
    have long tails. `log`, for example, transforms data into a symmetrical
    shape and reduces outliers.

    Remember: Nonlinear transformations can help simple linear models (and NNs
    to some extent). Nonlinear transformations are irrelevant to tree based
    models.

    More complex models, like SVMs, kNN, and NNs *may* benefit from binning,
    interactions, and polynomials, but the implications there are usually much
    less than in the case of simplier linear models.
    """

    rnd = np.random.RandomState(0)
    X_org = rnd.normal(size=(1000, 3))
    w = rnd.normal(size=3)

    X = rnd.poisson(10 * np.exp(X_org))
    y = np.dot(X_org, w)

    ridge = linear_model.Ridge()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    score = ridge.fit(X_train, y_train).score(X_test, y_test)
    print(f"Score: {score}")

    # Take the log of each feature, giving features a more normal distribution.
    # Use +1 to ensure no feature is 0

    X_train_log = np.log(X_train + 1)
    X_test_log = np.log(X_test + 1)

    score = ridge.fit(X_train_log, y_train).score(X_test_log, y_test)
    print(f"Score w/ log: {score}")


def breast_cancer_with_noise(noise_features: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    cancer = datasets.load_breast_cancer()

    # get deterministic random numbers
    rng = np.random.RandomState(42)
    noise = rng.normal(size=(len(cancer.data), noise_features))
    # add noise features to the data
    # the first 30 features are from the dataset, the next 50 are noise
    return (np.hstack([cancer.data, noise]), cancer.target)


def test_univariate_statistical_feature_selection() -> None:
    """Univariate statistics.

    Univariate (one variable) statistical selection compares each feature
    individually to the target, only keeping the highest correlated features.
    With classification, this is known as ANOVA (analysis of variance).
    """

    from sklearn.feature_selection import SelectPercentile

    X, y = breast_cancer_with_noise(50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.5
    )
    # use f_classif (the default) and SelectPercentile to select 50% of features
    select = SelectPercentile(percentile=50)
    select.fit(X_train, y_train)
    # transform training set
    X_train_selected = select.transform(X_train)

    print("X_train.shape: {}".format(X_train.shape))
    print("X_train_selected.shape: {}".format(X_train_selected.shape))

    print("Feature selection (notice how the non-noise features are selected):")
    print(select.get_support())


def test_model_based_feature_selection() -> None:
    """Model based feature selection.

    Using a model to select features. Features with the highest performance
    (feature_importances_ for trees or coefficients for linear models) are kept.

    Linear models with an L1 penalty use only a subset of features.

    Using a model is much more powerful than univariate feature selection since
    it takes into account feature interactions.
    """
    X, y = breast_cancer_with_noise(50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.5
    )

    # Using median, only 1/2 of the features will be selected. We could use
    # different thresholds (count, percentage).
    #
    # Using a complex model like this to select features is much more powerful than
    select = feature_selection.SelectFromModel(
        ensemble.RandomForestClassifier(n_estimators=100, random_state=0),
        threshold="median",
    )

    select.fit(X_train, y_train)
    X_train_l1 = select.transform(X_train)
    print(f"X_train.shape: {X_train.shape}")
    print(f"X_train_l1.shape: {X_train_l1.shape}")

    # Look at the mask: notice the model does a much better job of selecting
    # non-noise features than univariate analysis did.

    print(select.get_support())


def test_recursive_feature_elimination() -> None:
    """Recusrive Feature Elimination (RFE).

    RFE will build a model, examine feature importance, disregard the least
    important features, and continue iterating until the pre-specified number of
    features remain.

    While this is expensive, it's
    """
    X, y = breast_cancer_with_noise(50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.5
    )

    select = feature_selection.RFE(
        ensemble.RandomForestClassifier(n_estimators=100, random_state=42),
        n_features_to_select=40,
    )

    select.fit(X_train, y_train)
    # visualize the selected features:
    print(select.get_support())
