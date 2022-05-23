import logging

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from sklearn import datasets, ensemble, model_selection, tree

logger = logging.getLogger(__name__)

SEED = 42


#
# Demo 1:
#
# * Splitting, fitting, and scoring single decision trees.
# * Examining the decision tree using matplotlib
# * Graphing how "important" feature are with feature_importances_
#
def run_dt(dt: tree.DecisionTreeClassifier, plot_tree: bool = False) -> None:
    cancer = datasets.load_breast_cancer()

    # Creatd a DataFrame of our entire data set.
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y: np.ndarray = cancer.target

    # Examine the data set
    #
    # * Are all our columns (features) numeric?
    # * Text or categorical variables need to be converted prior to training.
    logger.info(X.describe())
    logger.info(f"\n{X.head()}")

    # Examine the target: is it numeric or categorical?
    #
    # If categorical, encode it.
    logger.info(y[0 : np.min([25, y.size])])

    #
    # train_test_split splits the data into two sets.
    #
    # Training set: Used to train the model
    # Test set: Used to "test" the model by determining how well it does on new data.
    #
    # This is somewhat similar to Unit vs. integration testing.
    #   * Unit tests "train" the code to work on known test cases.
    #   * Integration tests "test" the code by verifying it works in "real world" scenarios.
    #
    # Why not use all the data?
    #   * The goal isn't to understand the data you already *know*. It's to
    #     understand data you *don't* know.
    #
    # fit
    #
    # predict
    #
    logger.info("Splitting data")
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=SEED
    )
    logger.info(
        f"Split data {X.shape} into training: {X_train.shape} and test: {X_test.shape}"
    )

    logger.info("Training model")
    dt.fit(X_train, y_train)

    logger.info("Evaluating model")
    logger.info(f"Accuracy on training set: {dt.score(X_train, y_train):.3f}")
    logger.info(f"Accuracy on test set: {dt.score(X_test, y_test):.3f}")

    feature_imp = pd.DataFrame([dt.feature_importances_], columns=dt.feature_names_in_)
    logger.info(f"Tree feature_importances: {feature_imp}")

    if plot_tree:
        text = tree.export_text(
            dt,
            feature_names=list(cancer.feature_names),
            show_weights=True,
        )
        logger.info(text)

        # Plot tree
        plt.figure(figsize=(40, 20))
        tree.plot_tree(dt, feature_names=list(cancer.feature_names))
        plt.show()

        # Feature importances
        n_features = len(cancer.feature_names)
        plt.barh(np.arange(n_features), dt.feature_importances_, align="center")
        plt.yticks(np.arange(n_features), cancer.feature_names)
        plt.xlabel("feature importance")
        plt.ylabel("feature")
        plt.ylim(-1, n_features)
        plt.show()


def test_default_decision_tree() -> None:
    run_dt(tree.DecisionTreeClassifier(random_state=SEED), plot_tree=True)


#
# Demo 2: Hyperparameter tuning
#
def test_pruned_tree_4() -> None:
    run_dt(tree.DecisionTreeClassifier(max_depth=4, random_state=SEED))


def test_pruned_tree_2j() -> None:
    run_dt(tree.DecisionTreeClassifier(max_depth=2, random_state=SEED))


def test_tuned_decision_tree() -> None:
    cancer = datasets.load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y: np.ndarray = cancer.target
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, stratify=y, random_state=SEED
    )

    dt = tree.DecisionTreeClassifier(random_state=SEED)

    param_grid = {
        "max_depth": [1, 2, 3, 4, 5, None],
        "min_samples_split": [2, 3, 4],
    }
    grid_search = model_selection.GridSearchCV(
        dt,
        param_grid,
        cv=10,
        return_train_score=True,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info("Best cross-validation score: {:.3f}".format(grid_search.best_score_))
    logger.info(f"Accuracy on training set: {grid_search.score(X_train, y_train):.3f}")
    logger.info(f"Test score: {grid_search.score(X_test, y_test):.3f}")


#
# Demo 3: Ensembling: Random Forest
#


def test_random_forest() -> None:
    cancer = datasets.load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y: np.ndarray = cancer.target
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=SEED
    )

    rf = ensemble.RandomForestClassifier(random_state=SEED)

    rf.fit(X_train, y_train)

    logger.info(f"Accuracy on training set: {rf.score(X_train, y_train):.3f}")
    logger.info(f"Accuracy on test set: {rf.score(X_test, y_test):.3f}")

    feature_imp = pd.DataFrame([rf.feature_importances_], columns=rf.feature_names_in_)
    logger.info(f"Tree feature_importances: {feature_imp}")

    plot_tree = True
    if plot_tree:
        # Plot tree
        plt.figure(figsize=(20, 10))
        tree.plot_tree(rf.estimators_[0], feature_names=list(cancer.feature_names))

        plt.figure(figsize=(20, 10))
        tree.plot_tree(rf.estimators_[4], feature_names=list(cancer.feature_names))
        plt.show()

        # Feature importances are aggregared over all trees in the forest. In a
        # similar fashion to how a forest is typically stronger than a single
        # tree, feature_importances of the forest are more reliable than an
        # individual tree importances.
        n_features = len(cancer.feature_names)
        plt.barh(np.arange(n_features), rf.feature_importances_, align="center")
        plt.yticks(np.arange(n_features), cancer.feature_names)
        plt.xlabel("feature importance")
        plt.ylabel("feature")
        plt.ylim(-1, n_features)
        plt.show()


def test_tuned_random_forest() -> None:
    cancer = datasets.load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y: np.ndarray = cancer.target
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=SEED
    )

    rf = ensemble.RandomForestClassifier(n_estimators=5, random_state=SEED)

    # Note the combination of "forest" level parameters (n_estimators,
    # n_samples) and "tree" level parameters (max_depth, min_samples_split,
    # max_features).
    param_grid = {
        # n_estimators: larger is always better, however there are diminishing
        # returns.
        "n_estimators": [100, 500, 1000, 2000],
        #
        # Inject randomness into the "random forest"
        #
        # max_samples: the amount of data each tree is trained on
        # max_features: the number of features considered at each split
        "max_samples": [100, 200, None],
        #
        # A lower max_features == more random trees and reduces overfitting
        # Typically, use "sqrt for classification and n_features for regression.
        "max_features": [1, 2, len(X.columns)],
    }
    grid_search = model_selection.GridSearchCV(
        rf,
        param_grid,
        return_train_score=True,
        refit=True,
        n_jobs=-1,
        verbose=2,
    )
    grid_search.fit(X_train, y_train)
    logger.info("Best parameters: {}".format(grid_search.best_params_))
    logger.info("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    logger.info("Test score: {:.2f}".format(grid_search.score(X_test, y_test)))
