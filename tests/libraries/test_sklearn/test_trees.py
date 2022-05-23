import logging

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, model_selection, tree

logger = logging.getLogger(__name__)

SEED = 42


# TODO:
# * feature_importances_ (tree analysis)
#


def run_dt(dt: tree.DecisionTreeClassifier) -> None:
    cancer = datasets.load_breast_cancer()

    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = cancer.target

    #
    # train_test_split Unit vs. integration testing.
    #   * Unit tests "train". Integration tests "test".
    #
    # Why not use all the data?
    #   * The goal isn't to understand the data you already *know*. It's to
    #     understand data you *don't* know.
    #
    # fit
    #
    # predict
    #
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, stratify=y, random_state=SEED
    )

    dt.fit(X_train, y_train)

    logger.info(f"Accuracy on training set: {dt.score(X_train, y_train):.3f}")
    logger.info(f"Accuracy on test set: {dt.score(X_test, y_test):.3f}")

    feature_imp = pd.DataFrame([dt.feature_importances_], columns=dt.feature_names_in_)
    logger.info(f"Tree feature_importances: {feature_imp}")


def test_default_decision_tree() -> None:
    run_dt(tree.DecisionTreeClassifier(random_state=SEED))


def test_pruned_tree_4() -> None:
    run_dt(tree.DecisionTreeClassifier(max_depth=4, random_state=SEED))


def test_pruned_tree_2j() -> None:
    run_dt(tree.DecisionTreeClassifier(max_depth=2, random_state=SEED))


def test_visualize_tree() -> None:
    cancer = datasets.load_breast_cancer()

    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = cancer.target

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, stratify=y, random_state=SEED
    )

    dt = tree.DecisionTreeClassifier(max_depth=4, random_state=SEED)
    dt.fit(X_train, y_train)

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
