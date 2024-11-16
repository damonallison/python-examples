"""Model evaluation includes the following:

* Determining how a model performs on unseen (test) data.
* Hyperparameter tuning to build an optimal model (GridSearchCV).


Model Evaluation
----------------
Simple model evaluation is done with the `score` method (classification) or
accuracy metrics (MAE, RMSE for regression). `score` is different for every
model type. The metrics used for scoring vary based on the model type.

Classification scoring:

* Accuracy
* Precision / Recall / F1
* ROC-AUC
* Confusion matrix

Regression scoring:

* MAE / MSE / RMSE
* R^2


For classification models, `accuracy` is returned. For regression models, R^2 is
returned which represents the proportion of variance in the dependent variable
that is predicted from the independent variables (features).

R^2 = 1 - (sum of squared residuals / total sum of squares)

* Sum of squared residuals = sum of squared differences between the actual and
  predicted values.

* Total sum of squares: Sum of squared differences between the actual values and
  the mean of the dependent variable.


R^2 provides a measure of how well the model captures variability in the data.
It does *not* indicate whether the model is appropriate for making predictions
on new data. Therefore, it is often used in conjunction with other methods (like
RMSE).

Regression model metrics:

* Mean Absolute Error (MAE): The average absolute differences between predicted
  and actual values.

* Mean Squared Error (MSE): Average of the squared differences between predicted
  and actual values.

* Root Mean Squared Error (RMSE): The root of MSE. RMSE is interpretable since
  it shares the same unit as the dependent variable.

Note that RMSE is sensitive to outliers. Typically both RMSE and R^2 are used
together to score a regression model.

Scaling
-------
Models that are distance based (kNN, linear regression, logistic regression,
NNs) are subject to scaling imbalances. **Scale data before model training**.

Cross Validation
----------------
Cross validation is a process of training and validating multiple models on
different slices of data. Cross validation is more robust since it ensures the
entire data set is used as the test set in one fold. There are multiple CV
strategies available, however KFold, StratifiedKFold, and GroupKFold are the
most common.

Learning Curves
---------------
Plots that show the model's performance on both training and validation datasets
ofer different training sizes or epochs.

---

Grid Search (Hyperparameter Tuning)
-----------------------------------
Grid search is the process of finding the best model hyperparameters by
"searching" thru a grid of possible hyperparameter combinations.

When performing grid search, we split the training set into "training" and
"validation" sets. The validation set is used to determine how well the model
*should* perform on new data. We select the best model based on it's performance
on the validation set. We then test the model with the test set.

We do *not* want to include the test set in any grid search validation.
Otherwise, we "leak" data - the model's hyperparameters will be learned in part
on the test set, which we need to avoid. This is why we split the training data
into test / validation sets.

The grid search process:
------------------------
1. Define a grid of hyperparameter values to test.
2. Split the training set into training / validation sets.
3. Train and validate a model w/ each hyperparameter combination.
4. Save the hyperparameter combination which scored the best on the validation
   set using cross validation.
5. Retrain the model with the saved hyperparameters and full training set.
6. Evaluate the model on the test set.

Important: grid search is an expensive process. For each hyperparameter
combination, we generate `k` models to perform cross validation. Assuming you
have a 3*4 grid of parameters with ``5 fold CV, 12 * 5 == 60 models will be
created and evaluated.

The goal of grid search is to find the optimal set of hyperparameters. Every
grid will have an "optimal" set of parameters chosen from the values in the
grid. But how do you know if you have the correct values in your grid? How do
you know better values don't exist?

Examining accuracy results either textually or in a heatmap will show you how
each combination scored. There are a few indicators that will tell you if you've
chosen the correct ranges of hyperparameters.

* Ideally, the highest accuracy model will fall in the middle of the
  hyperparameter grid for each parameter. That reflects each parameter's range
  has been set correctly. Both ends of the parameter's range have been
  evaluated.

* If the best scores are on the edges (w/ maximum or minimum values for 1 or
  more parameters), your search space is not wide enough for those parameters.
  The model may get more accurate if you increase or decrease those parameter
  values.

* If accuracy is not changing between the min / max values of a parameter,
  either the parameter is not important to the results or you haven't chosen a
  wide enough search space for that parameter. Try increasing the range of the
  parameter. If the increased range still yields consistent accuracy, that
  parameter is not important to tune (or perhaps could be eliminated).


---


Model metrics (scoring)
-----------------------

Binary classification
---------------------------------
Binary classification is classifying data into two groups. There are 4 ways an
individual result could turn out:

* True positive: correctly predicted positive.
* False positive: incorrect positive prediction. Type 1 error.
* False negative: incorrect false prediction. Type 2 error.
* True negative: correctly predicted negative.

When evaluating a binary classifier, associate a cost to each error type. For
example, when predicting cancer, the cost of a false negative (type 2) is *much*
higher than a false positive (type 1). If you predict a false positive, the
patient would need to have more tests performed which would ultimately prove no
cancer. If you predict a false negative, the patient truly has cancer and
doctors will think they are fine.

Watch for imbalanced data sets. If 99% of emails are spam, simply predicting
every email will be spam will be 99% accurate. With imbalanced data sets,
accuracy is an inadequate measure for quantifying predictive performance.

We need alternative metrics to determine which models perform well.

Confusion Matrix
----------------
A confusion matrix gives you the true / false positive / negative counts.


Precision-Recall curve
----------------------

Binary classification models have a precision / recall tradeoff. Precision is
how accurate your positives are. Recall is how accurate you are at predicting
positives (Recall == True Positive Rate).

* Accuracy = (TP + TN) / (TP + TN + FP + FN)
* Precision = TP / (TP + FP)
* Recall = TP / (FP + FN)

There is a tradeoff in precision vs. recall. You can have 100% precision by only
predicting the sample you are most confident in as positive. However, you'll
miss a lot of positives (high false negatives) and your recall will suffer. If
you lower the confidence interval, you'll have high false positives (low
precision).

The goal is to determine how agressive you want to be when predicting positives.
The optimal threshold is found by testing all thresholds and finding the best
one.

When evaluting a model, the more accurate overall model will have the highest
area under the "receiver operator curve". The "receiver operator curve" is the
graph of true positive rate over false positive rate. The higher the AUC, the
higher the ability for a model to classify correctly across all thresholds.

Define your goals
-----------------

Often, the goal is not to have the most accurate overall model. You may want to
have high precision (confident in your positives) or high recall (low false
negatives). If you are diagnosing people with cancer for example, you want high
recall (low false negatives). Predicting someone as *not* having cancer when
they indeed do is MUCH more expensive than predicting someone as having cancer
who doesn't.
"""

import math
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from sklearn import datasets, ensemble, linear_model, metrics, model_selection, svm

# Many tests will have options to show plots. Setting DEBUG to `True` will
# display the plots.
DEBUG = False
SEED = 42


def test_available_scorers() -> None:
    """
    Retrieve the list of scorers. These names can be passed to get_scorer to
    retrieve a scorer.
    """

    # A "Predict" scorer will call "predict" on an estimator, so an estimator is required
    X, y = datasets.load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=0
    )
    lr = linear_model.LogisticRegression(random_state=SEED)
    lr.fit(X_train, y_train)

    assert "accuracy" in metrics.get_scorer_names()
    scorer = metrics.get_scorer("accuracy")

    assert scorer(lr, X_test, y_test) > 0.0

    # Scoring functions are also available on the metrics module
    assert metrics.accuracy_score([1, 1, 1, 1], [0, 0, 1, 1]) == 0.5


def test_score() -> None:
    """
    For classification problems, "accuracy" is used for scoring.
    """

    # make_blobs makes gaussian clusters
    X, y = datasets.make_blobs(n_samples=1000, n_features=2, random_state=0)

    X = cast(npt.NDArray[np.floating], X)
    y = cast(npt.NDArray[np.floating], y)

    # equally distributed labels between 0-3
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))
    colors = {
        0: "red",
        1: "green",
        2: "blue",
    }
    if DEBUG:
        for u in unique:
            only_u = X[y == u]
            plt.scatter(only_u[:, 0], only_u[:, 1], c=colors[u], label=u)
        plt.legend()
        plt.show()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=0
    )

    lr = linear_model.LogisticRegression().fit(X_train, y_train)

    # Manually score for accuracy
    y_pred = lr.predict(X_test)
    acc_score = metrics.accuracy_score(y_test, y_pred)

    # Accuracy score using the predictor
    lr_score = lr.score(X_test, y_test)

    assert math.isclose(acc_score, lr_score, abs_tol=0.001)


def test_cross_validation() -> None:
    """
    Cross validation is a more robust way to assess generalization performance.

    Cross validation splits data into multiple "folds", building a model on each
    fold. Scores are determined on each fold. The aggregate score (mean, for
    example) is determined for an overall model "score".

    The goal of CV is to more accurately determine model performance by using
    multiple variations of the test dataset. This provides a "strength in
    numbers" approach

    Benefits of CV:

    1. All data is used in a test set.

    Cross validation is more robust since multiple train / test splits are
    scored. Also, all data is used in a test set in one of the folds. This gives
    us more confidence in the overall model generalization performance (score\).

    By default, train_test_split will randomly create the train / test split. If
    we are "unlucky" and receive all hard to score examples in our test set, the
    model score will be artificially low. CV uses *all* data as test data at one
    point, giving us a better idea of our model's true score.

    2. CV helps determine model sensitivity.

    When we see accuraces between, say, 80% and 90%, we can be reasonably
    confident our overall model performance will be somewhere in the 80%s.

    A wider variance gives us less confidence in overall model performance.

    3. We use our data more effectively.

    The more folds we use, the more data we can include in the training set,
    which generally results in better model performance.

    --

    CV is expensive since you are building a model for each fold. However, the
    confidence you get from using CV is typically worth the extra compute cost.
    """

    X, y = datasets.load_iris(return_X_y=True)
    lr = linear_model.LogisticRegression(max_iter=1000, random_state=0)

    kf = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    # At least 5-fold CV is recommended
    #
    # Note that for classification tasks (like LogisticRegression), sckit-learn
    # uses stratified k-fold, which ensures the test set contains the same
    # overall proportion of each label as the entire data set. This ensures our
    # test set is a more accurate representation of the whole.
    #
    # NOTE: Always use stratified k-fold CV for classification tasks.
    #
    scores = model_selection.cross_val_score(lr, X, y, cv=kf)

    # cross_validate is similar to cross_val_score, but returns a dictionary
    # containing additional data:
    #
    # * fit_time
    # * score_time
    # * test_score
    # * train_score

    cv_results = model_selection.cross_validate(
        lr,
        X,
        y,
        cv=kf,
        verbose=0,
        return_estimator=False,
        return_train_score=True,
    )

    df = pd.DataFrame(cv_results)
    assert math.isclose(np.mean(scores), df["test_score"].mean(), abs_tol=0.01)


def test_cross_validation_custom_splitter() -> None:
    """
    A custom splitter (KFold) can be used to control the k-fold splitting
    process.

    Here, we override scikit-learn's default of using stratified k-fold for
    classification tasks. Instead, we manually split the data into three
    sequential folds.

    Since the data set is ordered by label (all 0's first, 1's, second, etc),
    and we have 3 classes, model performance will be 0 for 3 fold CV.
    """

    X, y = datasets.load_iris(return_X_y=True)
    lr = linear_model.LogisticRegression(max_iter=1000, random_state=0)

    kfold = model_selection.KFold(n_splits=3)
    scores = model_selection.cross_val_score(lr, X, y, cv=kfold)
    assert scores.mean() == 0.0

    # Using stratified k-fold ensures each fold has proportionally the same
    # number of each label as the population. We also shuffle the data for a bit
    # more randomness.
    #
    # Whenever we shuffle, we need to set a random_state if we want reproducable
    # results.
    #
    # You'll typically use StratifiedKFold for classification.
    kfold = model_selection.StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    scores = model_selection.cross_val_score(lr, X, y, cv=kfold)
    assert scores.mean() > 0.90


def test_leave_one_out_cv() -> None:
    """
    Leave one out CV will create a fold with a single sample as the entire test
    set.

    This is really expensive (n folds == n models) but could produce better
    results when you have smaller data sets.
    """
    X, y = datasets.load_iris(return_X_y=True)

    lr = linear_model.LogisticRegression(max_iter=1000, random_state=0)
    loo = model_selection.LeaveOneOut()
    scores = model_selection.cross_val_score(lr, X, y, cv=loo)

    assert len(scores) == len(X)
    assert scores.mean() > 0.90


def test_shuffle_split() -> None:
    """
    Shuffle split will split a configured number of samples for the training and
    test sets.

    Shuffle split allows you to control the number of iterations (folds)
    independently of the data. For exanple, you could use a training size of 50%
    and a test size of 20% (leaving 30% of the data out).

    This can be used with large data sets. This can be useful for experimenting
    with large data sets (or running tests - anywhere you could use a smaller
    data set).
    """

    X, y = datasets.load_iris(return_X_y=True)

    lr = linear_model.LogisticRegression(random_state=SEED)
    shuffle = model_selection.StratifiedShuffleSplit(
        n_splits=10, train_size=0.5, test_size=0.1
    )
    scores = model_selection.cross_val_score(lr, X, y, cv=shuffle)
    assert scores.mean() > 0.90


def test_cv_with_groups() -> None:
    """
    Grouping ensures that data belonging to the same group (say faces for facial
    recognition), are not split between the training and test sets.

    Examples of groups:

    * Patients in medical applications. You want to ensure you generalize on new
      patients.

    * Speakers in a speech recognition model. You want to ensure you generalize
      on new speakers.
    """

    X, y = datasets.make_blobs(n_samples=12, random_state=0)
    lr = linear_model.LinearRegression()
    # Assume the first 3 samples belong to the same group, then the next 4, etc.
    groups = ["Y", "Y", "Y", "G", "G", "G", "G", "B", "B", "B", "R", "R"]
    groupCV = model_selection.GroupKFold(n_splits=3)
    scores = model_selection.cross_val_score(lr, X, y, groups=groups, cv=groupCV)

    assert len(scores) == 3


def test_grid_search() -> None:
    """
    Grid search is used for hyperparameter tuning. Each combination of
    parameters in the grid are evaluated and the best performing parameter
    combination is kept as the "tuned hyperparameters" to use for final model
    generation."""
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=0
    )

    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "gamma": [0.001, 0.01, 0.1, 1, 10, 100],
    }
    # Since SVC is a classifier, stratified CV is used
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = model_selection.GridSearchCV(
        svm.SVC(), param_grid=param_grid, cv=kf, return_train_score=True
    )

    #
    # GridSearchCV will find the best hyperparameters and fit / return a new
    # model w/ them, which can be found with `best_estimator_`. Calling `score`
    # or `predict` on the grid search object will use the best estimator.
    #
    gs.fit(X_train, y_train)

    assert all((param in gs.best_params_.keys() for param in param_grid.keys()))

    #
    # All CV results are stored in cv_results_. Each record in cv_results_
    # represents a different parameter combination.
    #
    cv_results = pd.DataFrame(gs.cv_results_)
    # 6 x 6 - a row for each parameter combination
    assert len(cv_results) == 36

    # CV is performed on the grid using depth first parameter combinations. For
    # example, C=0.001 is used with all combinations of gamma first.
    #
    # C = 0.001, gamma=0.001
    # C = 0.001, gamma=0.01
    # C = 0.001, gamma=0.1
    # ...
    #
    # Verify the order
    param_C = scores = np.array(cv_results["param_C"]).reshape(6, 6)
    for i, row in enumerate(param_C):
        assert np.all(np.equal(x, param_grid["C"][i]) for x in row)

    print(cv_results.head())

    #
    # Plot the results in a heat map.
    #
    # This is an important step to determine if you're grid is setup correctly.
    # It shows you if you have selected the right parameters to tune and have
    # set appropriate ranges for each parameter.
    #
    # You want the highest test score to be somewhat in the middle of the
    # heatmap. This ensures you have found the best parameters for all parameters.
    #
    if DEBUG:
        scores = np.array(cv_results["mean_test_score"]).reshape(6, 6)
        sns.heatmap(
            scores,
            xticklabels=param_grid["gamma"],
            yticklabels=param_grid["C"],
            annot=True,
            fmt=".2f",
        )
        plt.show()
    #
    # The only place in the grid search process the test set is used is during
    # final model scoring. The test set is *NOT* used to find the best model.
    #
    assert gs.best_score_ > 0.0
    print(f"gs score: {gs.score(X_test, y_test)}")


def test_grid_search_multiple_grids() -> None:
    """
    In some cases, not all parameters will be valid when used together.

    For example, when using SVC with a kernel parameter, when kernel='linear',
    only C is used. When kernel='rbf', C and gamma are used.

    GridSearchCV allows you to pass a list of grids.
    """

    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=0
    )

    param_grid = [
        {"kernel": ["rbf"], "C": [0.01, 0.1], "gamma": [0.01, 0.1]},
        {"kernel": ["linear"], "C": [0.01, 0.1]},
    ]
    grid_search = model_selection.GridSearchCV(
        svm.SVC(), param_grid, cv=5, return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    print("Best parameters: {}".format(grid_search.best_params_))
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    print("Test score: {:.2f}".format(grid_search.score(X_test, y_test)))


def test_nested_cross_validation() -> None:
    """
    Nested cross validation is a more advanced form of CV by testing all
    combinations of hyperparameters on multiple folds.

    First, For each parameter combination, GridSearchCV will use cross
    validation to determine the mean test score by splitting the training set
    into training / validation and building / testing a model for each fold.

    Second, CV is used to vary the training / test split.

    In the examples above, we are only splitting the data once into train / test
    sets, then running GridSearchCV once using the single train / test split.
    This makes model performance dependent on a single split of the data. A more
    robust (but expensive) solution would be to run the entire GridSearchCV
    using multiple folds..

    In nested cross-validation, there is an outer loop over splits of the data
    into training and test sets. For each of them, a grid search is run (which
    might result in different best parameters for each split in the outer loop).
    Then, for each outer split, the test set score using the best settings is
    reported.

    Nested CV does *not* return a model, it returns a list of scores. The scores
    tells us how well a model generalizes. Nested CV is used to determine how
    well a given model works on a dataset.

    Note that nested CV is expensive. Here, we have 4 * 2 = 8 parameter
    combinations. We use 5-fold CV within GridSearchCV, resulting in 8 * 5 = 40
    models being trained. We use 5-fold CV to validate the entire dataset, which
    results in a total of 30 * 5 == 150 models.
    """

    X, y = datasets.load_iris(return_X_y=True)

    param_grid = {
        "C": [0.0001, 0.001, 0.01, 0.1],
        "gamma": [0.01, 0.1],
    }

    # the "inner loop" - which will evaluate the param grid for an individual fold
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid_search = model_selection.GridSearchCV(svm.SVC(), param_grid, cv=kf)
    grid_search.fit(X, y)

    # Here, we ran 5-fold CV for each parameter combination. A total of 40
    # models being trained and scored.
    assert len(grid_search.cv_results_["mean_test_score"]) == 8

    # the "outer loop" - which breaks the outer data set into folds
    scores = model_selection.cross_val_score(grid_search, X, y, cv=kf)

    # Each iteration of the outer loop results is a score for one execution of
    # GridSearchCV. With each iteration of GridSearchCV building 40 models, we
    # have a total of 40 * 5 == 200 models being trained.
    assert len(scores) == 5
    #
    # The result of nested CV can be summarized as "SVC can achieve a 97% mean
    # CV accuracy on the iris dataset". It will *not* produce a model.
    #
    assert scores.mean() > 0.90


#
# Binary Classifier Metrics
#


def test_confusion_matrix() -> None:
    """A confusion matrix is one of the best ways to evaluate a binary classifier.

    [
        [
            TN, FP
        ],
        [
            FN, TP
        ]
    ]

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (FP + FN)

    F1 score = 2 * (precision * recall) / precision + recall

    Precision: How accurate is the model's positive predictions?
    Look for high precision when you want to avoid false positives.

    Recall: How accurate is the model at predicting positives?
    Look for high recall when you want to avoid false negatives.

    There is a tradeoff between precision and recall.

    You can easily obtain a perfect recall by predicting positive for
    everything, however precision will be low.

    You can easily obtain a perfect precision by predicting a single positive
    prediction correctly, however recalll will be low.

    F1 considers both precision and recall for an overall better measure of
    model performance than accuracy, precision, and recall. By combining both
    precision and recall, F1 is the most comprehensive and best way to evaluate
    a binary classifier.
    """

    digits = datasets.load_digits()  # 8x8 image of a digit
    y = digits.target == 9

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        digits.data, y, random_state=0
    )

    logreg: linear_model.LogisticRegression = linear_model.LogisticRegression(
        C=0.1, max_iter=10000
    )
    logreg.fit(X_train, y_train)
    logreg_pred = logreg.predict(X_test)

    # Accuracy
    logreg.score(X_test, y_test)

    tn, fp, fn, tp = metrics.confusion_matrix(y_test, logreg_pred).ravel()

    accuracy = (tp + tn) / (tn + fp + fn + tp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    assert math.isclose(accuracy, metrics.accuracy_score(y_test, logreg_pred))
    assert math.isclose(precision, metrics.precision_score(y_test, logreg_pred))
    assert math.isclose(recall, metrics.recall_score(y_test, logreg_pred))
    assert math.isclose(f1, metrics.f1_score(y_test, logreg_pred))

    #
    # sklearn has a convenience function `classification_report` to calculate
    # and print all confusion matrix based mertrics.
    #
    # * "support" is the number of samples in each class.
    # * "accuracy"
    # * "macro avg" is the simple average across all classes
    # * "weighted avg" is the weighted average across all classes
    #
    print(
        metrics.classification_report(
            y_true=y_test, y_pred=logreg_pred, target_names=["not nine", "nine"]
        )
    )


def test_decision_functions() -> None:
    """Updating the decision_function will determine how aggressive the model is
    when predicting classes. Lowering the decision function threshold will
    increase the number of samples in the positive class (you're letting more
    examples in), which increases your true positive rate (recall), but allows
    more false positives as well, which decreases precision.
    """

    X, y = datasets.make_blobs(
        n_samples=(400, 50), cluster_std=[7.0, 2], random_state=22
    )
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=0
    )
    svc: svm.SVC = svm.SVC(gamma=0.05).fit(X_train, y_train)

    rpt1_dict = metrics.classification_report(
        y_test, svc.predict(X_test), output_dict=True
    )

    # By lowering the threshold from 0.0 (the default) to -0.8, we are allowing
    # more samples to be classified as positive. This increases recall, lowers
    # precision.

    y_pred_lower_threshold = svc.decision_function(X_test) > -0.8

    rpt2_dict = metrics.classification_report(
        y_test, y_pred_lower_threshold, output_dict=True
    )

    #
    # Verify recall of the positive class increases, precision decreases
    #
    assert rpt1_dict["1"]["recall"] < rpt2_dict["1"]["recall"]
    assert rpt1_dict["1"]["precision"] > rpt2_dict["1"]["precision"]


def test_precision_recall_curve() -> None:
    """By adjusting the decision threshold down, you increase recall. If you set
    the threshold low enough, you can always achieve 100% recall. However, the
    overall model will be useless.

    How do you set the optimum threshold, or "operating point"?

    You can determine the optimum threshold by testing all possible combinations
    of decision thresholds and plotting it's precision / recall.

    Note that different classifiers perform better in certain areas. Here we see
    that RF performs better at both ends of the spectrum, for very high recall
    or precision. SVC is better in the middle.

    The F1 score only captures one point on the precision / recall curve: the
    one given by the default threshold.

    Looking at the precision-recall curve doesn't quantifiably tell us which
    model performs better. We need to quantifiably measure the performance of
    each model to determine which is better.

    Average precision is a good summarization of the precision-recall curve.
    Average precision is the integral (or area) under the precision-recall
    curve. The model with the higher average precision is "better" than another
    model on average. However another model may perform better than another in
    areas of the curve.
    """
    X, y = datasets.make_blobs(
        n_samples=(4000, 500), cluster_std=[7.0, 2], random_state=22
    )
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=0
    )

    #
    # Support Vector Classifier
    #
    svc = svm.SVC(gamma=0.05).fit(X_train, y_train)
    precision, recall, thresholds = metrics.precision_recall_curve(
        y_test, svc.decision_function(X_test)
    )

    #
    # Find threshold closest to zero (the default)
    #
    close_zero = np.argmin(np.abs(thresholds))
    plt.plot(
        precision[close_zero],
        recall[close_zero],
        "o",
        markersize=10,
        label="threshold zero",
        fillstyle="none",
        c="k",
        mew=2,
    )
    plt.plot(precision, recall, label="svc")

    #
    # Random Forest Classifier
    #
    rf = ensemble.RandomForestClassifier(
        n_estimators=100, random_state=0, max_features=2
    )
    rf.fit(X_train, y_train)

    # RandomForestClassifier has predict_proba, but not decision_function
    precision_rf, recall_rf, thresholds_rf = metrics.precision_recall_curve(
        y_test, rf.predict_proba(X_test)[:, 1]
    )

    #
    # Thresholds must be between 0 and 1. Here we take close to the default
    # threshold.
    #
    close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
    plt.plot(
        precision_rf[close_default_rf],
        recall_rf[close_default_rf],
        "^",
        c="k",
        markersize=10,
        label="threshold 0.5 rf",
        fillstyle="none",
        mew=2,
    )

    plt.plot(precision_rf, recall_rf, label="rf")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.legend(loc="best")

    if DEBUG:
        plt.show()

    #
    # Determine the "area under curve" for each model's precision-recall curve.
    #
    # Because we need to test at different thresholds, you need to send in
    # predict_proba or decision_function, not the results of predict (i.e.,
    # binary predictions)
    #
    rf_avg = metrics.average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])
    svc_avg = metrics.average_precision_score(y_test, svc.decision_function(X_test))

    print(f"Average precision for rf: {rf_avg} svc: {svc_avg}")


def test_roc_auc() -> None:
    """Receiver Operator Characteristics (ROC) is another tool like the
    precision-recall curve that analyzes classifier behavior at different
    thresholds and uses an area based summary statistic to determine how the
    models compare. Models with a higher area under curve (AUC) are, on average,
    more accurate. (Although some models may be better at different areas).

    ROC shows the false positive rate (FPR) compared to the true positive rate
    (TPR). TPR is another name for recall, while FPR is the false positive rate
    out of all negative examples.

    * TPR = TP / (TP + FN)
    * FPR = FP / (FP + TN)

    The optimal threshold is the point to the top left of the ROC curve. It
    produces the highest recall (TPR) with the lowest number of false positives.

    The ROC curve is typically summarized by computing the "area under curve" or
    AUC. Predicting randomly will produce a ROC of 0.5.

    AUC is much better metric than accuracy for imbalanced classification
    problems.
    """

    X, y = datasets.make_blobs(
        n_samples=(4000, 500), cluster_std=[7.0, 2], random_state=22
    )
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=0
    )

    svc = svm.SVC(gamma=0.05).fit(X_train, y_train)

    #
    # The ROC curve evaluates the model using different thresholds (like
    # precision-recall)
    #
    fpr, tpr, thresholds = metrics.roc_curve(y_test, svc.decision_function(X_test))
    plt.plot(fpr, tpr, label="SVC ROC")

    # find threshold closest to zero
    close_zero = np.argmin(np.abs(thresholds))
    plt.plot(
        fpr[close_zero],
        tpr[close_zero],
        "o",
        markersize=10,
        label="svc threshold 0",
        fillstyle="none",
        c="k",
        mew=2,
    )

    rf = ensemble.RandomForestClassifier(
        n_estimators=100, random_state=0, max_features=2
    )
    rf.fit(X_train, y_train)

    # RandomForestClassifier has predict_proba, but not decision_function
    fpr_rf, tpr_rf, thresholds_rf = metrics.roc_curve(
        y_test, rf.predict_proba(X_test)[:, 1]
    )

    plt.plot(fpr_rf, tpr_rf, label="RF ROC")
    close_zero_rf = np.argmin(np.abs(thresholds_rf - 0.5))
    plt.plot(
        fpr_rf[close_zero_rf],
        tpr_rf[close_zero_rf],
        "^",
        markersize=10,
        label="rf threshold 0.5",
        fillstyle="none",
        c="k",
        mew=2,
    )
    plt.xlabel("FPR")
    plt.ylabel("TPR (recall)")
    plt.legend(loc=4)

    if DEBUG:
        plt.show()

    svc_auc = metrics.roc_auc_score(y_test, svc.decision_function(X_test))
    rf_auc = metrics.roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    print(f"SVC AUC: {svc_auc} RF AUC: {rf_auc}")


def test_accuracy_vs_roc() -> None:
    """Accuracy is not the best metric to use for binary classification
    problems - ROC AUC is. When classes are imbalanced, accuracy will be high.
    However the AUC will not always be.

    This test shows that for different values of gamma, we achieve the same
    accuracy. However the AUCs are different. The point closest to the top left
    is the optimal threshold.

    Use AUC as a metric for binary classifier evaluation, particularily when
    classes are imbalanced. It's a much better metric than accuracy.

    Once you have the optimal threshold, use it when making predictions on new
    data.
    """
    digits = datasets.load_digits()  # 8x8 image of a digit
    X = digits.data
    y = digits.target == 9

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=0
    )

    accuracy = []
    auc = []
    for gamma in [1, 0.05, 0.01]:
        svc = svm.SVC(gamma=gamma).fit(X_train, y_train)
        accuracy.append(svc.score(X_test, y_test))
        auc.append(metrics.roc_auc_score(y_test, svc.decision_function(X_test)))

    print(f"Accuracies: {accuracy}")
    print(f"auc: {auc}")
    # All accuracies are the same
    for acc in accuracy[1:]:
        assert acc == accuracy[0]

    # While all the AUC scores are different. This proves that accuracy is *not*
    # the best indicator for binary classification models.
    for a in auc[1:]:
        assert a != auc[0]


def test_grid_search_by_metric() -> None:
    """When performing grid search, you can use any metric you want when
    performing model selection by passing the metric to the `scoring` parameter.

    The most important values for "scoring":

    Classification:

    * `accuracy`
    * `roc_auc`
    * `average_precision` - Area under the precision / recall curve
    * `f1`, `f1_macro` and `f1_weighted` for the binary f1 score

    Regression:

    * `r2`
    * `mean_squared_error`
    * `mean_absolute_error`
    """

    digits = datasets.load_digits()  # 8x8 image of a digit
    X = digits.data
    y = digits.target == 9
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=0
    )
    svc = svm.SVC()
    #
    # The default classification scoring method is "accuracy"
    #
    cv_score = model_selection.cross_val_score(svc, X, y, scoring="accuracy", cv=5)
    print(f"Accuracy: {cv_score}")

    #
    # AUC
    #
    cv_score = model_selection.cross_val_score(
        svc, X, y, scoring="average_precision", cv=5
    )
    print(f"Precision-recall AUC: {cv_score}")

    #
    # Using different metrics w/ GridSearchCV
    #
    # we provide a somewhat bad grid to illustrate the point:
    param_grid = {"gamma": [0.0001, 0.01, 0.1, 1, 10]}
    grid = model_selection.GridSearchCV(svc, param_grid=param_grid)
    grid.fit(X_train, y_train)
    print("Grid-Search with accuracy")
    print("Best parameters:", grid.best_params_)
    print("Best cross-validation score (accuracy)): {:.3f}".format(grid.best_score_))
    print(
        "Test set average precision (AUC): {:.3f}".format(
            metrics.average_precision_score(y_test, grid.decision_function(X_test))
        )
    )
    print(
        "Test set accuracy: {:.3f}".format(
            # identical to grid.score here
            metrics.accuracy_score(y_test, grid.predict(X_test))
        )
    )

    #
    # Using AUC as a scoring metric. Notice best_params is different using AUC.
    #
    # As expected, the "accuracy" metric is higher when selecting by `accuracy``,
    # "average precision" is higher when selecting based on `average_precision`.
    #
    grid = model_selection.GridSearchCV(
        svc, param_grid=param_grid, scoring="average_precision"
    )
    grid.fit(X_train, y_train)
    print("Grid-Search with average precision")
    print("Best parameters:", grid.best_params_)
    print(
        "Best cross-validation score (average precision): {:.3f}".format(
            grid.best_score_
        )
    )
    print(
        "Test set average precision: {:.3f}".format(
            # identical to grid.score here
            metrics.average_precision_score(y_test, grid.decision_function(X_test))
        )
    )
    print(
        "Test set accuracy: {:.3f}".format(
            metrics.accuracy_score(y_test, grid.predict(X_test))
        )
    )
