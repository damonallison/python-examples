"""Model evaluation includes the following:

* Determining how a model performs on unseen (test) data
* Hyperparameter tuning to build an optimal model * GridSearchCV

Model Evaluation
----------------
Simple model evaluation is done with the `score` method (classification) or
accuracy metrics (MAE, RMSE for regression).

Cross Validation
----------------
Cross validation is a process of training and validating multiple models on
different slices of data. Cross validation is more robust since it ensures the
entire data set is used as the test set in one fold. There are multiple CV
strategies available, however KFold, StratifiedKFold, and GroupKFold are the
most common.

Grid Search
-----------
Grid search is the process of finding the best model hyperparameters by
"searching" thru a grid of possible hyperparameter combinations.

When performing grid search, we split the training set into "training" and
"validatin" sets. The validation set is used to determine how well the model
*should* perform on new data. We select the best model based on it's performance
on the validation set. We then test the model with the test set.

We do *not* want to include the test set in any grid search validation.
Otherwise, we "leak" data - the model's hyperparameters will be learned in part
on the test set, which we need to avoid.

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

* If the best stores are on the edges (w/ maximum or minimum values for 1 or
  more parameters), your search space is not wide enough for those parameters.
  The model may get more accurate if you increase or decrease those parameter
  values.

* If accuracy is not changing between the min / max values of a parameter,
  either the parameter is not important to the results or you haven't chosen a
  wide enough search space for that parameter. Try increasing the range of the
  parameter. If the increased range still yields consistent accuracy, that
  parameter is not important to tune.



Model metrics and scoring
------------------------

Metrics for binary classification
---------------------------------

* False positive: incorrect true prediction. Type 1 error.
* False negative: incorrect false prediction. Type 2 error.

When evaluating a binary classifier, associate a cost to each error type.

Watch for imbalanced data sets. If 99% of emails are spam, just predicting every
email will be spam will be 99% accurate. With imbalanced data sets, accuracy is
an inadequate measure for quantifying predictive performance.

We need alternative metrics to determine which models perform well.

Confusion Matrix
----------------
A confusion matrix gives you the true / false positive / negative counts.



"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets, linear_model, metrics, model_selection, svm


def test_score() -> None:
    """For classification problems, score is the fraction of correctly
    classified examples.
    """
    X, y = datasets.make_blobs(random_state=0)

    # make_blobs makes gaussian clusters
    #
    # Show a scatterplot
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=0
    )

    lr = linear_model.LogisticRegression().fit(X_train, y_train)
    score = lr.score(X_test, y_test)
    print(f"Test score = {score}")


def test_cross_validation() -> None:
    """Cross validation is a more robust way to assess generalization
    performance.

    Cross validation splits the data into multiple "folds", building a model on
    each fold. Scores are determined on each fold. The aggregate score (mean,
    for example) is determined for an overall model "score".

    Benefits of CV:

    1. All data is used in a test set.

    Cross validation is more robust since multiple train / test splits are
    scored. Also, all data is used in a test set in one of the folds. This gives
    us more confidence in the overall model generalization performance (score).

    By default, train_test_split will randomly create the train / test split. If
    we are "unlucky" and receive all hard to score examples in our test set, the
    model score will be artificially low. CV uses *all* data as test data at one
    point, giving us a better idea of our model's true score.

    2. CV helps determine model sensitivity.

    When we see accuraces between, say, 80% and 90%, we can be reasonably
    confident our overall model performance will be somewhere int he 80%s.

    A wider variance gives us less confidence in overall model performance.

    3. We use our data more effectively.

    The more number of folds we use, the more data we can include in the
    training set, which generally results in better model performance.

    --

    CV is expensive since you are building a model for each fold. However, the
    confidence you get from using CV is typically worth the extra compute cost.
    """

    X, y = datasets.load_iris(return_X_y=True)
    lr = linear_model.LogisticRegression(max_iter=1000, random_state=0)

    # At least 5-fold CV is recommended
    #
    # Note that for classification tasks (like LogisticRegression), sckit-learn
    # uses stratified k-fold, which ensures the test set contains the same
    # overall proportion of each label as the entire data set. This ensures our
    # test set is a more accurate representation of the whole.
    #
    # NOTE: Always use stratified k-fold CV for classification tasks.
    #
    scores = model_selection.cross_val_score(lr, X, y, cv=5)
    print(f"CV scores: {np.round(scores, 2)}")
    print(f"Mean score: {scores.mean():.2f}")

    # cross_validate is similar to cross_val_score, but returns a dictionary
    # containing additional data:
    #
    # * training times
    # * test times
    # * training score (optional)
    # * test scores

    cv_results = model_selection.cross_validate(
        lr,
        X,
        y,
        cv=3,
        n_jobs=-1,
        verbose=0,
        return_estimator=False,
        return_train_score=True,
    )

    df = pd.DataFrame(cv_results)
    print(df)
    print(f"Means: {df.mean()}")


def test_cross_validation_custom_splitter() -> None:
    """A custom splitter (KFold) can be used to control the k-fold splitting
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
    kfold = model_selection.StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    scores = model_selection.cross_val_score(lr, X, y, cv=kfold)
    assert scores.mean() > 0.50


def test_leave_one_out_cv() -> None:
    """Leave one out corss validation will create a fold with a single sample as
    the entire test set.

    This is really expensive (n folds == n models) but could produce better
    results when you have smaller data sets.
    """
    X, y = datasets.load_iris(return_X_y=True)

    lr = linear_model.LogisticRegression(max_iter=1000, random_state=0)
    loo = model_selection.LeaveOneOut()
    scores = model_selection.cross_val_score(lr, X, y, cv=loo)
    print(f"Number of folds: {len(scores)}")
    print(f"Mean score: {scores.mean()}")


def test_shuffle_split() -> None:
    """Shuffle split will split a configured number of samples for the training
    and test sets.

    Shuffle split allows you to control the number of iterations (folds)
    independently of the data. For exanple, you could use a training size of 50%
    and a test size of 20% (leaving 30% of the data out).

    This can be used with large data sets. This can be useful for experimenting
    with large data sets.
    """

    X, y = datasets.load_iris(return_X_y=True)

    lr = linear_model.LogisticRegression(max_iter=1000, random_state=0)
    shuffle = model_selection.StratifiedShuffleSplit(
        n_splits=10, train_size=0.5, test_size=0.1
    )
    scores = model_selection.cross_val_score(lr, X, y, cv=shuffle)
    print(f"Mean score: {scores.mean()}")


def test_cv_with_groups() -> None:
    """Grouping ensures that data belonging to the same group (say faces for
    facial recognition), are not split between the training and test sets.

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
    print(f"Mean score: {scores.mean()}")


def test_grid_search() -> None:
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=0
    )

    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "gamma": [0.001, 0.01, 0.1, 1, 10, 100],
    }
    # Since SVC is a classifier, stratified CV is used
    gs = model_selection.GridSearchCV(
        svm.SVC(), param_grid=param_grid, cv=5, return_train_score=True
    )

    #
    # GridSearchCV will find the best hyperparameters and fit / return a new
    # model w/ them, which can be found with `best_estimator_`. Calling `score`
    # or `predict` on the grid search object will use the best estimator.
    #
    gs.fit(X_train, y_train)
    print(f"gs best params: {gs.best_params_}")

    #
    # All CV results are stored in cv_results_. Each record in cv_results_
    # represents a different parameter combination.
    #
    cv_results = pd.DataFrame(gs.cv_results_)
    print(cv_results.head())

    #
    # Plot the results in a heat map.
    #
    # This is an important step to determine if you're grid is setup correctly.
    # It shows you if you have selected the right parameters to tune and have
    # set appropriate ranges for each parameter.
    #
    # scores = np.array(cv_results["mean_test_score"]).reshape(6, 6)

    # sns.heatmap(
    #     scores,
    #     xticklabels=param_grid["gamma"],
    #     yticklabels=param_grid["C"],
    #     annot=True,
    #     fmt=".2f",
    # )
    # plt.show()
    #
    # The only place in the grid search process the test set is used is during
    # final model scoring. The test set is *NOT* used to find the best model.
    #
    print(f"gs score: {gs.score(X_test, y_test)}")


def test_grid_search_numtiple_grids() -> None:
    """In some cases, not all parameters will be valid when used together.

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
    """Nested cross validation performs two cross validations.

    First, For each parameter combination, GridSearchCV will use cross
    validation to determine the mean test score by splitting the training set
    into training / validation and building / testing a model for each fold.

    Second, CV is used to vary the training / test split.

    In the examples above, we are only splitting the data once into train / test
    sets. This makes model performance dependent on a single split of the data
    (which is why we do CV).

    In nested cross-validation, there is an outer loop over splits of the data
    into training and test sets. For each of them, a grid search is run (which
    might result in different best parameters for each split in the outer loop).
    Then, for each outer split, the test set score using the best settings is
    reported.

    Nested CV does *not* return a model, it returns a list of scores. The scores
    tells us how well a model generalizes. Nested CV is used to determine how
    well a given model works on a dataset.

    Note that nested CV is expensive. Here, we have 4 * 2 = 6 parameter
    combinations. We use 5-fold CV within GridSearchCV, resulting in 6 * 5 = 30
    models being trained. We use 5-fold CV to validate the entire dataset, which
    results in a total of 30 * 5 == 150 models.
    """

    X, y = datasets.load_iris(return_X_y=True)

    param_grid = [
        {"kernel": ["rbf"], "C": [0.01, 0.1], "gamma": [0.01, 0.1]},
        {"kernel": ["linear"], "C": [0.01, 0.1]},
    ]
    grid_search = model_selection.GridSearchCV(svm.SVC(), param_grid, cv=5, n_jobs=-1)
    scores = model_selection.cross_val_score(grid_search, X, y, cv=5, n_jobs=-1)
    #
    # The result of nested CV can be summarized as "SVC can achieve a 97% mean
    # CV accuracy on the iris dataset". It will *not* produce a model.
    #
    # print(f"CV scores: {scores}")
    # print(f"Mean CV Score: {scores.mean()}")
    assert scores.mean() > 0.0


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

    F-score = 2 * (precision * recall) / precision + recall

    """

    digits = datasets.load_digits()  # 8x8 image of a digit
    y = digits.target == 9

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        digits.data, y, random_state=0
    )

    logreg = linear_model.LogisticRegression(C=0.1, max_iter=10000).fit(
        X_train, y_train
    )
    logreg_pred = logreg.predict(X_test)

    confusion = metrics.confusion_matrix(y_test, logreg_pred)
    print(f"Confusion matrix {confusion}")

    print(f"True negative: {confusion[0][0]}")
    print(f"False positive: {confusion[0][1]}")
    print(f"False negative: {confusion[1][0]}")
    print(f"True positive: {confusion[1][1]}")
