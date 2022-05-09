"""Pipelines provide the ability to wrap multiple seqential operations together.

Pipelinese and GridSearchCV

Pipelines are handy when using GridSearchCV as you can use different parameters
for each step in the pipeline.

GridSearchCV in a pipeline splits data into training / validation sets *before*
the pipeline steps run, ensuring preprocessing steps are fitted on *just* the
training set, not the validation set.

"""
from sklearn import (
    datasets,
    ensemble,
    linear_model,
    model_selection,
    pipeline,
    preprocessing,
    svm,
)
from matplotlib import pyplot as plt


def test_simple_pipeline() -> None:
    X, y = datasets.load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=0
    )

    #
    # Manually scaling and scoring with multiple steps.
    #
    scaler = preprocessing.MinMaxScaler()
    svc = svm.SVC()

    X_train_scaled = scaler.fit_transform(X_train)
    svc.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)
    manual_score = svc.score(X_test_scaled, y_test)

    #
    # Using a pipeline to scale and score in a single step.
    #
    pipe = pipeline.Pipeline(
        steps=[
            ("scaler", preprocessing.MinMaxScaler()),
            ("svc", svm.SVC()),
        ],
    )

    pipe.fit(X_train, y_train)
    pipe_score = pipe.score(X_test, y_test)

    assert manual_score == pipe_score


def test_pipeline_with_grid_search_cv() -> None:
    """When using Pipelines w/ GridSearchCV, you can specify parameters for each
    step in the pipeline.

    GridSearchCV will split the training data into training and validation sets
    for each fold before starting the pipeline. This ensures the validation set
    is *not* used in the preprocessing or training steps.
    """

    X, y = datasets.load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=0
    )

    # Use the format: `step__param`
    param_grid = {
        "svc__C": [0.001, 0.01, 0.1, 1],
        "svc__gamma": [0.001, 0.01, 0.1, 1],
    }

    pipe = pipeline.Pipeline(
        steps=[
            ("scaler", preprocessing.MinMaxScaler()),
            ("svc", svm.SVC()),
        ],
    )

    grid = model_selection.GridSearchCV(pipe, param_grid=param_grid, cv=5)
    grid.fit(X_train, y_train)

    print(f"Best CV accuracy: {grid.best_score_:.2f}")
    print(f"Test score: {grid.score(X_test, y_test)}")
    print(f"Best parameters: {grid.best_params_}")

    # Examining a pipeline
    print(f"Steps {pipe.steps}")

    # Getting the best estimator (the entire pipeline which produced the best
    # results):
    best_est = grid.best_estimator_
    print(f"Best est: {best_est}")

    # Accessing each step by name
    svc: svm.SVC = best_est.named_steps["svc"]
    print(f"SVC: {svc}")


def test_grid_search_multiple_steps() -> None:
    X, y = datasets.load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=0
    )

    pipe = pipeline.Pipeline(
        steps=[
            ("scaler", preprocessing.StandardScaler()),
            ("polynomial", preprocessing.PolynomialFeatures()),
            ("ridge", linear_model.Ridge()),
        ]
    )

    param_grid = {
        "polynomial__degree": [1, 2, 3],
        "ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
    }

    grid = model_selection.GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    # Plot the results
    plt.matshow(
        grid.cv_results_["mean_test_score"].reshape(3, -1), vmin=0, cmap="viridis"
    )
    plt.xlabel("ridge__alpha")
    plt.ylabel("polynominal__degree")
    plt.xticks(range(len(param_grid["ridge__alpha"])), param_grid["ridge__alpha"])
    plt.yticks(
        range(len(param_grid["polynomial__degree"])), param_grid["polynomial__degree"]
    )
    plt.colorbar()

    # We can see from the grid that polynomial features 2 helps, but 3rd degree
    # polynomials make thing worse (especially 3)
    #
    # Uncomment to see grid!
    #
    # plt.show()

    print(f"Best params: {grid.best_params_}")
    print(f"Test score: {grid.score(X_test, y_test)}")


def test_grid_searching_multiple_models() -> None:
    """Here, we use grid search to determine which model to use."""

    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=0
    )

    pipe = pipeline.Pipeline(
        steps=[
            ("preprocessing", preprocessing.StandardScaler()),
            ("classifier", svm.SVC()),
        ]
    )

    param_grid = [
        {
            "classifier": [svm.SVC()],
            "preprocessing": [preprocessing.StandardScaler()],
            "classifier__gamma": [0.001, 0.01, 0.1, 1, 10, 100],
            "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        },
        {
            "classifier": [ensemble.RandomForestClassifier()],
            "preprocessing": [None],  # RF doesn't need preprocessing, skip this step.
            "classifier__max_features": [1, 2, 3],
        },
    ]

    grid = model_selection.GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_}")
    print(f"Test score: {grid.score(X_test, y_test)}")
