"""Pandas introduces a 'Categorical' type for dealing with repeated instances of
distinct values.

Categorical columns require (much) less memory and improve performance with
operations like "groupby".

By default, categorical columns are unordered. They can be made ordered, which
will allow you to do equality and mathematical comparisons on the columns.


"""

import pandas as pd


def test_pandas_categoricals() -> None:
    """Pandas get_dummies() will create dummy columns for all object and
    categorical columns.

    You can also get_dummies for specific column(s).

    When working with ML datasets, it's important you get_dummies on the entire
    data set (before train_test_split). If you get_dummies on the train and test
    sets individually, you can't be certain you'll end up with the correct
    columns (as some categorical values may only be in one of the two sets.)

    Ensure numeric variables are truly continuous. Categorical columns could be
    encoded as integers (i.e., survey answers). pd.get_dummies will *not* create
    dummies for these columns by default.
    """

    df = pd.read_csv("./tests/libraries/test_pandas/data/test_categorical.csv")

    assert df.shape == (5, 4)
    assert df.columns[0] == "age"

    assert df["age"].dtype == int
    assert df["gender"].dtype == object
    assert df["employment_type"].dtype == object
    assert df["income"].dtype == int

    # By default nothing is categorical
    assert len(df.select_dtypes(include=["category"]).columns) == 0

    # Inspecting the data to determine which columns are potential categoricals.
    #
    # A low number of values with high value counts make ideal categorical
    # variables.
    #
    # print out all categorical columns. By default, pandas will get_dummies for
    # all "object" and "category" columns.
    for col in df.select_dtypes(include=["object"]):
        print(f"value counts for: {col}")
        print(df[col].value_counts())

    # Even with just a few rows, you can see the memory savings by using
    # categorical columns
    gender_mem_before = df.memory_usage(deep=True)["gender"]
    employment_type_mem_before = df.memory_usage(deep=True)["employment_type"]

    df[["gender", "employment_type"]] = df[["gender", "employment_type"]].astype(
        "category"
    )
    gender_mem_after = df.memory_usage(deep=True)["gender"]
    employment_type_mem_after = df.memory_usage(deep=True)["employment_type"]

    assert gender_mem_after < gender_mem_before
    assert employment_type_mem_after < employment_type_mem_before

    #
    # Categorical columns have "codes" and "categories". They are accessed with
    # the 'cat' special accessor.
    #
    assert set(df["gender"].cat.categories.values.tolist()) == {"male", "female"}
    assert set(df["employment_type"].cat.categories.values.tolist()) == {
        "full_time",
        "part_time",
        "contract",
    }

    df2 = pd.get_dummies(df, dtype=float)
    #
    # We should have the following columns:
    #
    # * (1) age
    # * (2) gender_[male|female]
    # * (3) employment_type_[full_time|contract|part_time]
    # * (1) income
    #
    assert df2.shape == (5, 7)
    print(df2.describe())
    print(df2.head())

    #
    # get_dummies for specific columns
    #
    df3 = pd.get_dummies(df, columns=["gender"])
    assert "gender_male" in df3.columns and "gender_female" in df3.columns
