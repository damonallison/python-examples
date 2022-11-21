"""Pandas introduces a 'Categorical' type for dealing with repeated instances of
distinct values.

Categorical columns require (much) less memory and improve performance with
operations like "groupby".

By default, categorical columns are unordered. They can be made ordered, but
numerical operations are not possible.

Order is assigned manually, not lexically (alphabetically). Each value is
internally given a unique integer value
"""

import pandas as pd


def test_pandas_categoricals() -> None:
    """Shows that, by default, categorical columns are read into DataFrame's as
    object and must be converted to "category" using .astype().

    This test also shows that category columns reduce DataFrame memory
    footprint.

    Finally, this test shows the use of get_dummies().

    Pandas get_dummies() will create dummy columns for all object and
    categorical columns.

    You can also get_dummies for specific column(s).

    When working with ML datasets, it's important you get_dummies on the entire
    data set (before train_test_split). If you get_dummies on the train and test
    sets individually, you can't be certain you'll end up with the correct
    columns (as some categorical values may only be in one of the two sets.)

    When working with real world data, some categorical columns could be encoded
    into numerical values. For example, an "age_group" column could be "0", "1",
    "2", "3". You'll need to manually convert these columns to "category" as
    pd.get_dummies() will not create dummies for numeric columns by default.
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
    # all "object" and "category" columns (we currently have no categorical
    # columns).
    for col in df.select_dtypes(include=["object", "category"]):
        assert (df[col].value_counts() > 0).all()
        # print(f"value counts for: {col}: df[col].value_counts()")

    # Even with just a few rows, you can see the memory savings by using
    # categorical columns
    gender_mem_before = df["gender"].nbytes
    employment_type_mem_before = df["employment_type"].nbytes

    df[["gender", "employment_type"]] = df[["gender", "employment_type"]].astype(
        "category"
    )
    gender_mem_after = df["gender"].nbytes
    employment_type_mem_after = df["employment_type"].nbytes

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
    df["gender"].cat.codes

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
    assert "gender_male" in df2.columns
    assert "employment_type_full_time" in df2.columns
    assert "income" in df2.columns

    #
    # get_dummies for specific columns
    #
    df3 = pd.get_dummies(df, columns=["gender"])
    assert "gender_male" in df3.columns and "gender_female" in df3.columns
