import pandas as pd


def test_pandas_categoricals() -> None:
    """Pandas get_dummies() will create dummy coliumns for all object columns.

    You can also get_dummies for specific column(s).

    When working with ML datasets, it's important you get_dummies on the entire
    data set (before train_test_split). If you get_dummies on each individual
    set, you can't be certain you'll end up with the correct columns (as some
    categorical values may only be in one of the two sets.)

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

    # print out all categorical columns. By default, pandas will get_dummies`
    # for all "object" and "category" columns.
    for col in df.select_dtypes(include=["object", "category"]):
        print(f"value counts for: {col}")
        print(df[col].value_counts())

    df2 = pd.get_dummies(df, dtype=float)

    # We should have the following columns:
    #
    # * (1) age
    # * (2) gender_[male|female]
    # * (3) employment_type_[full_time|contract|part_time]
    # * (1) income

    assert df2.shape == (5, 7)
    print(df2.describe())
    print(df2.head())

    # Single column only
    df3 = pd.get_dummies(df, columns=["gender"])
    assert "gender_male" in df3.columns and "gender_female" in df3.columns
