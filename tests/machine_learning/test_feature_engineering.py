"""
Feature engineering is the process of creating features which allow the model to
predict accurately.

Feature reduction
-----------------
* Multiple highly correlated features may not both be necessary. Consider
  removing one.
* Features where most values are the same provide little value. There is no
  variance or prediction power.
* Features which aren’t statistically related to the target have no value.

pandas.corr() - pairwise correlations. Two highly correlated variables may be
redundant.

There are three ways to measure feature importance:

* Before training (correlation)
* After training (importances)
* Statistical tests (ANOVA)
    * VarianceThreshold: only keep features exceeding a given variance
      threshold. Features with low variance do not contribute to the model.
    * Univariate tests: Test each feature for statistical significance and keep
      only the best.
        * ANOVA: Analysis of variance
        * T-test of means
        * Linear regressions (L1)


Recursive feature elinination (RFE) iteratively trains a model, drops the least
valuable features with each iteration. It stops when reducing features no longer
improves model performance.


"""
