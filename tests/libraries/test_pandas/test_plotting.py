import matplotlib.pyplot as plt
import pandas as pd
import pytest


# Mark all tests this module as 'plot'. These tests will be skipped with `make
# test` since they block when GUI rendering.
pytestmark = pytest.mark.plot


def test_hist() -> None:
    df = pd.DataFrame(
        {
            "one": [0, 0.5, 1, 2, 2.5, 3, 4, 5, 6, 7],
            "two": [0, 1, 2, 4, 5, 8, 10, 13, 15, 20],
        }
    )
    df.plot.line()
    df.plot.scatter(x="one", y="two")
    df.plot.bar(x="one", y="two")

    plt.show()
