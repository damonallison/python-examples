import matplotlib.pyplot as plt
import pandas as pd
import pytest


@pytest.mark.skip(reason="Displays a plot")
def test_hist() -> None:
    df = pd.DataFrame({"one": [0, 0.5, 1, 2, 2.5, 3, 4, 5, 6]})
    df.hist(bins=2)
    plt.show()
