"""
matplotlib was started to create a MATLAB-like plotting interface in Python.
It's rather old school and primitive now, providing basic figure / axis based
plots. Newer, interactive, plotting libraries like plotly are

Seaborn is an add-on toolkit which uses matplotlib for underlying plotting. It
simplifies creating common visualization types.

Bokeh, Altair, and Plotly are newer, web based, interactive data visualization
libraries.
"""

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Mark all tests this module as 'plot'. These tests will be skipped with `make
# test` since they block when GUI rendering.
pytestmark = pytest.mark.plot


def test_simple_plot() -> None:
    data = np.arange(10)

    # figures are the highest level matplotlib container which contains
    # subplots. Each subplot is an AxesSubplot instance, which are plotted on.
    fig = plt.figure(figsize=[10, 10])
    # add plots (ax) to a figure. in this case, we add 2x2 - up to (4) plots
    # total. The objects returned are AxesSubplot objects, which can be plotted
    # on.
    ax1 = fig.add_subplot(2, 2, 1)  # select the first subplot
    ax2 = fig.add_subplot(2, 2, 2)  # the second subplot
    ax3 = fig.add_subplot(2, 2, 3)  # the third subplot
    _ = fig.add_subplot(2, 2, 4)  # the 4th subplot

    # matplotlib provides global (top-level) plotting functions that hide the
    # details of working with axis objects. It's preferred to use the axis
    # objects.

    ax1.plot(data, color="red", linestyle="dashed", marker="o", label="a line")
    ax1.plot(
        np.arange(10)[::-1],
        color="blue",
        linestyle="dashed",
        marker="o",
        label="another line",
    )
    ax1.set_xlim(-1, 11)
    ax1.set_ylim(-1, 11)

    # Customize the x / y axis labels and ticks.
    ax1.set_title("how oatmeal effects IQ")
    ax1.set_xlabel("iq")
    ax1.set_xticks(np.arange(0, 11, 2))
    ax1.set_xticklabels(ax1.get_xticks(), rotation=30, fontsize=8)
    ax1.set_ylabel("oatmeal")

    ax1.legend()

    ax2.hist(np.random.standard_normal(100), bins=20, color="green", alpha=0.5)
    ax3.scatter(np.arange(30), np.arange(30) + 3 * np.random.standard_normal(30))

    # Create and register a second figure with pyplot
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(1, 1, 1)
    ax1.plot(data)

    # If you want to manually remove a figure from pyplot, call `plt.close` with
    # the figure.
    plt.close(fig2)

    # Display all open figures. After blocking, all figures are unregistered
    # from pyplot.
    plt.show(block=True)


def test_plot_saving(tmp_path: pathlib.Path) -> None:
    png_path = tmp_path / "test.png"

    fig, ax = plt.subplots()
    ax.plot(np.arange(10))
    fig.savefig(png_path)
    plt.close(fig)

    # Show the image within a plot
    img = plt.imread(png_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.show(block=True)


def test_plot_global_configuration() -> None:
    # matplotlib allows you to customize it's global configuration using `rc`
    try:
        plt.rc("font", family="Hack Nerd Font")
        fig, ax = plt.subplots()
        ax.plot(np.arange(100))
        plt.show(block=True)

    finally:
        # Reset matplotlib's defaults
        plt.rcdefaults()
