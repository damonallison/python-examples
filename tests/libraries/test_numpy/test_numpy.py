import numpy as np


def test_numpy_creation() -> None:
    """Creating numpy arrays.

    Numpy arrays are faster, more compact, and more memory efficient than python
    lists. Numpy allows you to specify a data types, which allows further
    optimization.
    """
    a = np.array([1, 2, 3], dtype=np.int64)
    assert a.shape == (3,)
    assert a.dtype == np.int64

    # In numpy, dimensions are called axes. axis=0 are rows, axis=1 are columns
    a2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert a2.dtype == np.int64
    assert a2.shape == (3, 3)
    assert a2.ndim == 2
    assert a2.size == 9

    # Creating arrays of zeros / ones
    assert np.zeros((1, 9)).shape == (1, 9)
    assert np.ones((1, 9)).shape == (1, 9)

    # Slicing will create a *view* into the original array.
    a5 = a[:-1]
    assert np.array_equal(a5, np.array([1, 2]))

    # Modifications to a view updates the underlying array
    a5[0] = 11
    a5[1] = 22
    assert np.array_equal(a, np.array([11, 22, 3]))  # a hasn't changed
    assert np.array_equal(a5, np.array([11, 22]))

    # Copy to create an entirely new array
    a = np.array([1, 2, 3], dtype=np.int64)
    a6 = a.copy()
    a6[0] = 11
    a6[1] = 22
    assert np.array_equal(a, np.array([1, 2, 3]))
    assert np.array_equal(a6, np.array([11, 22, 3]))


def test_reshaping() -> None:
    # Reshaping with -1 (unknown dimension) will find the appropriate dimension
    # that fitst the data being reshaped. Only one unknown dimension can be used
    # during reshaping.
    a = np.arange(0, 6)

    a2 = a.reshape([1, -1])
    assert a2.shape == (1, 6)

    a3 = a.reshape(-1, 1)
    assert a3.shape == (6, 1)

    # Note that reshape creates a *view* into the original array.
    # You need to .copy() the array if you want a copy.
    a3[0, 0] = 100
    assert a[0] == 100

    # flatten and ravel will both flatten an ndarray into a 1D vector.
    #
    # flatten will create a copy, ravel will not.

    a4 = a3.flatten()
    a4[0] = 200

    assert a4[0] == 200
    assert a3[0, 0] == 100  # a3 is unchanged since flatten created a copy

    a5 = a3.ravel()
    a5[0] = 1000

    assert a5[0] == 1000
    assert a3[0, 0] == 1000  # a3 is changed since ravel did *not* create a copy.


def test_numpy_indexing() -> None:
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Boolean indexing
    a[a > 5] == np.array([6, 7, 8, 9])
    a[(a > 5) & (a < 8)] == np.array([6, 7])

    # Masking
    mask = (a > 5) & (a < 8)
    assert np.array_equal(
        mask,
        np.array([[False, False, False], [False, False, True], [True, False, False]]),
    )


def test_numpy_array_operations() -> None:
    a1 = np.array([1, 2])
    a2 = np.ones(2)

    # Elementwise addition
    assert np.array_equal(a1 + a2, np.array([2, 3]))
    assert np.array_equal(a1 - a2, np.array([0, 1]))
    assert np.array_equal(a1 * a2, np.array([1, 2]))

    # Broadcasting - applying an operation to all elements in an array
    assert np.array_equal((a1 * 2), np.array([2, 4]))

    # Summary statistics
    assert a1.sum() == 3
    assert a1.min() == 1
    assert a1.max() == 2
    assert a1.mean() == 1.5


def test_numpy_matrices() -> None:
    m = np.arange(0, 6).reshape(2, 3)
    #
    # m = [
    #      [0, 1, 2],
    #      [3, 4, 5]
    #     ]

    # the axis parameter is the axis that gets *collapsed*
    #
    # when we set axis=0, we are collapsing rows and calculating the sum
    # (effectively summming columns, which is a bit confusing)
    #
    # when we set axis=1, we are collapsing *columns* and calculating the sum
    # (effectively summing rows, which is a bit confusing)
    assert np.array_equal(np.array([3, 5, 7]), m.sum(axis=0))
    assert np.array_equal(np.array([3, 12]), m.sum(axis=1))
