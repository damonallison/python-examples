"""
numpy
----
https://en.wikipedia.org/wiki/NumPy

numpy (numerical python) is a numerical computing library. It provides an array
data type (ndarray: n-dimensional array) and efficient computations on them.

Appending to an ndarray is not as simple as in python. ndarrays are immutable
from a programmer's standpoint. Appending to an ndarray will return a new
ndarray.

numpy computation is CPU based and arrays are loaded into memory. Computations
cannot run on GPUs (Tensorflow, JAX) or can be distributed across clusters of
CPUs (Dask).

Views
-----
Many common operations (assignment, slicing, reindexing, and others) create
views into the underlying array. Use `copy` to make a deep copy of the array.
For example, if you're taking a small slice from a large array, making a `copy`
allows the large array to be GC'd.
"""

import numpy as np


def test_numpy_creation() -> None:
    """Creating numpy arrays.

    Numpy arrays are faster, more compact, and more memory efficient than python
    lists. Numpy allows you to specify a data types, which allows further
    optimization.

    numpy arrays are not as flexible as python arrays. For example, appending to
    a numpy array will return a new array.
    """

    # You can create a numpy array from a python array, optionally specifying a
    # type. If a type is not specified, it is inferred.
    a = np.array([1, 2, 3], dtype=np.int64)
    assert a.ndim == 1
    assert a.shape == (3,)  # `shape` is always a tuple
    assert a.size == 3  # Total elements of the array (all dimensions)
    assert a.dtype == np.int64

    # In numpy, dimensions are called axes. axis=0 are rows, axis=1 are columns
    a2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert a2.dtype == np.int64
    assert a2.shape == (3, 3)
    assert a2.ndim == 2

    # ndarraysize() == total element count
    assert a2.size == 9
    assert np.size(a2) == a2.size

    # Creating arrays of zeros / ones
    assert np.zeros((1, 9)).shape == (1, 9)
    assert np.ones((1, 9)).shape == (1, 9)

    # Slicing will create a *view* into the original array.
    a2 = a[:-1]
    assert np.array_equal(a2, np.array([1, 2]))

    # Modifications to a view updates the underlying array
    a2[0] = 11
    a2[1] = 22
    assert np.array_equal(a, np.array([11, 22, 3]))
    assert np.array_equal(a2, np.array([11, 22]))

    # Copy to create an entirely new array
    a = np.array([1, 2, 3], dtype=np.int64)
    a6 = a.copy()
    a6[0] = 11
    a6[1] = 22
    assert np.array_equal(a, np.array([1, 2, 3]))
    assert np.array_equal(a6, np.array([11, 22, 3]))


def test_numpy_indexing() -> None:
    """indexing is the process of selecting a subset of an array.

    numpy arrays can be indexed by Python-like slicing or by using boolean
    masks.
    """

    a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    # Each dimension contains a slice range. `:` is used for any dimension not specified in the slice.
    assert np.array_equal(a[2], [3, 3, 3])
    assert np.array_equal(a[2, :], [3, 3, 3])

    assert np.array_equal(a[0:2, 0], np.array([1, 2]))
    assert np.array_equal(a[0:2, 0:-1], np.array([[1, 1], [2, 2]]))

    # Boolean masking. Parens are optional for single conditions, but required for using and (&) or or (|).
    low = a[(a < 3)]
    # Notice how a 1D array is returned
    assert np.array_equal(low, [1, 1, 1, 2, 2, 2])

    # A multi-statement predicate
    low = a[(a == 1) | (a == 2)]
    assert np.array_equal(low, [1, 1, 1, 2, 2, 2])

    # Create a boolean mask (parens required)
    mask = a < 3
    assert np.array_equal(
        mask,
        [
            [True, True, True],
            [True, True, True],
            [False, False, False],
        ],
    )
    assert np.array_equal(a[mask], [1, 1, 1, 2, 2, 2])


def test_numpy_slicing() -> None:

    a = np.arange(0, 6).reshape(2, 3)
    #
    # [
    #   [0, 1, 2],
    #   [3, 4, 5]
    # ]
    #

    # from start to end of the first row, take every 2nd element
    assert np.array_equal(a[0][::2], np.array([0, 2]))

    # reverse the first row
    assert np.array_equal(a[0][::-1], np.array([2, 1, 0]))

    # when fewer indicies are provided than the number of axes, the missing
    # indices are considered complete slices (i.e., ":")
    assert np.array_equal(a[0], np.array([0, 1, 2]))
    assert np.array_equal(a[0, :], np.array([0, 1, 2]))

    # The ... represent as many colons as needed to produce a complete indexing
    # tuple.
    assert np.array_equal(a[0, ...], np.array([0, 1, 2]))

    assert np.array_equal(a[..., 0], np.array([0, 3]))

    # a 3D array (two stacked 2D arrays)
    a = np.array(
        [
            [
                [0, 1, 2],
                [10, 12, 13],
            ],
            [
                [100, 101, 102],
                [110, 112, 113],
            ],
        ]
    )
    # a[1] is the same as a[1, :, :]
    # a[1, 1] is the same as a[1, 1, :]
    assert np.array_equal(a[1, 1], np.array([110, 112, 113]))


def test_numpy_stacking_splitting() -> None:

    a1 = np.array([[1, 1], [2, 2]])
    a2 = np.array([[3, 3], [4, 4]])

    assert np.array_equal(np.hstack((a1, a2)), [[1, 1, 3, 3], [2, 2, 4, 4]])
    assert np.array_equal(np.vstack((a1, a2)), [[1, 1], [2, 2], [3, 3], [4, 4]])

    a3 = np.hstack((a1, a2))
    # Split into equally shaped arrays
    assert np.array_equal(np.hsplit(a3, 2), np.array([a1, a2]))


def test_array_manipulation() -> None:
    """arrays should be thought of as immutable.

    `concatenate` will add to an array

    `delete` will remove from an array
    """
    a = np.array([1, 2, 3])
    a2 = np.concatenate((a, [1, 2, 3]))

    assert np.array_equal(a2, [1, 2, 3, 1, 2, 3])

    a3 = np.delete(a2, (0, 1))  # The index or list of indices to delete
    assert np.array_equal(a3, [3, 1, 2, 3])


def test_reshaping() -> None:
    """Reshaping with -1 (unknown dimension) will find the appropriate dimension
    that fitst the data being reshaped. Only one unknown dimension can be used
    during reshaping.

    reshape: returns it's argument with a modified shape.

    resize: directly modifies it's argument (mutation)

    ravel: flattens an n-dimentional array into a 1D vector, returning a view
    into the original array

    flatten: flattens an n-dimensional array into a 1D vector, returning a copy
    of the original array

    """
    a = np.arange(0, 6)
    assert a.shape == (6,)

    # [[0 1 2 3 4 5]]
    a2 = a.reshape([1, -1])
    assert a2.shape == (1, 6)
    assert a2[0, 5] == 5

    # [[0], [1], [2], [3], [4], [5]]
    a3 = a.reshape(-1, 1)
    assert a3.shape == (6, 1)
    assert a3[5, 0] == 5

    # Note that reshape creates a *view* into the original array.
    # You need to .copy() the array if you want a copy.
    a3[0, 0] = 100
    assert a[0] == 100

    # flatten and ravel will both flatten an ndarray into a 1D vector.
    #
    # *warning* flatten will create a copy, ravel will not.
    a4 = a3.flatten()
    a4[0] = 200

    assert a4[0] == 200
    assert a3[0, 0] == 100  # a3 is unchanged since flatten created a copy

    a5 = a3.ravel()
    a5[0] = 1000

    assert a5[0] == 1000
    assert a3[0, 0] == 1000  # a3 is changed since ravel did *not* create a copy.


def test_stacking_splitting() -> None:
    """
    hstack and vstack allow you to append arrays horizontally or vertically

    hsplit and vsplit allow you to split arrays horizontally or vertically
    """
    a1 = np.arange(0, 3)
    a2 = np.arange(3, 6)

    assert np.array_equal(np.hstack((a1, a2)), np.arange(0, 6))
    assert np.array_equal(np.vstack((a1, a2)), np.arange(0, 6).reshape(2, 3))

    assert np.array_equal(np.hsplit(np.arange(0, 6), 2), np.arange(0, 6).reshape(2, 3))


def test_numpy_adding_removing() -> None:
    a1 = np.array([1, 2, 3])
    a2 = np.array([4, 5])

    a3 = np.append(a1, a2)
    assert np.array_equal(a3, np.array([1, 2, 3, 4, 5]))


def test_numpy_array_operations() -> None:
    """Arithmetic operators on arrays apply elementwise. A new array is created
    and filled with the result.

    Some operations (+=, *=) operate in-place.
    """

    a1 = np.array([1, 2])
    a2 = np.ones(2)

    # Elementwise addition
    assert np.array_equal(a1 + a2, np.array([2, 3]))
    assert np.array_equal(a1 - a2, np.array([0, 1]))
    assert np.array_equal(a1 * a2, np.array([1, 2]))

    # Broadcasting - applying an operation to all elements in an array.
    #
    # In this case, numpy will treat the scaler '2' as [2, 2] to conform to the
    # length of `a1`.
    assert np.array_equal((a1 * 2), np.array([2, 4]))

    # Summary statistics
    assert a1.sum() == 3
    assert a1.min() == 1
    assert a1.max() == 2
    assert a1.mean() == 1.5

    # Note that summary statistics happen for the entire array. To get column or
    # row statistics, use `axis`.
    a3 = np.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ]
    )
    assert a3.sum() == 18
    assert np.array_equal(a3.sum(axis=0), [6, 6, 6])
    assert np.array_equal(a3.sum(axis=1), [3, 6, 9])
    assert np.array_equal(a3.max(axis=0), [3, 3, 3])
    assert np.array_equal(a3.min(axis=1), [1, 2, 3])


def test_numpy_matrices() -> None:
    m = np.arange(0, 6).reshape(2, 3)

    #
    # m = [
    #       [0, 1, 2],
    #       [3, 4, 5]
    #     ]
    #

    # the axis parameter is the axis that gets *collapsed*
    #
    # when we set axis=0, we are collapsing rows and calculating the sum
    # (effectively summming columns, which is a bit confusing)
    #
    # when we set axis=1, we are collapsing *columns* and calculating the sum
    # (effectively summing rows, which is a bit confusing)
    assert np.array_equal(np.array([3, 5, 7]), m.sum(axis=0))
    assert np.array_equal(np.array([3, 12]), m.sum(axis=1))


def test_numpy_rng() -> None:
    """Random number generation"""
    rng = np.random.default_rng()
    #
    i = np.random.randint(1, 101)
    assert i >= 1 and i < 101


def test_numpy_statistics() -> None:
    assert 12 == np.max(np.array([1, 2, 10, 12]))

    # [
    #   [0, 1, 2, 3],
    #   [4, 5, 6, 7],
    #   [8, 9, 10, 11],
    # ]
    a = np.arange(12).reshape(3, 4)

    # axis=0 will sum up each column
    assert np.array_equal(a.sum(axis=0), np.array([12, 15, 18, 21]))

    # axis=1 will sum up each row
    assert np.array_equal(a.sum(axis=1), np.array([6, 22, 38]))


#
# Advanced topics
#


def test_numpy_broadcasting_rules() -> None:
    # Arrays with the size of 1 along a particular dimension act as if they had
    # the size of the array with the largest shape along that dimension.
    #
    # Here, [2, 2] is treated as [[2, 2], [2, 2]] for broadcasting purposes.

    a1 = np.array([[1, 2], [3, 4]])
    a2 = np.array([2, 2])
    print(a1 + a2)


def test_numpy_indexing_boolean_arrays() -> None:
    """Boolean arrays are masks which specify if each element should be included."""
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Boolean arrays
    #
    # Notice the filter returns a 1D array with the selected elements.
    a[a > 5] == np.array([6, 7, 8, 9])
    a[(a > 5) & (a < 8)] == np.array([6, 7])

    # Here's what the boolean mask will look like. Notice it's a 3s3 2D array -
    # the same shape as the original array.
    mask = (a > 5) & (a < 8)
    assert np.array_equal(
        mask,
        np.array([[False, False, False], [False, False, True], [True, False, False]]),
    )
