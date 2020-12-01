from matplotlib import pyplot as plt


def plot():
    years = [1950, 1960, 1970]
    gdp = [300.2, 222.23, 3489.123]

    plt.plot(years, gdp, color="green", marker="o", linestyle="solid")
    plt.title("Nominal GDP")
    plt.ylabel("Billions of $")
    plt.show()


def vector():
    """Vectors are simply numeric lists.

    Lists are slow, use NumPy arrays.
    """
    def vector_add(v, w):
        return [vi + wi for vi, wi in zip(v, w)]

    print("running vector")
    assert [4, 4] == vector_add([1, 3], [3, 1])


def matrices():
    """Matrics are a set of (k x n) vectors.

    k = 3 (width) n = 2 (depth)

    [[1, 0, 2],
     [0, 1, 2]]

     Matrices are efficient for data lookup. For example, a matrix can be used
     to represent relationships between two variables.

     friendships = [(0, 1), (0, 2), (1, 2)]

    Could be represented in a matrix as:

     friendships = [[0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0]]

    """

    f = [(0, 1), (0, 2), (1, 2)]

    def make_matrix(rows, cols, entry_fn):
        return [[entry_fn(i, j) for j in range(cols)] for i in range(rows)]

    def are_friends(i, j):
        for x, y in f:
            if (x == i and y == j) or (x == j and y == i):
                return 1
        return 0

    m = make_matrix(len(f), len(f[0]), are_friends)
    assert not m[0][0]
    assert not m[1][1]
    assert m[0][1]
    assert m[2][0]


if __name__ == "__main__":
    matrices()
