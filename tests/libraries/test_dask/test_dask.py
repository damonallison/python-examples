"""Dask provides multi-core and distributed parallel execution.

Dask implements high level collections (Array, Bag, and DataFrame) which mimic
Numpy, lists, and Pandas but operate in parallel.

Dask also exposes low level schedulers which can be used to parallelize any
computation. They are an alternative to `threading` or `multiprocessing`.

The basics of dask are:

* Breaking up large compute / I/O jobs into multiple chunks
* Parallelization of work acorss

The distributed scheduler is the recommended engine even on single machines.

Ensure you add the `distributed` extra:

`poetry add -E distributed dask`

"""
from typing import List, Optional

import os
import time

from dask import delayed
from dask.delayed import Delayed
from dask.distributed import Client, LocalCluster
from distributed.client import Future

CLIENT = Client(LocalCluster(n_workers=os.cpu_count(), threads_per_worker=None))


# You can use @delayed as an annotation or call delayed(inc) without the need to
# annotate the function.
@delayed
def inc(x: int) -> Delayed:
    time.sleep(1.0)
    return x + 1


@delayed
def add(x: int, y: int) -> Delayed:
    time.sleep(1.0)
    return x + y


def test_delayed() -> None:
    x = inc(1)
    y = inc(2)
    z = add(x, y)
    assert all(type(elt) == Delayed for elt in [x, y, z])
    # creates a png of the delayed graph at /tmp/mydask.png
    z.visualize(filename="/tmp/mydask.png")

    # run computations on the cluster
    assert x.compute() == 2
    assert y.compute() == 3
    assert z.compute() == 5


def test_for_loop() -> None:
    data = [1, 2, 3, 4, 5, 6, 7, 8]

    results: List[Delayed] = []

    # with get_client() as c:
    for x in data:
        results.append(inc(x))
    total = delayed(sum(results))
    assert total.compute() == 44


def test_map_gather() -> None:
    """When submitting a task, a Future is returned. In general, we want to
    *avoid* bringing the back from the cluster whenever possible. We want to
    operate on futures remotely by chaining them together as much as possible
    until the final results are truly needed."""

    def square(x: int) -> int:
        return x * x

    data = [1, 2, 3]
    # map / submit return future objects.
    #
    # submit a list of function calls to the scheduler
    squares: List[Future] = CLIENT.map(square, data)  # returns a list[Future]

    # submit a single function call to the scheduler
    total: Future = CLIENT.submit(sum, squares)
    # gather results to your local machine with with Future.result for a
    # single future, or Client.gather for many futures at once

    assert total.result() == 14
    assert CLIENT.gather(squares) == [1, 4, 9]
