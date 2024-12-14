"""asyncio provides support for coroutines and tasks.

Coroutines are subroutines that can be entered, exited, and resumed at many
different points (similar to generators). All coroutines are run independently
on the same run loop. asyncio is a form of "cooperative multitasking".
Couroutines must yield by `await`ing on another coroutine.

Tasks are used to control coroutine execution. Tasks can be used to run multiple
coroutines concurrently.

Key asyncio concepts:

* coroutines: functions which can be paused (via await) and resumed.
* event loop: manages running coroutines
* executor: for running blocking code on the event loop using threads or
  processes
* queue: for sending data between coroutines (similar to threading.Queue)

---

Asyncio in Python (Book)

Chapter 1: Introducing asyncio

I/O is 100 or 1000 times slower than CPU. Spawning threads for I/O work is
inefficient and introduce race conditions. Asyncio uses a single thread (with a
single run loop) to manage tasks. Each task must coopoerate and yield control
back to the main run loop and not hog execution time (i.e., block while
waiting).

Asyncio eliminates certain type of concurrency issues. For example, it's clear
where tasks are suspended / resumed. However there are still issues that need to
be solved with asyncio:

* How will you communicate with a resources (like a database) that allows for
  only a few connections?

* How do you terminate the run loop when receiving a signal?

* How do you deal with blocking code?

The GIL effectively pins execution of all threads to a single CPU, preventing
multicore parallelism.


Chapter 2: The Truth About Threads

Benefits of threading:

* Shared memory. Threads can access shared memory, avoiding moving data between
  processes.

Dawbacks of threading:

* Difficult to reason about.
* All shared memory, even seemingly benign assigment statements, must be locked.
  `a = a + 1` is *not* deterministic.

* Resource intensive and inefficient. Each thread requires 8MB (default) stack
  space. The OS will context switch between all threads - regardless if the
  thread has work to do or not.


Chapter 3: Asyncio Walk-Through

Much of the asyncio library is targeted at framework designers, not end-user
developers.

The work for end-user developers:

* Create and manage (i.e., close) the event loop.
* Create and schedule tasks to run on the event loop.
* Cancel and teardown the event loop, gracefully exiting.


---

To run asyncio tests w/ pytest, pip install pytest-asyncio:

pytest-asyncio:
     https://pypi.org/project/pytest-asyncio/

@pytest.mark.asyncio will execute the test within a default run loop.

"""

import asyncio
import contextlib
import inspect
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import pytest

logger = logging.getLogger(__name__)


async def say_hi(name: str) -> str:
    """An example coroutine function.

    A coroutine function is a "normal" function with the ability to suspend /
    resume. A coroutine is suspended when an `await` is encountered. When
    suspended, control is returned to the run loop.
    """
    return f"hi {name}"


async def raise_exc(msg: str) -> None:
    """Exceptions thrown from a coroutine are raised in the function `await`ing
    the result."""
    raise Exception(msg)


async def long_running(seconds: float) -> str:
    try:
        while True:
            await asyncio.sleep(seconds)
    except asyncio.CancelledError:
        return "cancelled"


@pytest.mark.asyncio
async def test_coro_manual_invocation() -> None:
    #
    # `async` functions are still functions.
    #
    assert inspect.isfunction(say_hi)
    assert inspect.iscoroutinefunction(say_hi)
    assert asyncio.iscoroutinefunction(say_hi)

    #
    # A cororutine is an instance of a coroutine function. It is *not* the
    # function itself.
    #
    assert not asyncio.iscoroutine(say_hi)

    #
    # Evaluating (invoking) the function will create an instance of a coroutine.
    #
    # Evaluating does *not* execute the coroutine, it simply creates the
    # coroutine for later execution.
    #
    coro = say_hi("damon")
    assert asyncio.iscoroutine(coro)

    #
    # Manually resume (start) the coroutine by sending it `None`. Typically this
    # is done by the event loop.
    #
    # When a coroutine returns, a special `StopIteration` exception is raised,
    # which contains the coroutine result.
    #
    value: Optional[str] = None
    try:
        coro.send(None)  # Start the coroutine.
        assert False, "StopIteration should have been raised - coro is completed."
    except StopIteration as e:
        value = e.value

    assert value == "hi damon"


@pytest.mark.asyncio
async def test_coro_manual_exception() -> None:
    """Shows manually handling an exception raised by a coroutine."""

    coro = raise_exc("boom")
    value: Optional[Exception] = None
    try:
        coro.send(None)
    except Exception as e:
        value = e

    assert str(value) == "boom"


@pytest.mark.asyncio
async def test_coro_inject_exception() -> None:
    """Shows injecting an exception into the coroutine.

    When a coroutine calls `await`, the awaited function may throw an exception,
    which the calling coroutine will need to handle.

    Here, we manually simulate an exception being returned into coro. This
    raises the exception in the coroutine at the "await" point.
    """
    coro = long_running(1000.0)
    value: Optional[Exception] = None
    try:
        coro.send(None)
        coro.throw(Exception("boom"))
    except Exception as e:
        value = e

    assert str(value) == "boom"


@pytest.mark.asyncio
async def test_coro_cancellation() -> None:
    coro = long_running(1000.0)
    value: Optional[str] = None
    try:
        coro.send(None)
        # Simulates task cancellation, which throws a "CancelledError" into the coroutine.
        coro.throw(asyncio.CancelledError)
    except StopIteration as e:
        value = e.value

    assert value == "cancelled"


@pytest.mark.asyncio
async def test_simple() -> None:
    #
    # You don't typically drive the coroutine manually as we did above.
    #
    # Rather, you simply `await` the coroutine function. The event loop will
    # handle advancing the coroutine and handling `StopIteration` for you.
    #
    assert await say_hi("damon") == "hi damon"
    with pytest.raises(Exception) as e:
        await raise_exc("boom")

    assert str(e.value) == "boom"


@pytest.mark.asyncio
async def test_tasks() -> None:
    """
    Tasks are coroutine instances running on the event loop.

    Tasks can be used to start multiple coroutines concurrently.
    """

    event_loop = asyncio.get_running_loop()

    async def sleep(seconds: float):
        await asyncio.sleep(seconds)

    start = time.time()

    # Start both tasks
    t1 = event_loop.create_task(sleep(0.1))
    t2 = event_loop.create_task(sleep(0.2))
    await t2
    await t1

    end = time.time()
    elapsed = end - start

    # Since both tasks were running concurrently, their total elapsed time will
    # be around the longest task.
    assert math.isclose(elapsed, 0.2, abs_tol=0.05)


@pytest.mark.asyncio
async def test_executor() -> None:
    """
    Functions which are not async must be wrapped in an executor to use with
    asyncio. For example, an I/O based function that doesn't have an async
    version would need to be ran in an executor.

    The default executor is a ThreadPoolExecutor, which runs in a background
    thread.

    This test verifies the executor truly runs tasks in a non-main thread.
    """

    def echo(value: str) -> str:
        """A non-async function which can't be awaited."""
        assert threading.current_thread() is not threading.main_thread()
        return value

    val = "damon"
    f: asyncio.Future = asyncio.get_running_loop().run_in_executor(None, echo, val)
    await asyncio.wait_for(f, timeout=1.0)

    assert f.done()
    assert f.result() == val


@pytest.mark.asyncio
async def test_async_context_manager() -> None:
    """@asynccontextmanager is an decorator for creating async context managers
    without having to define __aenter__ and __aexit__

    https://docs.python.org/3/library/contextlib.html#contextlib.asynccontextmanager
    """

    val = "data"

    @contextlib.asynccontextmanager
    async def download_data(data: str):
        await asyncio.sleep(0.01)  # assume a long running I/O operation
        yield data

    async with download_data(val) as d:
        assert d == val


@pytest.mark.asyncio
async def test_async_generators() -> None:
    """Async iterators provide support for `async for` - the ability to iterate
    over a sequence which has async operations.

    Async generators return async iterator instances with each invocation.
    """

    async def async_gen(val: int):
        for i in range(val):
            await asyncio.sleep(0)  # do something async
            yield i

    expected = [0, 1, 2, 3, 4]
    # async list comp works as you'd expect
    assert [i async for i in async_gen(5)] == expected

    # As does async for
    lst: List[int] = []
    async for i in async_gen(5):
        lst.append(i)
    assert lst == expected


def test_asyncio_lifecycle() -> None:
    """You typically won't have to deal with the asyncio lifecycle, as it is
    taken care of by an underlying framework you use.

    This test shows how to startup and shutdown an asyncio runloop.
    """

    blocking_completed = False
    non_blocking_completed = False

    async def run_for(delay: int) -> None:
        for i in range(delay):
            logger.info(f"run_for sleeping {i + 1} of {delay}")
            await asyncio.sleep(1.0)
        logger.info("run_for complete")
        nonlocal non_blocking_completed
        non_blocking_completed = True

    def run_for_blocking(delay: int) -> None:
        for i in range(delay):
            logger.info(f"run_for_blocking {i + 1} of {delay}")
            time.sleep(1.0)
        logger.info("run_for_blocking complete")
        nonlocal blocking_completed
        blocking_completed = True

    async def spawn(delay: int, blocking: bool) -> None:
        """Spawn another long running task."""
        loop = asyncio.get_event_loop()
        if blocking:
            loop.run_in_executor(run_for_blocking(delay))
        else:
            loop.create_task(run_for(delay))

    # You need a loop instance before you can run any coroutines. Anywhere you
    # call `get_event_loop`, you'll get the same loop instance (assuming you're
    # using a single thread). If you're inside an async function, you should
    # call `asyncio.get_running_loop()` instead.
    loop = asyncio.get_event_loop()

    #
    # Create our own executor so we can manually control shutdown.
    #
    # NOTE: asyncio.run() will properly wait for the default executor to
    # shutdown. You don't need to create your own.
    #
    executor = ThreadPoolExecutor()
    loop.set_default_executor(executor)

    #
    # Create async and blocking tasks
    #
    t = loop.create_task(spawn(1, False))
    t2 = loop.create_task(spawn(2, True))

    #
    # Starts the event loop. Until this point, *nothing* was started.
    #
    group = asyncio.gather(t, t2, return_exceptions=True)
    loop.run_until_complete(group)
    assert t.done()
    assert not t.cancelled()
    assert t2.done()
    assert not t2.cancelled()

    #
    # Start the shutdown process.
    #
    # We now have running async and blocking tasks. Both need to be terminated.
    #
    # The non-blocking coro is running on the event loop. The blocking coro is
    # running on the executor, it's *not* a task.
    #
    # Let's start by shutting down the non-blocking task first, then the
    # executor.
    #
    # Gather and cancel all tasks. Note the future running in the executor is
    # *not* a task and will not be returned in all_tasks. You'd have to manually
    # wait for the executor to finish.
    pending = asyncio.all_tasks(loop=loop)
    assert len(pending) == 1

    #
    # return_exceptions=False is the default. In this mode, one raised exception
    # from any tasks in the group will stop the loop before all tasks are
    # completed.
    #
    # raise_exceptions=True will force all tasks in the group to be completed.
    # Exceptions will be returned as return values.
    #
    group = asyncio.gather(*pending, return_exceptions=True)
    loop.run_until_complete(group)
    assert non_blocking_completed

    pending = asyncio.all_tasks(loop=loop)
    assert len(pending) == 0

    #
    # Now close the executor.
    #
    executor.shutdown(wait=True, cancel_futures=False)
    assert blocking_completed

    #
    # A closed loop cannot be restarted.
    #
    loop.close()


def test_asyncio_run() -> None:
    """asyncio.run() cleanly handles shutdown of both tasks running on the event
    loop, as well as futures running in executors.

    asyncio.run() is typically how you'd start the main run loop for your
    application.
    """

    blocking_complete = False
    non_blocking_complete = False

    def run_for_blocking(delay: int) -> None:
        nonlocal blocking_complete
        for i in range(delay):
            logger.info("run_for_blocking sleeping...")
            time.sleep(1.0)
        blocking_complete = True

    async def main() -> None:
        #
        # Executors run synchronous (blocking) code in asyncio in a separate thread.
        #
        # Here, we start a long running task on a separate thread.
        #
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, run_for_blocking, 1)
        nonlocal non_blocking_complete
        non_blocking_complete = True

    #
    # asyncio.run() will wait for all pending tasks and executor tasks spawned
    # from the main() coroutine to finish before returning.
    #
    asyncio.run(main())

    assert non_blocking_complete
    assert blocking_complete
