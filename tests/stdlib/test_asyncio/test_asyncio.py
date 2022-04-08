"""asyncio provides support for coroutines and tasks.

Coroutines are subroutines that can be entered, exited, and resumed at many
different points (similar to generators). All coroutines are run independently
on the same run loop. asyncio is "cooperative multitasking". Couroutines must
yield by `await`ing on another coroutine.

Tasks are used to control coroutine execution. Tasks can be used to run multiple
coroutines concurrently.

pytest-asyncio:
     https://pypi.org/project/pytest-asyncio/

Key asyncio concepts:

* coroutine
* event loop
* executor: for running blocking code
* queue: for sending data into a coroutine


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

* Difficult to reason about. * All shared memory, even seemingly benign
  assigment statements, must be locked. `a = a + 1` is *not* deterministic.

* Resource intensive and inefficient. Each thread requires 8MB (default) stack
  space. The OS will context switch between all threads - regardless if the
  thread has work to do or not.


Chapter 3: Asyncio Walk-Through

Much of the asyncio library is targeted at framework designers, not end-user
developers.

The work for end-user developers:

* Create and manage (i.e., close) the event loop.
* Create and schedule tasks to run on the event loop.



"""

import asyncio
import pytest


async def say_hi(name: str) -> str:
    await asyncio.sleep(0.1)
    return f"hi {name}"


@pytest.mark.asyncio
async def test_coro_type() -> None:
    # A cororutine is an instance of a coroutine function. It is *not* the
    # function itself.
    assert asyncio.iscoroutinefunction(say_hi)
    assert not asyncio.iscoroutine(say_hi)
    assert asyncio.iscoroutine(say_hi("damon"))

    coro = say_hi("damon")
    print(coro.send(None))
    # with pytest.raises(StopIteration) as si:
    #     print(si)


@pytest.mark.asyncio
async def test_simple() -> None:
    assert await say_hi("damon") == "hi damon"


@pytest.mark.asyncio
async def test_task() -> None:
    t = asyncio.create_task(say_hi("damon"))
    assert await t == "hi damon"
