"""asyncio provides support for coroutines and tasks.

Coroutines are subroutines that can be entered, exited, and resumed at many
different points. All coroutines are run independently on the same run loop.

Tasks are used to control coroutine execution. Tasks can be used to run multiple
coroutines concurrently.

pytest-asyncio:
     https://pypi.org/project/pytest-asyncio/

asyncio is "cooperative multitasking". Couroutines must yield by `await`ing on
another coroutine.

Key asyncio concepts:

* coroutine
* event loop
* executor: for running blocking code
* queue: for sending data into a coroutine

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
