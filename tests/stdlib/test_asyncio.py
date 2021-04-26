"""asyncio provides support for coroutines and tasks.

Coroutines are subroutines that can be entered, exited, and resumed at many
different points.

Tasks are used to control coroutine execution. Tasks can be used to run multiple
coroutines concurrently.

"""

import asyncio
import pytest

async def say_hi(name: str) -> str:
    await asyncio.sleep(0.1)
    return f"hi {name}"

class TestAsyncIO:

    @pytest.mark.asyncio
    async def test_simple(self) -> None:
        assert await say_hi("damon") == "hi damon"

    @pytest.mark.asyncio
    async def test_task(self) -> None:
        t = asyncio.create_task(say_hi("damon"))
        assert await t == "hi damon"

