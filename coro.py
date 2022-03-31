from typing import Any
import asyncio
import inspect
import time


async def run_for(delay: int) -> None:
    try:
        print(f"run_for starting w/ delay: {delay}")
        for i in range(delay + 1):
            print(f"run_for sleeping {i} of {delay}")
            await asyncio.sleep(1.0)
    except Exception as ex:
        print(f"run_for received exception: {str(ex)}")


def run_for_blocking(delay: int) -> None:
    try:
        print(f"run_for_blocking starting w/ delay: {delay}")
        for i in range(delay + 1):
            print(f"run_for_blocking sleeping {i} of {delay}")
            time.sleep(1.0)
    except Exception as ex:
        print(f"run_for_blocking received exception: {str(ex)}")


async def spawn(delay: int) -> None:
    print(f"spawning w/ delay: {delay}")
    # in an async function, this is how you get a pointer to the loop
    loop = asyncio.get_running_loop()
    # creates a task and adds it to the the run_loop. this "starts" the task.
    loop.create_task(run_for(delay))
    print("spawn returning")


async def say_hello(name: str) -> str:
    await asyncio.sleep(0.1)
    return f"{time.ctime()}: hello, {name}"


def echo(x: Any) -> None:
    print(x)


def run():
    # run() is a convenience function for creating and managing a asyncio event
    # loop. It will always create a new event loop (and fail if an event loop
    # already exists).

    greeting = asyncio.run(say_hello("damon"))
    print(greeting)

    loop = asyncio.get_running_loop()
    loop.create_task(echo("hello"))


def run_with_loop():
    # The function itself is still a function.
    inspect.isfunction(say_hello)
    # Determine if a function is a coroutine function (both are equal)
    assert inspect.iscoroutinefunction(say_hello)
    assert asyncio.iscoroutinefunction(say_hello)

    loop = asyncio.get_event_loop()

    # Create a few tasks
    task_async = loop.create_task(run_for(5))

    # NOTE: The blocking task finishes before the async task. If the executor is
    # running when we attempt to close the loop, bad things happen. We need to
    # manually wait for the executor to finish before closing the loop.

    task_sync = loop.run_in_executor(None, run_for_blocking, 2)
    loop.create_task(say_hello("damon"))

    # Starts the event loop. Until this point, *nothing* has started.
    loop.run_until_complete(task_async)

    # Cancel all tasks. Note the future running in the executor is *not*
    # a task and will not be returned in all_tasks. You'd have to manually
    # wait for the executor to finish.
    pending = asyncio.all_tasks(loop=loop)
    for task in pending:
        if task.done():
            print(f"task is done: {task.get_name()}")
        elif task.cancelled():
            print(f"task was cancelled: {task.get_name()}")
        else:
            print(f"cancelling: {task.get_name()}")
            task.cancel()

    group = asyncio.gather(*pending, return_exceptions=True)
    loop.run_until_complete(group)
    loop.close()


print(f"__name__ == {__name__}")

if __name__ == "__main__":
    run_with_loop()
