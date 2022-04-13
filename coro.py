from typing import Any
import asyncio
import inspect

import logging
import time
import threading

FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def log(val: str) -> None:
    # logger.info(f"{time.ctime()}: {val}")
    logger.info(val)


async def run_for(delay: int) -> None:
    try:
        log(
            f"run_for starting on thread {threading.current_thread().getName()} w/ delay: {delay}"
        )
        for i in range(delay):
            log(f"run_for sleeping {i + 1} of {delay}")
            await asyncio.sleep(1.0)
        log(f"run_for complete")
    except Exception as ex:
        log(f"run_for received exception: {str(ex)}")


def run_for_blocking(delay: int) -> None:
    try:
        threading.current_thread().getName()
        log(
            f"run_for_blocking starting on thread {threading.current_thread().getName()} w/ delay: {delay}"
        )
        for i in range(delay):
            log(f"run_for_blocking sleeping {i + 1} of {delay}")
            time.sleep(1.0)
        log(f"run_for_blocking complete")
    except Exception as ex:
        log(f"run_for_blocking received exception: {str(ex)}")


async def spawn(delay: int) -> None:
    log(f"spawning w/ delay: {delay}")
    # in an async function, this is how you get the run loop pointer
    loop = asyncio.get_running_loop()
    # creates a task and adds it to the the run_loop. this "starts" the task
    loop.create_task(run_for(delay))
    log("spawn returning")


async def say_hello(name: str) -> str:
    """An async, quick function that returns a value"""
    await asyncio.sleep(0.1)
    return f"{time.ctime()}: hello, {name}"


def echo(x: Any) -> str:
    return f"exhoing {x}"


def run():
    # run() is a convenience function for creating and managing a asyncio event
    # loop. It will always create (and terminate) a new event loop (and fail if
    # an event loop already exists).
    #
    # This function always creates a new event loop and closes it at the end. It
    # should be used as a main entry point for asyncio programs, and should
    # ideally only be called once.
    #
    # The top level of most code probably use this method.
    log(asyncio.run(say_hello("damon")))


def run_with_loop():
    # The function itself is still a function.
    inspect.isfunction(say_hello)
    # Determine if a function is a coroutine function (both are equal)
    assert inspect.iscoroutinefunction(say_hello)
    assert asyncio.iscoroutinefunction(say_hello)

    # You need a loop instance before you can run any coroutines. Anywhere you
    # call `get_event_loop`, you'll get he same loop instance (assuming you're
    # using a single thread). If you're inside an async function, you should
    # call `asyncio.get_running_loop()` instead.
    loop = asyncio.get_event_loop()

    # Create a task and schedule it on the run loop.
    task_async = loop.create_task(run_for(5))

    # Executors run synchronous (blocking) code in asyncio in a separate thread.
    #
    # NOTE: The blocking task finishes before the async task. If the executor is
    # running when we attempt to close the loop, bad things happen. We need to
    # manually wait for the executor to finish before closing the loop.
    #
    future_sync = loop.run_in_executor(None, run_for_blocking, 2)

    # def on_cancelled(fut: asyncio.Future):
    #     log(f"future: {fut}")
    #     log(f"future cancelled: {fut.cancelled()}")
    #     log(f"future done: {fut.done()}")
    #     log(f"future exc: {fut.exception()}")

    # future_sync.add_done_callback(on_cancelled)
    # assert future_sync.cancel("you're done")

    loop.create_task(say_hello("damon"))

    # Starts the event loop. Until this point, *nothing* has started.
    loop.run_until_complete(task_async)

    # Cancel all tasks. Note the future running in the executor is *not*
    # a task and will not be returned in all_tasks. You'd have to manually
    # wait for the executor to finish.
    pending = asyncio.all_tasks(loop=loop)
    for task in pending:
        if task.done():
            log(f"task is done: {task.get_name()}")
        elif task.cancelled():
            log(f"task was cancelled: {task.get_name()}")
        else:
            log(f"cancelling: {task.get_name()}")
            task.cancel()

    group = asyncio.gather(*pending, return_exceptions=True)
    loop.run_until_complete(group)

    # Scan for any exceptions returned from the group tasks
    for res in group.result():
        assert not isinstance(
            res, Exception
        ), f"Unexpected exception was thrown in a task: {str(res)}"

    # A closed loop cannot be restarted.
    loop.close()


log(f"__name__ == {__name__}")

if __name__ == "__main__":
    # run()
    run_with_loop()
