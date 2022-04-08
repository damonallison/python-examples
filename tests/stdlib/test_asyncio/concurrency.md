# Python Concurrency

## Terminology

### Concurrency

* A `concurrent` system supports two or more actions in progress at the same
  time, but not running simultaneously.

* On a single CPU machine, multiple tasks may be in flight, but only one will be
  executing at a time.

### Parallelism

* A system is said to be parallel if it can support two or more actions
  executing simultaneously (requires multiple CPUs)

### Parallel vs. Concurrent

* An application can be parallel but not concurrent (multiprocess).

* An application can be concurrent but not parallel (threading, asyncio).

### Global Interpreter Lock (GIL)

Python cannot run multiple threads concurrently.

* For CPU bound operations, use multiple processes (parallelism).
* For I/O bound operations, use `threading` or `asyncio`.

### multiprocessing

The only way to achieve true parallelism in Python is to spawn multiple
processes.

### Threading

* Pre-emptive multitasking (OS managed)

### asyncio

* Cooperative multitasking (process managed). The code must `await` to return
  control back to the executor.

* The benefit of `asyncio` is you don’t have to worry about locking. You can
  guarantee only one task will be running and will not be preemptively
  cancelled.

* Coroutines are functions which have multiple exit points (async def).

* `task` is a wrapper around a coroutine.




