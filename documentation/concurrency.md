# Python Concurrency

## Terminology

### Concurrency

* Multiple tasks running at the same time (potentially not executing exactly at the same time).
* On a single CPU machine, multiple tasks may be in flight, but only one will be executing at a time.

### Parallelism:

* Two tasks actually executing at the same time.
* Requires multiple CPUs.

### Parallel vs. Concurrent

> A system is said to be concurrent if it can support two or more actions in progress at the same time, but not running simultaneously.
* . A system is said to be parallel if it can support two or more actions executing simultaneously.

> An application can be parallel but not concurrent (multiprocess). An application can be concurrent but not parallel (threading, asyncio).


### Global Interpreter Lock (GIL)

Python cannot run multiple threads concurrently.

* For CPU bound operations, use multiple processes (parallelism).
* For I/O bound operations, use `threading` or `asyncio`.

### multiprocessing

Multiple processes. The only way to achieve true parallelism in Python is to spawn multiple processes.

### Threading

* Pre-emptive multitasking (OS managed)

### asyncio

* Cooperative multitasking (process managed)