# Python Concurrency

David Beazley - [Understanding the Python GIL](https://www.youtube.com/watch?v=Obt-vMVdM8s)

Python concurrency from the ground up.


## GIL

* The GIL was implemented to simplify the runtime (memory management)
* The GIL makes your code cooperatively multitasking.
* The runtime context switches between threads. Threads are pushed onto a wait
  queue. When a thread is popped off the wait queue, it's handed to the OS.

NOTE: This was pre-Python 3.2, before a new GIL was implemented:

* Single CPU: Threads alternate between longer time period (less thrashing /
  context switching)

* Multiple CPUs: Shorter time periods, constant thrashing (CPU cores fight). The
  OS is trying to start a thread per process, which the GIL will not allow.

* Do I/O bound threads cause thrashing? YES! The cores fight with each other
  when the core count is higher.

* tl;dr: The old GIL didn't account for multi-core computers well and caused a
  lot of thrashing.


The new GIL (Python 3.2)

* Rather than using ticks, threads wait for a longer duration and signal to each
  other. This reduces churn.

* CPU and I/O bound tasks are MUCH slower in the new GIL.

* tl;dr: The new GIL needs priority and the ability to preempt (similar to OS
  threads).

What's the preferred way to do multithreading in python?
  * `asyncio` and `multiprocessing`?

## Python Concurrency From the Ground Up (PyCon 2015)

Python 3.5

The GIL prioritizes CPU bound threads. That is *not* how operating system works
(the OS gives priority to short running tasks).

Should you use threads?
* Threads were solving the problems of blocking.

Generators `yield`, which allows a function to pause / resume. This allows you
to setup cooperative multi-tasking (a.k.a., coroutines).

Once you go to asyncio, you have to use asyncio all the way down (to prevent
blocking).


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




