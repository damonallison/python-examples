import yappi


def add(x: int, y: int) -> int:
    return x + y


def sub(x: int, y: int) -> int:
    return add(x, y) - add(x, y) + x - y


yappi.set_clock_type("wall")
yappi.start()

for _ in range(100):
    sub(2, 2)

yappi.stop()

yappi.get_func_stats().print_all()
yappi.get_thread_stats().print_all()
