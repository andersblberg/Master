import time
from functools import wraps

"""
Module: timer.py
Provides utilities to measure execution time of functions or code blocks.
"""  

def timeit(func):
    """
    Decorator that measures the execution time of the decorated function.
    The decorated function returns a tuple: (original_function_result, elapsed_time_seconds).

    Usage:
        @timeit
        def my_function(...):
            ...
        result, elapsed = my_function(...)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
    return wrapper


def measure_time(func, *args, **kwargs):
    """
    Measures the execution time of func(*args, **kwargs) without using a decorator.
    Returns a tuple: (original_function_result, elapsed_time_seconds).

    Usage:
        result, elapsed = measure_time(my_function, arg1, arg2, kwarg1=value)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

class Timer:
    """
    Context manager for timing a code block.

    Usage:
        with Timer("block name") as t:
            # code to measure
            ...
        print(f"Elapsed for {t.name}: {t.elapsed:.4f} seconds")
    """
    def __init__(self, name=None):
        self.name = name
        self.start = None
        self.end = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        # Do not suppress exceptions
        return False

if __name__ == "__main__":
    # Example usage:
    def test(n):
        s = 0
        for i in range(n): s += i
        return s

    # Using decorator
    @timeit
    def decorated_test(n):
        return test(n)

    (res1, t1) = decorated_test(1000000)
    print(f"Decorated test result: {res1}, elapsed: {t1:.4f} s")

    # Using measure_time
    res2, t2 = measure_time(test, 1000000)
    print(f"Measured test result: {res2}, elapsed: {t2:.4f} s")

    # Using context manager
    with Timer("block1") as t:
        _ = test(1000000)
    print(f"Block1 ({t.name}) elapsed: {t.elapsed:.4f} s")
