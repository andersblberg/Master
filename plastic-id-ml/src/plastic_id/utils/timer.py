from contextlib import ContextDecorator
from time import perf_counter
from rich.console import Console

console = Console()


class timed(ContextDecorator):
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, *exc):
        duration = perf_counter() - self.start
        console.log(f"[bold cyan]{self.label}[/] took {duration:.3f}s")
        return False
