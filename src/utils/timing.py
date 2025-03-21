import time
import logging
import functools
from typing import Any, Callable, TypeVar

# Type variable for decorator
F = TypeVar('F', bound=Callable[..., Any])

class Timer:
    """Utility class for timing code execution."""
    
    def __init__(self, name="Operation"):
        """Initialize timer with operation name."""
        self.name = name
        
    def __enter__(self):
        """Start timer when entering context."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        """Log elapsed time when exiting context."""
        elapsed_time = time.time() - self.start_time
        logging.info(f"{self.name} took {elapsed_time:.4f} seconds")

def timed(func: F) -> F:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logging.info(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper 