import time
import functools
import logging

# Configure logger
logger = logging.getLogger("minimeai.timing")

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
        logger.info(f"{self.name} took {elapsed_time:.4f} seconds")

def timed(func):
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper 