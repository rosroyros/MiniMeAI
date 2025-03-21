import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # File handler for timing logs
    timing_handler = RotatingFileHandler(
        os.path.join(log_dir, "timing.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    timing_handler.setLevel(logging.INFO)
    timing_format = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    timing_handler.setFormatter(timing_format)
    
    # Add handler to timing logger
    timing_logger = logging.getLogger("minimeai.timing")
    timing_logger.addHandler(timing_handler)
    
    return root_logger 