#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the new date_utils.py module.
"""
import sys
import logging
from datetime import datetime

# Add the project root to the Python path to allow imports from src/
sys.path.append('.')

# Import our new date utilities
from src.utils.date_utils import parse_timestamp, get_safe_timestamp, format_timestamp

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def test_timestamps():
    """Test various date formats with our timestamp utilities."""
    
    # Sample date strings in various formats
    test_dates = [
        # Email format (RFC 2822)
        "Mon, 25 Mar 2025 15:30:45 +0000",
        
        # ISO 8601 formats
        "2025-03-25T15:30:45+00:00",
        "2025-03-25T15:30:45.123+00:00",
        "2025-03-25T15:30:45",
        
        # Common formats
        "2025-03-25 15:30:45",
        "25/03/2025 15:30:45",
        "03/25/2025 15:30:45",
        "25.03.2025 15:30:45",
        
        # Just date
        "2025-03-25",
        
        # Edge cases
        "",                       # Empty string
        "Invalid date format",    # Invalid format
        "Jan 1, 2025",            # Non-standard format
    ]
    
    # Test datetime object and timestamp
    test_objects = [
        datetime.now(),           # Current datetime
        int(datetime.now().timestamp()),  # Current timestamp as int
        float(datetime.now().timestamp()),  # Current timestamp as float
        0,                        # Zero timestamp
        -1,                       # Negative timestamp
        9999999999,               # Very large timestamp
    ]
    
    print("\n===== Testing parse_timestamp() =====")
    for date_str in test_dates:
        timestamp = parse_timestamp(date_str)
        print(f"Date: '{date_str}'")
        print(f"  Timestamp: {timestamp}")
        if timestamp:
            print(f"  Human readable: {format_timestamp(timestamp)}")
        print()
    
    print("\n===== Testing with datetime objects and timestamps =====")
    for obj in test_objects:
        timestamp = parse_timestamp(obj)
        print(f"Input: {obj} (type: {type(obj).__name__})")
        print(f"  Timestamp: {timestamp}")
        if timestamp:
            print(f"  Human readable: {format_timestamp(timestamp)}")
        print()
    
    print("\n===== Testing get_safe_timestamp() with different strategies =====")
    for strategy in ["none", "now", "past"]:
        print(f"\nStrategy: {strategy}")
        
        # Test with invalid date
        invalid_date = "this is not a date"
        timestamp = get_safe_timestamp(invalid_date, strategy)
        print(f"Invalid date with '{strategy}' strategy:")
        print(f"  Timestamp: {timestamp}")
        print(f"  Human readable: {format_timestamp(timestamp)}")
    
    # Test format_timestamp with various formats
    print("\n===== Testing format_timestamp() with different formats =====")
    now_ts = int(datetime.now().timestamp())
    formats = [
        "%Y-%m-%d %H:%M:%S",  # Default
        "%Y-%m-%d",           # Just date
        "%b %d, %Y",          # Month name
        "%d/%m/%Y %H:%M",     # European
        "%m/%d/%Y %I:%M %p",  # US with AM/PM
    ]
    
    for fmt in formats:
        formatted = format_timestamp(now_ts, fmt)
        print(f"Format: '{fmt}'")
        print(f"  Result: {formatted}")

if __name__ == "__main__":
    test_timestamps() 