#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date and timestamp utilities for consistent handling across the codebase.
"""
import logging
import time
from datetime import datetime
from typing import Optional, Union
import email.utils

logger = logging.getLogger(__name__)

def parse_timestamp(date_input: Union[str, datetime, int, float, None]) -> Optional[int]:
    """
    Unified timestamp parser that handles various date/time input formats.
    
    Args:
        date_input: Can be:
            - String in various date formats
            - datetime object
            - existing timestamp (int/float)
            - None
            
    Returns:
        Unix timestamp as integer, or None if parsing fails
    """
    # Handle None case
    if date_input is None:
        return None
        
    # Already a timestamp number
    if isinstance(date_input, (int, float)):
        # Validate timestamp is reasonable (between 2000 and 2100)
        if 946684800 <= date_input <= 4102444800:  # 2000-01-01 to 2100-01-01
            return int(date_input)
        else:
            logger.warning(f"Timestamp value out of reasonable range: {date_input}")
            return None
            
    # Handle datetime object
    if isinstance(date_input, datetime):
        return int(date_input.timestamp())
        
    # String formats - must be string at this point
    if not isinstance(date_input, str):
        logger.warning(f"Unsupported date input type: {type(date_input)}")
        return None
        
    date_str = date_input.strip()
    if not date_str:
        return None
    
    # First try parsing with email.utils (best for email RFC 2822 format)
    try:
        date_tuple = email.utils.parsedate_tz(date_str)
        if date_tuple:
            return int(email.utils.mktime_tz(date_tuple))
    except Exception as e:
        logger.debug(f"email.utils parsing failed for '{date_str}': {e}")
    
    # Try common date formats
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",  # RFC 2822 format
        "%Y-%m-%dT%H:%M:%S%z",       # ISO 8601
        "%Y-%m-%dT%H:%M:%S.%f%z",    # ISO 8601 with microseconds
        "%Y-%m-%dT%H:%M:%S",         # ISO 8601 without timezone
        "%Y-%m-%dT%H:%M:%S.%f",      # ISO 8601 with microseconds, without timezone
        "%Y-%m-%d %H:%M:%S",         # SQL-like
        "%Y-%m-%d %H:%M:%S.%f",      # SQL-like with microseconds
        "%d/%m/%Y %H:%M:%S",         # Common format DD/MM/YYYY
        "%d/%m/%Y %H:%M",            # Common format without seconds
        "%m/%d/%Y %H:%M:%S",         # US format MM/DD/YYYY
        "%m/%d/%Y %H:%M",            # US format without seconds
        "%d.%m.%Y %H:%M:%S",         # European format
        "%d.%m.%Y %H:%M",            # European format without seconds
        "%Y-%m-%d",                  # Just date
        "%d/%m/%Y",                  # Just date DD/MM/YYYY
        "%m/%d/%Y",                  # Just date MM/DD/YYYY
        "%d.%m.%Y",                  # Just date with dots
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return int(dt.timestamp())
        except ValueError:
            continue
    
    logger.warning(f"Failed to parse date string: '{date_str}'")
    return None

def get_safe_timestamp(date_input: Union[str, datetime, int, float, None], 
                       default_strategy: str = "none") -> int:
    """
    Get a timestamp that will never return None. Uses fallback strategies for unparseable dates.
    
    Args:
        date_input: Any date input (string, datetime, timestamp)
        default_strategy: What to do if parsing fails:
            - "none": Return 0 (documents will sort at bottom)
            - "now": Return current time
            - "past": Return timestamp from 1 year ago (documents will sort at bottom)
            
    Returns:
        Unix timestamp as integer, never None
    """
    timestamp = parse_timestamp(date_input)
    
    if timestamp is not None:
        return timestamp
        
    # Handle default strategy
    if default_strategy == "now":
        return int(time.time())
    elif default_strategy == "past":
        return int(time.time()) - 31536000  # One year ago
    else:  # "none" or any other value
        return 0

def format_timestamp(timestamp: Optional[int], 
                    format_str: str = "%Y-%m-%d %H:%M:%S") -> Optional[str]:
    """
    Format a timestamp as a human-readable string.
    
    Args:
        timestamp: Unix timestamp
        format_str: Output format
        
    Returns:
        Formatted date string or None if timestamp is None/invalid
    """
    if timestamp is None or timestamp <= 0:
        return None
        
    try:
        return datetime.fromtimestamp(timestamp).strftime(format_str)
    except (ValueError, TypeError, OverflowError) as e:
        logger.warning(f"Invalid timestamp {timestamp}: {e}")
        return None 