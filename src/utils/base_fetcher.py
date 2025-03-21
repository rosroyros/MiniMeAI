#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base fetcher class that all data source fetchers should inherit from.
"""
import logging
import time
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime

# Import the new date utilities
from src.utils.date_utils import parse_timestamp, get_safe_timestamp

logger = logging.getLogger(__name__)

class BaseFetcher(ABC):
    """Base class for all data fetchers that ensures consistent source type handling.
    
    This class provides a common framework for all data source fetchers to ensure
    consistent metadata handling, especially for source types and timestamps.
    
    All specialized fetchers (email, whatsapp, web, etc.) should inherit from this base class.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the fetcher with configuration.
        
        Args:
            config: Configuration dictionary for the fetcher
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def enrich_item(self, item: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """Add common metadata to an item.
        
        This ensures all items have consistent metadata regardless of source.
        
        Args:
            item: The item to enrich
            source_type: The type of data source (e.g., 'email', 'document')
            
        Returns:
            The enriched item
        """
        # Set source type if not already set
        if "source_type" not in item:
            item["source_type"] = source_type
            
        # Ensure timestamp is present
        if "timestamp" not in item or not item["timestamp"]:
            # Try to extract from date field
            if "date" in item:
                # Use our new date utilities
                item["timestamp"] = get_safe_timestamp(item["date"], default_strategy="now")
            else:
                # Use current time as fallback
                item["timestamp"] = int(time.time())
            
        return item
    
    def get_timestamp_from_date(self, date_str: str) -> Optional[int]:
        """Convert date string to timestamp.
        
        This is a common utility method for all fetchers to ensure consistent
        timestamp handling across different data sources.
        
        Args:
            date_str: Date string in any recognized format
            
        Returns:
            Unix timestamp as integer, or None if parsing fails
        """
        # Use our new date utilities module
        return parse_timestamp(date_str)
    
    @abstractmethod
    def fetch_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch data from the source.
        
        This abstract method must be implemented by all derived classes.
        
        Args:
            limit: Maximum number of items to fetch
            
        Returns:
            List of items fetched from the source with proper metadata
        """
        pass 