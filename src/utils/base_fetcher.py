import logging
import time
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

class BaseFetcher(ABC):
    """Base class for all data fetchers that ensures consistent source type handling.
    
    This class provides a common framework for all data source fetchers to ensure
    consistent metadata handling, especially for source types and timestamps.
    
    All specialized fetchers (email, whatsapp, web, etc.) should inherit from this base class.
    """
    
    def __init__(self, source_type: str, logger: Optional[logging.Logger] = None):
        """Initialize the base fetcher with a specific source type.
        
        Args:
            source_type: A string identifier for the source type (e.g., "email", "whatsapp")
            logger: Optional logger instance. If not provided, a new logger will be created.
        """
        self.source_type = source_type
        self.logger = logger or logging.getLogger(f"{source_type}_fetcher")
    
    def enrich_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all items have proper metadata including source type.
        
        This method adds or updates the source_type field in the item metadata
        and ensures other common fields are present.
        
        Args:
            item: The item dictionary to enrich with metadata
            
        Returns:
            The enriched item with guaranteed source_type field
        """
        # Always set the source type
        item["source_type"] = self.source_type
        
        # Ensure timestamp exists if possible
        if "timestamp" not in item and "date" in item:
            item["timestamp"] = self.get_timestamp_from_date(item["date"])
        elif "timestamp" not in item:
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
        if not date_str:
            return None
            
        # Common date formats to try
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",  # RFC 2822 format
            "%Y-%m-%dT%H:%M:%S%z",       # ISO 8601
            "%Y-%m-%d %H:%M:%S",         # SQL-like
            "%d/%m/%Y %H:%M:%S",         # Common format
            "%d/%m/%Y %H:%M",            # Common format without seconds
            "%d.%m.%Y %H:%M:%S",         # European format
            "%d.%m.%Y %H:%M",            # European format without seconds
        ]
        
        # Try standard formats
        from datetime import datetime
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return int(dt.timestamp())
            except ValueError:
                continue
                
        # Try email-specific format
        try:
            import email.utils
            date_tuple = email.utils.parsedate_tz(date_str)
            if date_tuple:
                return int(email.utils.mktime_tz(date_tuple))
        except Exception:
            pass
            
        self.logger.warning(f"Could not parse date: {date_str}")
        return None
    
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