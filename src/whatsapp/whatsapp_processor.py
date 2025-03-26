"""
WhatsApp message processor for MiniMeAI.
This module provides functionality for processing WhatsApp messages and preparing them
for storage in the vector database.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("whatsapp_processor")

class WhatsAppProcessor:
    """
    Process WhatsApp messages for MiniMeAI.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the WhatsApp processor.
        
        Args:
            data_path: Path to store processed messages
        """
        self.data_path = data_path or os.environ.get('PROCESSED_DATA_PATH', '/app/data/processed_messages.pickle')
        self.processed_cache = {}
        logger.info(f"WhatsApp processor initialized with data path: {self.data_path}")
        
    def process_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a WhatsApp message.
        
        Args:
            message_data: Raw WhatsApp message data
            
        Returns:
            Processed message ready for vector storage
        """
        source_id = message_data.get('sourceId')
        
        # Skip if we've already processed this message
        if source_id in self.processed_cache:
            logger.info(f"Message {source_id} already processed, skipping")
            return self.processed_cache[source_id]
        
        # Extract and normalize timestamp
        timestamp = message_data.get('timestamp')
        if timestamp:
            try:
                # Parse the ISO format timestamp
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                # Convert to a standardized format
                timestamp = dt.isoformat()
            except Exception as e:
                logger.warning(f"Error parsing timestamp: {e}")
                timestamp = datetime.now().isoformat()
        else:
            timestamp = datetime.now().isoformat()
        
        # Create a standardized document format
        processed_message = {
            "id": source_id,
            "source": "whatsapp",
            "source_id": source_id,
            "timestamp": timestamp,
            "content": message_data.get('body', ''),
            "sender": message_data.get('metadata', {}).get('sender', message_data.get('from', 'unknown')),
            "recipient": "me",  # Assuming messages received by the system
            "metadata": {
                "original": message_data
            }
        }
        
        # Cache the processed message
        if source_id:
            self.processed_cache[source_id] = processed_message
        
        logger.info(f"Processed WhatsApp message {source_id}")
        return processed_message
    
    def process_batch(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of WhatsApp messages.
        
        Args:
            messages: List of raw WhatsApp message data
            
        Returns:
            List of processed messages ready for vector storage
        """
        processed_messages = []
        
        for message in messages:
            try:
                processed = self.process_message(message)
                if processed:
                    processed_messages.append(processed)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
        
        logger.info(f"Processed {len(processed_messages)} WhatsApp messages")
        return processed_messages 