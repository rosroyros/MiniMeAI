#!/usr/bin/env python3
"""
Test script for WhatsApp integration.
This script tests the WhatsApp processor module.
"""

import json
import os
import sys
import time
from datetime import datetime

# Add the parent directory to the path to import WhatsAppProcessor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whatsapp.whatsapp_processor import WhatsAppProcessor

def main():
    """Main test function."""
    print("Testing WhatsApp Processor...")
    
    # Create a WhatsApp processor
    processor = WhatsAppProcessor(data_path="test_processed_messages.json")
    
    # Create a test message
    test_message = {
        "source": "whatsapp",
        "sourceId": f"test-{int(time.time())}",
        "from": "12345678901@c.us",
        "body": "This is a test WhatsApp message",
        "hasMedia": False,
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "sender": "Test User",
            "isGroup": False,
            "isForwarded": False,
            "type": "chat"
        }
    }
    
    # Process the test message
    processed = processor.process_message(test_message)
    
    # Print the processed message
    print("\nProcessed Message:")
    print(json.dumps(processed, indent=2))
    
    # Test batch processing
    test_messages = [
        test_message,
        {
            "source": "whatsapp",
            "sourceId": f"test-batch-{int(time.time())}",
            "from": "9876543210@c.us",
            "body": "This is another test WhatsApp message",
            "hasMedia": False,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "sender": "Another Test User",
                "isGroup": False,
                "isForwarded": False,
                "type": "chat"
            }
        }
    ]
    
    # Process the batch
    processed_batch = processor.process_batch(test_messages)
    
    # Print the number of processed messages
    print(f"\nProcessed {len(processed_batch)} messages in batch")
    
    print("\nWhatsApp Processor Test Completed Successfully!")

if __name__ == "__main__":
    main() 