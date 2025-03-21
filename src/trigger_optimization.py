#!/usr/bin/env python3
import os
import sys
import requests
import json
from datetime import datetime

def main():
    """Manually trigger full optimization of vector collections."""
    # Get parameters from environment or use defaults
    host = os.getenv("VECTOR_DB_HOST", "vector_db")
    port = os.getenv("VECTOR_DB_PORT", "8000")
    collection = os.getenv("VECTOR_COLLECTION_NAME", "emails")
    
    # URL for the optimize endpoint
    url = f"http://{host}:{port}/api/v1/collections/{collection}/optimize"
    
    print(f"\nTriggering full optimization for collection: {collection}")
    print(f"Vector DB URL: {url}\n")
    
    try:
        # Send request to trigger optimization
        start_time = datetime.now()
        print(f"Starting optimization at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("This may take several minutes for large collections...\n")
        
        response = requests.post(url, json={"target_dim": 128}, timeout=600)  # 10 minute timeout
        response.raise_for_status()
        result = response.json()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Format the output nicely
        print(f"Optimization result: {result['message']}")
        print(f"Completed in: {duration:.2f} seconds")
        
        # Check the status after optimization
        status_url = f"http://{host}:{port}/api/v1/collections/{collection}/status"
        status_response = requests.get(status_url, timeout=10)
        status = status_response.json()
        
        print(f"\nUpdated optimization status:")
        print(f"Optimized: {'Yes' if status['is_optimized'] else 'No'}")
        print(f"Optimization coverage: {status['optimization_coverage']}")
        print(f"Total vectors: {status['total_vectors']}")
        print(f"Optimized vectors: {status['optimized_vectors']}")
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to vector DB. Make sure the service is running.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main() 