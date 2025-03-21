#!/usr/bin/env python3
import os
import sys
import requests
import json
from datetime import datetime

def main():
    """Check the optimization status of vector collections."""
    # Get parameters from environment or use defaults
    host = os.getenv("VECTOR_DB_HOST", "vector_db")
    port = os.getenv("VECTOR_DB_PORT", "8000")
    collection = os.getenv("VECTOR_COLLECTION_NAME", "emails")
    
    # URL for the status endpoint
    url = f"http://{host}:{port}/api/v1/collections/{collection}/status"
    
    print(f"\nChecking optimization status for collection: {collection}")
    print(f"Vector DB URL: {url}\n")
    
    try:
        # Send request to the status endpoint
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        status = response.json()
        
        # Format the output nicely
        print(f"Collection: {status['name']}")
        print(f"Optimized: {'Yes' if status['is_optimized'] else 'No'}")
        print(f"Optimization coverage: {status['optimization_coverage']}")
        print(f"Total vectors: {status['total_vectors']}")
        print(f"Optimized vectors: {status['optimized_vectors']}")
        print(f"Pending new vectors: {status['pending_vectors']}")
        
        # Format timestamps
        last_hourly = datetime.fromisoformat(status['last_hourly_optimization'])
        last_daily = datetime.fromisoformat(status['last_daily_optimization'])
        next_opt = datetime.fromisoformat(status['next_optimization'])
        
        print(f"\nLast hourly optimization: {last_hourly.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Last daily optimization: {last_daily.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Next scheduled optimization: {next_opt.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate time until next optimization
        now = datetime.now()
        time_until = (next_opt - now).total_seconds()
        if time_until > 0:
            hours, remainder = divmod(time_until, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Time until next optimization: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        else:
            print("Next optimization is due now")
            
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