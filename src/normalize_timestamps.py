#!/usr/bin/env python3
import os
import sys
import requests
import json
import argparse
from datetime import datetime, timedelta
import logging
import time
from email.utils import parsedate_to_datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_reliable_timestamp(date_str: str) -> int:
    """
    Attempt to extract a reliable timestamp from various date formats.
    
    Args:
        date_str: A string containing a date in various possible formats
        
    Returns:
        int: Unix timestamp (seconds since epoch)
    """
    if not date_str:
        logger.warning(f"Empty date string provided")
        return None
        
    # Try RFC 2822 format (email standard) first
    try:
        dt = parsedate_to_datetime(date_str)
        return int(dt.timestamp())
    except Exception as e:
        logger.debug(f"RFC 2822 parsing failed for '{date_str}': {e}")
    
    # Try ISO 8601 and other common formats
    formats = [
        "%Y-%m-%d %H:%M:%S",  # 2023-01-31 14:30:00
        "%Y-%m-%dT%H:%M:%S",  # 2023-01-31T14:30:00
        "%Y-%m-%dT%H:%M:%SZ", # 2023-01-31T14:30:00Z
        "%Y-%m-%d",           # 2023-01-31
        "%d/%m/%Y %H:%M:%S",  # 31/01/2023 14:30:00
        "%d/%m/%Y",           # 31/01/2023
        "%m/%d/%Y %H:%M:%S",  # 01/31/2023 14:30:00
        "%m/%d/%Y",           # 01/31/2023
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return int(dt.timestamp())
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date string: '{date_str}'")
    return None

def normalize_collection_timestamps(host: str, port: str, collection: str, batch_size: int = 100, dry_run: bool = False) -> dict:
    """
    Normalize timestamps in the specified collection.
    
    Args:
        host: Database host
        port: Database port
        collection: Collection name
        batch_size: Number of documents to process in each batch
        dry_run: If True, don't actually update the documents
        
    Returns:
        dict: Summary statistics about the normalization process
    """
    base_url = f"http://{host}:{port}"
    stats = {
        "total_documents": 0,
        "documents_needing_timestamp": 0,
        "documents_successfully_updated": 0,
        "documents_failed": 0,
        "execution_time_seconds": 0
    }
    
    start_time = time.time()
    
    # Get documents that have a date field but no timestamp field
    query_url = f"{base_url}/api/v1/collections/{collection}/query"
    
    filter_condition = {
        "filter": {
            "field_exists": {"date": True},
            "field_missing": {"timestamp": True}
        }
    }
    
    try:
        # First, get total count of documents
        logger.debug(f"Getting collection count from {base_url}/api/v1/collections/{collection}/count")
        response = requests.post(
            f"{base_url}/api/v1/collections/{collection}/count",
            json={}
        )
        if response.status_code == 200:
            stats["total_documents"] = response.json().get("count", 0)
            logger.debug(f"Collection count: {stats['total_documents']}")
        else:
            logger.error(f"Failed to get collection count: {response.status_code} - {response.text}")
            return stats
            
        # Fetch documents without timestamp but with date
        logger.debug("Querying for documents with date but no timestamp")
        response = requests.post(
            query_url,
            json={
                "query_text": "",
                "n_results": 10000,  # Get as many as reasonable
                "filter": {
                    "field_exists": {"date": True},
                    "field_missing": {"timestamp": True}
                }
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to query collection: {response.status_code} - {response.text}")
            return stats
            
        documents = response.json()
        doc_ids = documents["ids"]
        metadatas = documents["metadatas"]
        
        stats["documents_needing_timestamp"] = len(doc_ids)
        logger.info(f"Found {len(doc_ids)} documents needing timestamp normalization")
        
        if dry_run:
            logger.info("Dry run mode - not applying updates")
            stats["execution_time_seconds"] = time.time() - start_time
            return stats
            
        # Process in batches
        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            updates = []
            
            for doc_id, metadata in zip(batch_ids, batch_metadatas):
                date_str = metadata.get("date")
                if date_str:
                    timestamp = get_reliable_timestamp(date_str)
                    if timestamp:
                        updates.append({
                            "id": doc_id,
                            "metadata": {"timestamp": timestamp}
                        })
                        logger.debug(f"Adding update for document {doc_id}: timestamp {timestamp} from date {date_str}")
            
            if updates:
                logger.info(f"Applying batch update for {len(updates)} documents")
                update_url = f"{base_url}/api/v1/collections/{collection}/update"
                update_payload = {"updates": updates}
                logger.debug(f"Update payload: {json.dumps(update_payload)}")
                
                update_response = requests.post(
                    update_url,
                    json=update_payload
                )
                
                if update_response.status_code == 200:
                    response_data = update_response.json()
                    updated_count = response_data.get("updated_count", 0)
                    stats["documents_successfully_updated"] += updated_count
                    logger.info(f"Successfully updated {updated_count} documents")
                    logger.debug(f"Update response: {update_response.text}")
                else:
                    logger.error(f"Failed to update documents: {update_response.status_code} - {update_response.text}")
                    stats["documents_failed"] += len(updates)
        
    except Exception as e:
        logger.error(f"Error during timestamp normalization: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        
    stats["execution_time_seconds"] = time.time() - start_time
    return stats

def main():
    parser = argparse.ArgumentParser(description='Normalize timestamps in the collection')
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', default='8000', help='Database port')
    parser.add_argument('--collection', default='emails', help='Collection name')
    parser.add_argument('--remote', action='store_true', help='Set to true for remote access')
    parser.add_argument('--dry-run', action='store_true', help='Analyze without updating documents')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of documents to process in each batch')
    parser.add_argument('--verbose', action='store_true', help='Show more detailed logging')
    args = parser.parse_args()
    
    # Set host for remote access if needed
    host = args.host
    port = args.port
    if args.remote:
        # Use the correct remote host
        host = "10.0.0.58"  # or use environment variable
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Starting timestamp normalization for collection {args.collection} on {host}:{port}")
    
    results = normalize_collection_timestamps(
        host=host,
        port=port,
        collection=args.collection,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )
    
    # Write results to a file
    output_file = f"{args.collection}_timestamp_normalization.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Normalization complete. Results saved to {output_file}")
    
    # Print summary
    print("\n==== Timestamp Normalization Summary ====")
    print(f"Total documents in collection: {results['total_documents']}")
    print(f"Documents needing timestamp: {results['documents_needing_timestamp']}")
    if results['total_documents'] > 0:
        percent_needing = results['documents_needing_timestamp']/results['total_documents']*100
        print(f"Documents needing timestamp: {results['documents_needing_timestamp']} ({percent_needing:.2f}%)")
    else:
        print(f"Documents needing timestamp: {results['documents_needing_timestamp']} (0.00%)")
    print(f"Documents successfully updated: {results['documents_successfully_updated']}")
    print(f"Documents failed: {results['documents_failed']}")
    print(f"Execution time: {results['execution_time_seconds']:.2f} seconds")
    
if __name__ == "__main__":
    main() 