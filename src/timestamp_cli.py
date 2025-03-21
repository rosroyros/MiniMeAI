#!/usr/bin/env python3
"""
Timestamp CLI - A command line utility for working with document timestamps in vector collections

This utility provides various commands for:
- Analyzing timestamp quality in collections
- Normalizing/fixing missing timestamps
- Validating existing timestamps
- Testing search results with recency-based queries
"""

import argparse
import logging
import os
import sys
import json
import time
from datetime import datetime

# Import our utility modules
try:
    from analyze_collection_timestamps import analyze_collection_timestamps
    from normalize_timestamps import normalize_collection_timestamps
    from validate_timestamps import validate_timestamps
except ImportError:
    print("Error: Unable to import timestamp utility modules.")
    print("Make sure you are running this script from the src directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_recency_search(host: str, port: str, collection: str) -> dict:
    """
    Test recency-based search to ensure timestamps are working properly.
    
    Args:
        host: Database host
        port: Database port
        collection: Collection name
        
    Returns:
        dict: Search results
    """
    import requests
    
    base_url = f"http://{host}:{port}"
    
    try:
        # Search for recent documents
        response = requests.post(
            f"{base_url}/api/v1/collections/{collection}/query",
            json={
                "query_text": "recent", 
                "n_results": 10
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to search for recent documents: {response.status_code} - {response.text}")
            return {"error": "Failed to search for recent documents"}
            
        results = response.json()
        
        # Add formatted timestamps to results for easier reading
        if "metadatas" in results:
            for metadata in results["metadatas"]:
                if "timestamp" in metadata and isinstance(metadata["timestamp"], (int, float)):
                    timestamp = metadata["timestamp"]
                    metadata["timestamp_formatted"] = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during recency search: {e}")
        return {"error": str(e)}

def main():
    # Create the main parser
    parser = argparse.ArgumentParser(
        description='Timestamp CLI for vector collections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Global arguments
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', default='8000', help='Database port')
    parser.add_argument('--collection', default='emails', help='Collection name')
    parser.add_argument('--remote', action='store_true', help='Use the remote server at 10.0.0.58')
    parser.add_argument('--output', help='Output file for results (JSON)')
    parser.add_argument('--verbose', action='store_true', help='Show more detailed logging')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze timestamp quality in a collection')
    
    # Normalize command
    normalize_parser = subparsers.add_parser('normalize', help='Add timestamps to documents missing them')
    normalize_parser.add_argument('--dry-run', action='store_true', help='Analyze without making changes')
    normalize_parser.add_argument('--batch-size', type=int, default=100, help='Number of documents to process in each batch')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate existing timestamps')
    validate_parser.add_argument('--fix', action='store_true', help='Fix invalid timestamps')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Test recency-based search')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set host for remote access if needed
    host = args.host
    port = args.port
    if args.remote:
        host = "10.0.0.58"
    
    # Default output filename based on command
    if not args.output:
        args.output = f"{args.collection}_{args.command}_results.json"
    
    # Execute the specified command
    if args.command == 'analyze':
        logger.info(f"Analyzing timestamps in collection {args.collection} on {host}:{port}")
        results = analyze_collection_timestamps(host, port, args.collection)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Analysis complete. Results saved to {args.output}")
        
        # Print a summary
        print("\n=== TIMESTAMP ANALYSIS SUMMARY ===")
        print(f"Total documents: {results['total_docs']:,}")
        print(f"Documents with timestamp: {results['has_timestamp_field']:,} ({results['percent_with_timestamp']}%)")
        print(f"Valid timestamps: {results['valid_timestamps']:,} ({results['percent_valid_timestamps']}%)")
        
        # Age distribution
        print("\nAGE DISTRIBUTION:")
        if "age_distribution" in results:
            if "percentages" in results["age_distribution"]:
                # New format
                for key, percentage in results["age_distribution"]["percentages"].items():
                    count = results["age_distribution"]["counts"][key]
                    print(f"  {key}: {count:,} ({percentage:.2f}%)")
            else:
                # Old format
                for key in ["last_day", "last_week", "last_month", "last_year", "older_than_year", "unknown_age"]:
                    if key in results["age_distribution"]:
                        count = results["age_distribution"][key]
                        percent_key = f"percent_{key}"
                        percentage = results["age_distribution"].get(percent_key, 0)
                        print(f"  {key}: {count:,} ({percentage}%)")
        
    elif args.command == 'normalize':
        logger.info(f"Normalizing timestamps in collection {args.collection} on {host}:{port}")
        if args.dry_run:
            logger.info("Dry run mode - no changes will be made")
            
        results = normalize_collection_timestamps(
            host=host,
            port=port,
            collection=args.collection,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Normalization complete. Results saved to {args.output}")
        
        # Print a summary
        print("\n=== TIMESTAMP NORMALIZATION SUMMARY ===")
        print(f"Total documents: {results['total_documents']:,}")
        print(f"Documents needing timestamp: {results['documents_needing_timestamp']:,}")
        if not args.dry_run:
            print(f"Documents successfully updated: {results['documents_successfully_updated']:,}")
            print(f"Documents failed: {results['documents_failed']:,}")
        print(f"Execution time: {results['execution_time_seconds']:.2f} seconds")
        
    elif args.command == 'validate':
        logger.info(f"Validating timestamps in collection {args.collection} on {host}:{port}")
        if args.fix:
            logger.info("Fix mode enabled - invalid timestamps will be corrected")
            
        results = validate_timestamps(
            host=host,
            port=port,
            collection=args.collection,
            fix_issues=args.fix
        )
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Validation complete. Results saved to {args.output}")
        
        # Print a summary
        print("\n=== TIMESTAMP VALIDATION SUMMARY ===")
        print(f"Total documents: {results['total_documents']:,}")
        print(f"Documents with timestamp: {results['documents_with_timestamp']:,} ({results['percent_with_timestamp']:.2f}%)")
        print(f"Valid timestamps: {results['valid_timestamps']:,} ({results['percent_valid']:.2f}%)")
        print(f"Invalid timestamps: {results['invalid_timestamps']:,} ({results['percent_invalid']:.2f}%)")
        print(f"Abnormal timestamps: {results['abnormal_timestamps']:,} ({results['percent_abnormal']:.2f}%)")
        
        if args.fix:
            print(f"Fixed timestamps: {results['fixed_timestamps']:,}")
        
    elif args.command == 'search':
        logger.info(f"Testing recency-based search in collection {args.collection} on {host}:{port}")
        
        results = test_recency_search(host, port, args.collection)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Search test complete. Results saved to {args.output}")
        
        # Print a summary
        print("\n=== RECENCY SEARCH TEST RESULTS ===")
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            ids = results.get("ids", [])
            metadatas = results.get("metadatas", [])
            
            print(f"Found {len(ids)} documents")
            print("\nTOP RESULTS:")
            
            for i, (doc_id, metadata) in enumerate(zip(ids, metadatas)):
                if i >= 5:  # Show only top 5
                    break
                    
                print(f"\nDocument {i+1}:")
                print(f"  ID: {doc_id}")
                timestamp_fmt = metadata.get("timestamp_formatted", "unknown")
                print(f"  Timestamp: {timestamp_fmt}")
                
                # Print a few other metadata fields if available
                for field in ["source_type", "subject", "from", "to"]:
                    if field in metadata:
                        print(f"  {field.capitalize()}: {metadata[field]}")
    else:
        parser.print_help()
        
if __name__ == "__main__":
    main() 