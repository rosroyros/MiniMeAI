import argparse
import json
import logging
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_timestamps(host: str, port: str, collection: str, fix_issues: bool = False) -> Dict[str, Any]:
    """
    Validate timestamps in a collection and report issues.
    
    Args:
        host: Database host
        port: Database port 
        collection: Collection name
        fix_issues: Whether to automatically fix invalid timestamps
        
    Returns:
        dict: Dictionary with validation results
    """
    base_url = f"http://{host}:{port}"
    
    # Initialize results
    results = {
        "total_documents": 0,
        "documents_with_timestamp": 0,
        "valid_timestamps": 0,
        "invalid_timestamps": 0,
        "abnormal_timestamps": 0,  # Valid but unusual (future, very old)
        "fixed_timestamps": 0,
        "issues": []
    }
    
    try:
        # Get collection count
        response = requests.post(
            f"{base_url}/api/v1/collections/{collection}/count", 
            json={}
        )
        if response.status_code != 200:
            logger.error(f"Failed to get collection count: {response.status_code} - {response.text}")
            return results
            
        results["total_documents"] = response.json().get("count", 0)
        
        # Query for documents with timestamp field
        response = requests.post(
            f"{base_url}/api/v1/collections/{collection}/query",
            json={
                "query_text": "",
                "n_results": 10000,  # Get as many as reasonable
                "filter": {
                    "field_exists": {"timestamp": True}
                }
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to query documents: {response.status_code} - {response.text}")
            return results
            
        documents = response.json()
        doc_ids = documents["ids"]
        metadatas = documents["metadatas"]
        
        results["documents_with_timestamp"] = len(doc_ids)
        logger.info(f"Found {len(doc_ids)} documents with timestamp field")
        
        # Get current time and reasonable time boundaries (10 years ago to 1 day in future)
        now = time.time()
        ten_years_ago = now - (10 * 365 * 24 * 60 * 60)
        one_day_future = now + (24 * 60 * 60)
        
        updates = []
        
        # Validate each timestamp
        for i, (doc_id, metadata) in enumerate(zip(doc_ids, metadatas)):
            if i % 1000 == 0 and i > 0:
                logger.info(f"Processed {i} documents...")
                
            timestamp = metadata.get("timestamp")
            
            # Check if timestamp exists and is a valid number
            if timestamp is None:
                results["invalid_timestamps"] += 1
                issue = {
                    "id": doc_id,
                    "issue": "null_timestamp",
                    "value": None
                }
                results["issues"].append(issue)
                
                # Fix if requested
                if fix_issues and "date" in metadata:
                    date_str = metadata.get("date")
                    if date_str:
                        # Try to parse the date string
                        try:
                            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            new_timestamp = int(dt.timestamp())
                            updates.append({
                                "id": doc_id,
                                "metadata": {"timestamp": new_timestamp}
                            })
                        except ValueError:
                            # Try email format
                            try:
                                from email.utils import parsedate_to_datetime
                                dt = parsedate_to_datetime(date_str)
                                new_timestamp = int(dt.timestamp())
                                updates.append({
                                    "id": doc_id,
                                    "metadata": {"timestamp": new_timestamp}
                                })
                            except Exception:
                                # Can't parse, use current time as fallback
                                updates.append({
                                    "id": doc_id,
                                    "metadata": {"timestamp": int(now)}
                                })
                continue
                
            # Check if timestamp is a number
            if not isinstance(timestamp, (int, float)):
                results["invalid_timestamps"] += 1
                issue = {
                    "id": doc_id,
                    "issue": "non_numeric_timestamp",
                    "value": str(timestamp)
                }
                results["issues"].append(issue)
                
                # Fix if requested
                if fix_issues:
                    updates.append({
                        "id": doc_id,
                        "metadata": {"timestamp": int(now)}
                    })
                continue
                
            # Check if timestamp is zero
            if timestamp == 0:
                results["invalid_timestamps"] += 1
                issue = {
                    "id": doc_id,
                    "issue": "zero_timestamp",
                    "value": 0
                }
                results["issues"].append(issue)
                
                # Fix if requested
                if fix_issues:
                    updates.append({
                        "id": doc_id,
                        "metadata": {"timestamp": int(now)}
                    })
                continue
                
            # Check if timestamp is reasonable (not too old, not too far in future)
            if timestamp < ten_years_ago:
                results["abnormal_timestamps"] += 1
                issue = {
                    "id": doc_id,
                    "issue": "timestamp_too_old",
                    "value": timestamp,
                    "date": datetime.fromtimestamp(timestamp).isoformat()
                }
                results["issues"].append(issue)
                
                # Fix if requested and truly unreasonable (before 1990)
                if fix_issues and timestamp < 631152000:  # Jan 1, 1990
                    updates.append({
                        "id": doc_id,
                        "metadata": {"timestamp": int(now)}
                    })
                    
            elif timestamp > one_day_future:
                results["abnormal_timestamps"] += 1
                issue = {
                    "id": doc_id,
                    "issue": "timestamp_in_future",
                    "value": timestamp,
                    "date": datetime.fromtimestamp(timestamp).isoformat()
                }
                results["issues"].append(issue)
                
                # Fix if requested
                if fix_issues:
                    updates.append({
                        "id": doc_id,
                        "metadata": {"timestamp": int(now)}
                    })
            else:
                # Timestamp is valid
                results["valid_timestamps"] += 1
        
        # Apply updates if fixing
        if fix_issues and updates:
            logger.info(f"Fixing {len(updates)} invalid timestamps...")
            
            # Process in batches of 100
            for i in range(0, len(updates), 100):
                batch = updates[i:i+100]
                response = requests.post(
                    f"{base_url}/api/v1/collections/{collection}/update",
                    json={"updates": batch}
                )
                
                if response.status_code == 200:
                    updated_count = response.json().get("updated_count", 0)
                    results["fixed_timestamps"] += updated_count
                    logger.info(f"Fixed batch of {updated_count} timestamps")
                else:
                    logger.error(f"Failed to update documents: {response.status_code} - {response.text}")
                    
                # Small delay to avoid overwhelming the server
                time.sleep(0.1)
        
        # Calculate percentages
        results["percent_with_timestamp"] = (results["documents_with_timestamp"] / results["total_documents"]) * 100 if results["total_documents"] > 0 else 0
        results["percent_valid"] = (results["valid_timestamps"] / results["documents_with_timestamp"]) * 100 if results["documents_with_timestamp"] > 0 else 0
        results["percent_invalid"] = (results["invalid_timestamps"] / results["documents_with_timestamp"]) * 100 if results["documents_with_timestamp"] > 0 else 0
        results["percent_abnormal"] = (results["abnormal_timestamps"] / results["documents_with_timestamp"]) * 100 if results["documents_with_timestamp"] > 0 else 0
        
    except Exception as e:
        logger.error(f"Error during timestamp validation: {e}")
        
    return results

def main():
    parser = argparse.ArgumentParser(description='Validate timestamps in a collection')
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', default='8000', help='Database port')
    parser.add_argument('--collection', default='emails', help='Collection name')
    parser.add_argument('--remote', action='store_true', help='Set to true for remote access')
    parser.add_argument('--fix', action='store_true', help='Fix invalid timestamps')
    parser.add_argument('--output-file', default=None, help='Output file for validation results (JSON)')
    args = parser.parse_args()
    
    # Set host for remote access if needed
    host = args.host
    port = args.port
    if args.remote:
        # Use the correct remote host
        host = "10.0.0.58"  # or use environment variable
    
    logger.info(f"Validating timestamps for collection {args.collection} on {host}:{port}")
    if args.fix:
        logger.info("Fix mode enabled - invalid timestamps will be corrected")
    
    # Validate timestamps
    results = validate_timestamps(
        host=host,
        port=port,
        collection=args.collection,
        fix_issues=args.fix
    )
    
    # Determine output filename
    output_file = args.output_file
    if not output_file:
        output_file = f"{args.collection}_timestamp_validation.json"
    
    # Save validation results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n==== Timestamp Validation Summary ====")
    print(f"Total documents in collection: {results['total_documents']:,}")
    print(f"Documents with timestamp field: {results['documents_with_timestamp']:,} ({results['percent_with_timestamp']:.2f}%)")
    print(f"Valid timestamps: {results['valid_timestamps']:,} ({results['percent_valid']:.2f}%)")
    print(f"Invalid timestamps: {results['invalid_timestamps']:,} ({results['percent_invalid']:.2f}%)")
    print(f"Abnormal timestamps: {results['abnormal_timestamps']:,} ({results['percent_abnormal']:.2f}%)")
    
    if args.fix:
        print(f"Fixed timestamps: {results['fixed_timestamps']:,}")
    
    if results["issues"]:
        print(f"\nFound {len(results['issues'])} issues (see {output_file} for details)")
        issue_types = {}
        for issue in results["issues"]:
            issue_type = issue["issue"]
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            
        print("\nIssue types:")
        for issue_type, count in issue_types.items():
            print(f"  {issue_type}: {count:,}")
    else:
        print("\nNo issues found - all timestamps are valid!")
    
    print(f"\nDetailed validation results saved to {output_file}")

if __name__ == "__main__":
    main() 