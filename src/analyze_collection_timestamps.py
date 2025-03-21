#!/usr/bin/env python3
import os
import sys
import requests
import json
import argparse
from datetime import datetime

def analyze_collection_timestamps(host: str, port: str, collection: str) -> dict:
    """
    Fetch and analyze timestamp information for a collection.
    
    Args:
        host: Database host
        port: Database port
        collection: Collection name
        
    Returns:
        dict: Timestamp analysis results
    """
    url = f"http://{host}:{port}/api/v1/collections/{collection}/timestamp_analysis"
    
    try:
        # Call the timestamp analysis endpoint
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to vector database: {e}")
        sys.exit(1)

def format_report(analysis: dict) -> str:
    """
    Format the timestamp analysis into a readable report.
    
    Args:
        analysis: Timestamp analysis results
        
    Returns:
        str: Formatted report
    """
    report = []
    report.append("=" * 80)
    report.append(f"TIMESTAMP ANALYSIS REPORT")
    report.append("=" * 80)
    
    # Basic statistics
    report.append("\nBASIC STATISTICS:")
    report.append(f"Total documents: {analysis['total_docs']:,}")
    report.append(f"Documents with date field: {analysis['has_date_field']:,} ({analysis['percent_with_date']}%)")
    report.append(f"Documents with timestamp field: {analysis['has_timestamp_field']:,} ({analysis['percent_with_timestamp']}%)")
    report.append(f"Valid timestamps: {analysis['valid_timestamps']:,} ({analysis['percent_valid_timestamps']}%)")
    
    # Timestamp range
    report.append("\nTIMESTAMP RANGE:")
    report.append(f"Minimum timestamp: {analysis['timestamp_min'] if analysis['timestamp_min'] else 'N/A'}")
    report.append(f"Maximum timestamp: {analysis['timestamp_max'] if analysis['timestamp_max'] else 'N/A'}")
    
    # Age distribution
    report.append("\nAGE DISTRIBUTION:")
    age_counts = analysis["age_distribution"]["counts"]
    age_percentages = analysis["age_distribution"]["percentages"]
    
    report.append(f"Last day: {age_counts['last_day']:,} ({age_percentages['last_day']:.2f}%)")
    report.append(f"Last week: {age_counts['last_week']:,} ({age_percentages['last_week']:.2f}%)")
    report.append(f"Last month: {age_counts['last_month']:,} ({age_percentages['last_month']:.2f}%)")
    report.append(f"Last year: {age_counts['last_year']:,} ({age_percentages['last_year']:.2f}%)")
    report.append(f"Older: {age_counts['older']:,} ({age_percentages['older']:.2f}%)")
    report.append(f"Unknown age: {age_counts['unknown']:,} ({age_percentages['unknown']:.2f}%)")
    
    # Source type distribution
    if analysis.get("source_type_distribution"):
        report.append("\nSOURCE TYPE DISTRIBUTION:")
        source_types = analysis["source_type_distribution"]
        source_types_with_timestamps = analysis.get("source_type_with_timestamp", {})
        
        for source_type, count in sorted(source_types.items(), key=lambda x: x[1], reverse=True):
            timestamp_count = source_types_with_timestamps.get(source_type, 0)
            timestamp_percentage = (timestamp_count / count) * 100 if count > 0 else 0
            
            report.append(f"{source_type}: {count:,} documents, "
                         f"{timestamp_count:,} with timestamps ({timestamp_percentage:.2f}%)")
    
    # Impact analysis
    report.append("\nIMPACT ANALYSIS:")
    missing_timestamp_percent = age_percentages["unknown"]
    if missing_timestamp_percent > 20:
        report.append(f"CRITICAL: {missing_timestamp_percent:.2f}% of documents have no timestamp")
        report.append("This will significantly impact time-based searches and sorting")
    elif missing_timestamp_percent > 5:
        report.append(f"WARNING: {missing_timestamp_percent:.2f}% of documents have no timestamp")
        report.append("This may affect time-based searches and sorting")
    else:
        report.append(f"GOOD: Only {missing_timestamp_percent:.2f}% of documents have no timestamp")
    
    return "\n".join(report)

def main():
    """Analyze timestamp issues in vector collections by querying the vector_db API."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze timestamp issues in vector collections.')
    parser.add_argument('--host', default=os.getenv("VECTOR_DB_HOST", "localhost"), 
                        help='Host where the vector DB is running (default: VECTOR_DB_HOST env var or localhost)')
    parser.add_argument('--port', default=os.getenv("VECTOR_DB_PORT", "8000"),
                        help='Port where the vector DB is running (default: VECTOR_DB_PORT env var or 8000)')
    parser.add_argument('--collection', default=os.getenv("VECTOR_COLLECTION_NAME", "emails"),
                        help='Name of the collection to analyze (default: VECTOR_COLLECTION_NAME env var or emails)')
    parser.add_argument('--remote-ip', default="10.0.0.58",
                        help='Remote server IP address (default: 10.0.0.58)')
    parser.add_argument('--use-remote', action='store_true',
                        help='Use the remote server IP instead of localhost/host')
    parser.add_argument('--output-file', default=None, help='Output file for analysis results (JSON)')
    
    args = parser.parse_args()
    
    # Determine host to use
    host = args.remote_ip if args.use_remote else args.host
    port = args.port
    collection = args.collection
    
    # URL for the timestamp analysis endpoint
    url = f"http://{host}:{port}/api/v1/collections/{collection}/timestamp_analysis"
    
    print(f"\nAnalyzing timestamps for collection: {collection}")
    print(f"Vector DB URL: {url}\n")
    
    try:
        # Analyze timestamps
        analysis = analyze_collection_timestamps(
            host=host,
            port=port,
            collection=collection
        )
        
        # Format and print report
        report = format_report(analysis)
        print(report)
        
        # Determine output filename
        output_file = args.output_file
        if not output_file:
            output_file = f"{collection}_timestamp_analysis.json"
        
        # Save raw analysis results to file
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\nRaw analysis results saved to {output_file}")
        
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to vector DB at {url}. Make sure the service is running.")
        print("If using Docker, make sure port forwarding is enabled or connect directly to the container network.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main() 