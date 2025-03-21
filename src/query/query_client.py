import argparse
import requests
import json
from typing import Optional

def query_minimeai(query: str, limit: int = 5, api_url: str = "http://localhost:5000"):
    """Send a query to the MiniMeAI API."""
    try:
        response = requests.post(
            f"{api_url}/api/query",  # Fixed endpoint with /api/ prefix
            json={"query": query, "limit": limit}
        )
        response.raise_for_status()
        result = response.json()
        
        return result
    except Exception as e:
        print(f"Error querying MiniMeAI: {e}")
        return None

def format_result(result: Optional[dict]):
    """Format the API response for display."""
    if not result:
        return "Failed to get a response from MiniMeAI."
    
    if "response" in result:
        return f"\n\nMiniMeAI response:\n{result['response']}"
    
    # Handle error case
    return f"\n\nError: {result.get('error', 'Unknown error')}"

def main():
    """Run the query client."""
    parser = argparse.ArgumentParser(description="Query MiniMeAI")
    parser.add_argument("query", help="The question to ask MiniMeAI")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of relevant emails to consider")
    parser.add_argument("--api", default="http://localhost:5000", help="MiniMeAI API URL")
    
    args = parser.parse_args()
    
    print(f"Querying MiniMeAI: '{args.query}'")
    result = query_minimeai(args.query, args.limit, args.api)
    print(format_result(result))

if __name__ == "__main__":
    main()
