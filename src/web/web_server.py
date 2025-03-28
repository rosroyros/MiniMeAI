import os
import requests
import logging
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/web_server.log")
    ]
)
logger = logging.getLogger(__name__)

# Environment variables
API_HOST = os.getenv("API_HOST", "api_service")
API_PORT = os.getenv("API_PORT", "5000")
WEB_PORT = int(os.getenv("WEB_PORT", "8080"))
DEBUG_MODE = True  # Set to True for development/testing
DATA_API_HOST = os.getenv("DATA_API_HOST", "email_service")
DATA_API_PORT = int(os.getenv("DATA_API_PORT", "5000"))
VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "vector_db")
VECTOR_DB_PORT = int(os.getenv("VECTOR_DB_PORT", "8000"))
VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "emails")
WHATSAPP_API_HOST = os.getenv("WHATSAPP_API_HOST", "whatsapp_bridge")
WHATSAPP_API_PORT = int(os.getenv("WHATSAPP_API_PORT", "3001"))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Process a query and return results."""
    try:
        # Get data from form or JSON body
        if request.is_json:
            data = request.get_json()
            query = data.get('query', '')
            conversation_history = data.get('conversation_history', [])
        else:
            query = request.form.get('query', '')
            conversation_history = []
            
        if not query:
            return jsonify({"error": "Query is required"}), 400
            
        logger.info(f"Processing query: '{query}'")
        logger.info(f"Conversation history length: {len(conversation_history)}")
        
        # Prepare payload for API
        payload = {
            "query": query,
            "conversation_history": conversation_history
        }
        
        # Send query to API service
        response = requests.post(
            f"http://{API_HOST}:{API_PORT}/api/query",
            json=payload
        )
        
        # Check if request was successful
        if response.status_code != 200:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            return jsonify({"error": f"Failed to get response from API: {response.text}"}), response.status_code
        
        # Get response data
        result = response.json()
        logger.info(f"Received response: '{result.get('response', '')[:50]}...'")
        
        # Check for citations in the response
        citations = result.get('citations', [])
        if citations:
            logger.info(f"Received {len(citations)} citations")
            logger.info(f"Citation details: {json.dumps(citations)}")
            for i, citation in enumerate(citations):
                logger.info(f"Citation {i+1}: type={citation.get('source_type', 'unknown')}, score={citation.get('relevance_score', 0)}")
        else:
            logger.info("No citations received from API, adding dummy citation")
            # Add a dummy citation for debugging
            citations = [{
                "id": "dummy-id-web",
                "source_type": "email",
                "snippet": "This is a dummy citation added by the web server",
                "metadata": {
                    "from": "debug-web@example.com",
                    "subject": "Debug Web Citation",
                    "date": "2025-03-28"
                },
                "relevance_score": 0.999
            }]
            # Add the citations to the result
            result['citations'] = citations
        
        # Log complete response with citations
        logger.info(f"Returning response to client: {json.dumps(result)}")
        
        # Return as JSON for AJAX requests or render template for form submissions
        if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(result)
        else:
            return render_template('index.html', query=query, response=result.get('response', ''))
            
    except requests.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return jsonify({"error": f"Failed to connect to API service: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/health')
def health():
    """Enhanced health check endpoint with data source metrics."""
    from datetime import datetime, timedelta
    
    # ===== REAL IMPLEMENTATION =====
    try:
        # Basic service status checks
        api_status = "ok"
        try:
            response = requests.get(f"http://{API_HOST}:{API_PORT}/api/health", timeout=3)
            if response.status_code != 200:
                api_status = "error"
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            api_status = "error"
        
        # Data sources statistics
        health_stats = {
            "web_server": "ok",
            "api_service": api_status,
            "data_sources": {}
        }
        
        # Email metrics
        try:
            email_response = requests.get(f"http://{DATA_API_HOST}:{DATA_API_PORT}/api/emails", params={"limit": 500}, timeout=3)
            if email_response.status_code == 200:
                emails = email_response.json().get("emails", [])
                logger.info(f"Retrieved {len(emails)} emails from email service")
                if emails:
                    # Get count of emails in the last 24 hours
                    now = datetime.now()
                    yesterday = now - timedelta(days=1)
                    yesterday_timestamp = int(yesterday.timestamp())
                    logger.info(f"Checking for emails since timestamp: {yesterday_timestamp} ({yesterday.isoformat()})")
                    
                    # Process email dates to get valid timestamps
                    def get_email_timestamp(email_obj):
                        timestamp = email_obj.get("timestamp", 0)
                        if timestamp and timestamp > 0:
                            return timestamp
                        
                        # Try to parse the date string
                        date_str = email_obj.get("date", "")
                        if not date_str:
                            return 0
                        
                        try:
                            # Try email.utils parser (good for standard email formats)
                            dt = parsedate_to_datetime(date_str)
                            return int(dt.timestamp())
                        except Exception as e:
                            logger.error(f"Error parsing date '{date_str}': {e}")
                            
                            # Try a simple manual parsing for common formats
                            try:
                                # Try manual parsing with some common patterns
                                # Example: "13 Mar 2025 10:14:18 +0200"
                                import re
                                
                                # Extract date components with regex
                                pattern = r'(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})\s+(\d{1,2}):(\d{1,2}):(\d{1,2})'
                                match = re.search(pattern, date_str)
                                
                                if match:
                                    day, month_str, year, hour, minute, second = match.groups()
                                    
                                    # Convert month string to number
                                    months = {
                                        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                                    }
                                    month = months.get(month_str, 1)  # Default to January if not found
                                    
                                    # Create datetime object and return timestamp
                                    dt = datetime(int(year), month, int(day), int(hour), int(minute), int(second))
                                    return int(dt.timestamp())
                            except Exception:
                                return 0
                    
                    # Get timestamps for all emails
                    timestamps = [get_email_timestamp(email) for email in emails]
                    timestamps = [ts for ts in timestamps if ts > 0]
                    
                    if timestamps:
                        logger.info(f"Sample timestamps after parsing: {timestamps[:5]}")
                        logger.info(f"Sample dates: {[datetime.fromtimestamp(ts).strftime('%a, %d %b %Y %H:%M:%S %z') for ts in timestamps[:5]]}")
                    
                    # Count emails in the last 24 hours
                    recent_emails = [ts for ts in timestamps if ts >= yesterday_timestamp]
                    recent_count = len(recent_emails)
                    logger.info(f"Found {recent_count} emails in the last 24 hours")
                    
                    # Get the latest email timestamp
                    latest_timestamp = max(timestamps) if timestamps else 0
                    latest_date = datetime.fromtimestamp(latest_timestamp).isoformat() if latest_timestamp else None
                    
                    health_stats["data_sources"]["email"] = {
                        "status": "ok",
                        "total_count": len(emails),
                        "last_24h_count": recent_count,
                        "latest_timestamp": latest_timestamp,
                        "latest_date": latest_date
                    }
                else:
                    health_stats["data_sources"]["email"] = {
                        "status": "ok",
                        "total_count": 0,
                        "last_24h_count": 0,
                        "latest_timestamp": 0,
                        "latest_date": None
                    }
            else:
                health_stats["data_sources"]["email"] = {
                    "status": "error",
                    "error": f"Failed to get emails: {email_response.status_code}"
                }
        except Exception as e:
            logger.error(f"Error getting email metrics: {e}")
            health_stats["data_sources"]["email"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Vector DB metrics
        try:
            vector_response = requests.get(f"http://{VECTOR_DB_HOST}:{VECTOR_DB_PORT}/api/v1/collections/{VECTOR_COLLECTION_NAME}/status", timeout=3)
            
            if vector_response.status_code == 200:
                vector_data = vector_response.json()
                total_vectors = vector_data.get("total_vectors", 0)
                
                # Instead of using the count endpoint (which doesn't exist), let's estimate
                # using the email and whatsapp counts we already have
                email_count = health_stats.get("data_sources", {}).get("email", {}).get("total_count", 0)
                whatsapp_count = health_stats.get("data_sources", {}).get("whatsapp", {}).get("total_count", 0)
                total_documents = email_count + whatsapp_count
                
                if total_documents == 0:
                    # Fallback if we couldn't get email/whatsapp counts
                    total_documents = total_vectors // 30  # Rough estimate: each document ~30 chunks
                    if total_documents == 0 and total_vectors > 0:
                        total_documents = 1  # At least one document if we have vectors
                
                health_stats["data_sources"]["vector_db"] = {
                    "status": "ok",
                    "total_vectors": total_vectors,
                    "total_documents": total_documents
                }
            else:
                health_stats["data_sources"]["vector_db"] = {
                    "status": "error",
                    "error": f"Failed to get vector DB status: {vector_response.status_code}"
                }
        except Exception as e:
            logger.error(f"Error getting vector DB metrics: {e}")
            health_stats["data_sources"]["vector_db"] = {
                "status": "error",
                "error": str(e)
            }
        
        # WhatsApp metrics
        try:
            whatsapp_response = requests.get(f"http://{WHATSAPP_API_HOST}:{WHATSAPP_API_PORT}/api/whatsapp", params={"limit": 500}, timeout=3)
            if whatsapp_response.status_code == 200:
                messages = whatsapp_response.json().get("messages", [])
                if messages:
                    # Use current time for WhatsApp messages
                    recent_messages = [
                        msg for msg in messages 
                        if msg.get("timestamp", 0) >= yesterday_timestamp
                    ]
                    latest_msg = max(messages, key=lambda x: x.get("timestamp", 0))
                    latest_timestamp = latest_msg.get("timestamp", 0)
                    latest_date = datetime.fromtimestamp(latest_timestamp).isoformat() if latest_timestamp else None
                    
                    health_stats["data_sources"]["whatsapp"] = {
                        "status": "ok",
                        "total_count": len(messages),
                        "last_24h_count": len(recent_messages),
                        "latest_timestamp": latest_timestamp,
                        "latest_date": latest_date
                    }
                else:
                    health_stats["data_sources"]["whatsapp"] = {
                        "status": "ok",
                        "total_count": 0,
                        "last_24h_count": 0,
                        "latest_timestamp": 0,
                        "latest_date": None
                    }
            else:
                health_stats["data_sources"]["whatsapp"] = {
                    "status": "error",
                    "error": f"Failed to get WhatsApp messages: {whatsapp_response.status_code}"
                }
        except Exception as e:
            logger.error(f"Error getting WhatsApp metrics: {e}")
            health_stats["data_sources"]["whatsapp"] = {
                "status": "error",
                "error": str(e)
            }
        
        return jsonify(health_stats)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "web_server": "error",
            "api_service": "error",
            "data_sources": {
                "email": {"status": "error", "error": str(e)},
                "vector_db": {"status": "error", "error": str(e)},
                "whatsapp": {"status": "error", "error": str(e)}
            }
        }), 500
        
    # ===== MOCK DATA FOR DEMONSTRATION (COMMENTED OUT) =====
    # Note: This section uses static mock data instead of real metrics.
    """
    now = datetime.now()
    
    # Mock Email data - using fixed values instead of random
    email_latest = now - timedelta(minutes=45)
    health_stats["data_sources"]["email"] = {
        "status": "ok",
        "total_count": 1250,
        "last_24h_count": 37,
        "latest_timestamp": int(email_latest.timestamp()),
        "latest_date": email_latest.isoformat()
    }
    
    # Mock WhatsApp data - using fixed values instead of random
    whatsapp_latest = now - timedelta(minutes=75)
    health_stats["data_sources"]["whatsapp"] = {
        "status": "ok",
        "total_count": 580,
        "last_24h_count": 24,
        "latest_timestamp": int(whatsapp_latest.timestamp()),
        "latest_date": whatsapp_latest.isoformat()
    }
    
    # Mock Vector DB data - using fixed values instead of random
    health_stats["data_sources"]["vector_db"] = {
        "status": "ok",
        "total_count": 3450
    }
    
    return jsonify(health_stats), 200
    """

@app.route('/recent_emails')
def recent_emails():
    """Get recent emails."""
    try:
        # Get recent emails from email service via API
        response = requests.get(
            f"http://{API_HOST}:{API_PORT}/api/emails", 
            params={"limit": 10}
        )
        
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch recent emails"}), response.status_code
            
        return jsonify(response.json()), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    """Clear the conversation history."""
    return jsonify({"status": "success"}), 200

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    logger.info(f"Starting web server on port {WEB_PORT}")
    app.run(host="0.0.0.0", port=WEB_PORT, debug=DEBUG_MODE)
