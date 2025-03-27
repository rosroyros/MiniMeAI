import os
import requests
import logging
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for
from datetime import datetime, timedelta

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
                    
                    # Log some sample timestamps for debugging
                    sample_size = min(5, len(emails))
                    sample_timestamps = [e.get("timestamp", 0) for e in emails[:sample_size]]
                    sample_dates = [e.get("date", "unknown") for e in emails[:sample_size]]
                    logger.info(f"Sample timestamps: {sample_timestamps}")
                    logger.info(f"Sample dates: {sample_dates}")
                    
                    # Filter emails from last 24 hours
                    recent_emails = [e for e in emails if e.get("timestamp", 0) >= yesterday_timestamp]
                    logger.info(f"Found {len(recent_emails)} emails in the last 24 hours")
                    
                    # Find most recent email timestamp
                    timestamps = [e.get("timestamp", 0) for e in emails]
                    latest_timestamp = max(timestamps) if timestamps else 0
                    
                    health_stats["data_sources"]["email"] = {
                        "status": "ok",
                        "total_count": len(emails),
                        "last_24h_count": len(recent_emails),
                        "latest_timestamp": latest_timestamp,
                        "latest_date": datetime.fromtimestamp(latest_timestamp).isoformat() if latest_timestamp else None
                    }
                else:
                    health_stats["data_sources"]["email"] = {
                        "status": "ok",
                        "total_count": 0,
                        "last_24h_count": 0,
                        "latest_timestamp": None,
                        "latest_date": None
                    }
        except Exception as e:
            logger.error(f"Email metrics check failed: {e}")
            health_stats["data_sources"]["email"] = {
                "status": "error",
                "error": str(e)
            }
        
        # WhatsApp metrics
        try:
            whatsapp_response = requests.get(f"http://{WHATSAPP_API_HOST}:{WHATSAPP_API_PORT}/api/whatsapp", params={"limit": 500}, timeout=3)
            if whatsapp_response.status_code == 200:
                messages = whatsapp_response.json().get("messages", [])
                if messages:
                    # Get count of messages in the last 24 hours
                    now = datetime.now()
                    yesterday = now - timedelta(days=1)
                    yesterday_timestamp = int(yesterday.timestamp())
                    
                    # Filter messages from last 24 hours
                    recent_messages = [m for m in messages if m.get("timestamp", 0) >= yesterday_timestamp]
                    
                    # Find most recent message timestamp
                    timestamps = [m.get("timestamp", 0) for m in messages]
                    latest_timestamp = max(timestamps) if timestamps else 0
                    
                    health_stats["data_sources"]["whatsapp"] = {
                        "status": "ok",
                        "total_count": len(messages),
                        "last_24h_count": len(recent_messages),
                        "latest_timestamp": latest_timestamp,
                        "latest_date": datetime.fromtimestamp(latest_timestamp).isoformat() if latest_timestamp else None
                    }
                else:
                    health_stats["data_sources"]["whatsapp"] = {
                        "status": "ok",
                        "total_count": 0,
                        "last_24h_count": 0,
                        "latest_timestamp": None,
                        "latest_date": None
                    }
        except Exception as e:
            logger.error(f"WhatsApp metrics check failed: {e}")
            health_stats["data_sources"]["whatsapp"] = {
                "status": "unavailable",
                "error": str(e)
            }
        
        # Vector DB stats
        try:
            # Try the status endpoint instead of count, since it exists
            vector_db_response = requests.get(f"http://{VECTOR_DB_HOST}:{VECTOR_DB_PORT}/api/v1/collections/{VECTOR_COLLECTION_NAME}/status", timeout=3)
            if vector_db_response.status_code == 200:
                status_data = vector_db_response.json()
                vector_count = status_data.get("count", 0)
                if "count" not in status_data and "document_count" in status_data:
                    vector_count = status_data.get("document_count", 0)
                health_stats["data_sources"]["vector_db"] = {
                    "status": "ok",
                    "total_count": vector_count
                }
            else:
                health_stats["data_sources"]["vector_db"] = {
                    "status": "error",
                    "error": f"Status code: {vector_db_response.status_code}"
                }
        except Exception as e:
            logger.error(f"Vector DB metrics check failed: {e}")
            health_stats["data_sources"]["vector_db"] = {
                "status": "error",
                "error": str(e)
            }
        
        return jsonify(health_stats), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "web_server": "ok",
            "api_service": "unknown",
            "error": str(e)
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
