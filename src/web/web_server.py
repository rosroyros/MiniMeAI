import os
import requests
import logging
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for

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
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

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
    """Health check endpoint."""
    try:
        # Check API service health
        response = requests.get(f"http://{API_HOST}:{API_PORT}/api/health")
        api_status = response.status_code == 200
        
        status = {
            "web_server": "ok",
            "api_service": "ok" if api_status else "error"
        }
        
        return jsonify(status), 200 if api_status else 500
    except Exception as e:
        return jsonify({
            "web_server": "ok",
            "api_service": "error",
            "error": str(e)
        }), 500

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
