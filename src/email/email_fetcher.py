import os
import time
import imaplib
import email
import email.utils
import logging
import json
from datetime import datetime, timedelta
from email.header import decode_header
from typing import List, Dict, Any, Optional
from flask import Flask, jsonify, request
from flask_cors import CORS

# Simple internal timing implementation to avoid import errors
import functools

class Timer:
    """Utility class for timing code execution."""
    
    def __init__(self, name="Operation"):
        """Initialize timer with operation name."""
        self.name = name
        
    def __enter__(self):
        """Start timer when entering context."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        """Log elapsed time when exiting context."""
        elapsed_time = time.time() - self.start_time
        logging.info(f"{self.name} took {elapsed_time:.4f} seconds")

def timed(func):
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logging.info(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/email_fetcher.log")
    ]
)
logger = logging.getLogger(__name__)

# Environment variables
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_IMAP_SERVER = os.getenv("EMAIL_IMAP_SERVER", "imap.gmail.com")
EMAIL_IMAP_PORT = int(os.getenv("EMAIL_IMAP_PORT", "993"))
EMAIL_CACHE_FILE = os.getenv("EMAIL_CACHE_FILE", "config/email_cache.json")
EMAIL_FETCH_INTERVAL = int(os.getenv("EMAIL_FETCH_INTERVAL", "600"))  # 10 minutes
EMAIL_FETCH_COUNT = int(os.getenv("EMAIL_MAX_EMAILS", "100"))  # Using EMAIL_MAX_EMAILS from .env
EMAIL_SYNC_DAYS = int(os.getenv("EMAIL_MAX_DAYS", "7"))  # Using EMAIL_MAX_DAYS from .env
PRESERVE_READ_STATUS = os.getenv("PRESERVE_READ_STATUS", "True").lower() == "true"

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global cache for emails
email_cache = {}

class EmailFetcher:
    def connect_to_imap(self):
        """Connect to IMAP server."""
        try:
            # Connect to the IMAP server
            mail = imaplib.IMAP4_SSL(EMAIL_IMAP_SERVER, EMAIL_IMAP_PORT)
            
            # Login
            mail.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            
            return mail
        except Exception as e:
            logger.error(f"Error connecting to IMAP server: {e}")
            return None

    @timed
    def fetch_emails(self, limit=EMAIL_FETCH_COUNT) -> List[Dict[str, Any]]:
        """Fetch emails from IMAP server while preserving unread status."""
        try:
            # Connect to IMAP
            mail = self.connect_to_imap()
            if not mail:
                logger.error("Failed to connect to IMAP server")
                return []
            
            # Select the mailbox (INBOX)
            mail.select("INBOX", readonly=PRESERVE_READ_STATUS)  # Set readonly=True to preserve flags
            
            # Calculate date filter for sync period
            sync_date = (datetime.now() - timedelta(days=2)).strftime("%d-%b-%Y")
            search_criteria = f'(SINCE "{sync_date}")'
            
            logger.info(f"Searching for emails since {sync_date}")
            
            # Search for emails
            logger.info(f"Searching with criteria: {search_criteria}")
            status, messages = mail.search(None, search_criteria)
            if status != "OK":
                logger.error(f"Failed to search emails: {status}")
                return []
            # Get email IDs
            email_ids = messages[0].split()
            if not email_ids:
                logger.info("No emails found matching criteria")
                return []
            logger.info(f"Found {len(email_ids)} emails matching search criteria")
            
            # Fetch basic info for all emails to sort by date
            date_dict = {}
            for email_id in email_ids:
                try:
                    status, header_data = mail.fetch(email_id, '(BODY.PEEK[HEADER.FIELDS (DATE)])')
                    if status == "OK":
                        header = email.message_from_bytes(header_data[0][1])
                        date_str = header.get("Date", "")
                        # Parse the date
                        try:
                            # Try to parse the date string to get a datetime object
                            date_tuple = email.utils.parsedate_tz(date_str)
                            if date_tuple:
                                date_obj = datetime.fromtimestamp(email.utils.mktime_tz(date_tuple))
                                date_dict[email_id] = date_obj
                            else:
                                date_dict[email_id] = datetime.min
                        except Exception as e:
                            logger.error(f"Error parsing date {date_str}: {e}")
                            date_dict[email_id] = datetime.min
                except Exception as e:
                    logger.error(f"Error fetching date for email {email_id}: {e}")
                    date_dict[email_id] = datetime.min

            # Sort emails by date (newest first)
            sorted_ids = sorted(date_dict.keys(), key=lambda x: date_dict[x], reverse=True)
            
            # Limit to the specified number of emails
            email_ids = sorted_ids[:limit]
            
            logger.info(f"Processing {len(email_ids)} emails, newest first")
            
            # Fetch each email
            emails = []
            for email_id in email_ids:
                try:
                    # Convert ID to string
                    msg_id = email_id.decode("utf-8")
                    
                    # Check if email is already in cache
                    if msg_id in email_cache:
                        emails.append(email_cache[msg_id])
                        continue
                    
                    # Check if the message is unread before fetching
                    status, flags_data = mail.fetch(email_id, '(FLAGS)')
                    if status != "OK":
                        logger.error(f"Failed to fetch flags for email {msg_id}")
                        continue
                    
                    # Parse the flags
                    flags_str = flags_data[0].decode("utf-8")
                    is_unread = "\\Seen" not in flags_str
                    logger.debug(f"Email {msg_id} unread status before fetch: {is_unread}")
                    
                    # Fetch the email - using BODY.PEEK to not set the \Seen flag
                    status, data = mail.fetch(email_id, "(BODY.PEEK[] ENVELOPE)")
                    if status != "OK":
                        logger.error(f"Failed to fetch email {msg_id}")
                        continue
                    
                    # Parse the email
                    raw_email = data[0][1]
                    msg = email.message_from_bytes(raw_email)
                    
                    # Extract headers
                    headers = {}
                    headers["from"] = str(decode_header(msg.get("From", ""))[0][0])
                    headers["to"] = str(decode_header(msg.get("To", ""))[0][0])
                    headers["subject"] = str(decode_header(msg.get("Subject", ""))[0][0])
                    headers["date"] = msg.get("Date", "")
                    
                    # Extract content
                    content = {"text": "", "html": ""}
                    
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(part.get("Content-Disposition"))
                            
                            # Skip attachments
                            if "attachment" in content_disposition:
                                continue
                            
                            # Get the content
                            try:
                                body = part.get_payload(decode=True)
                                if body:
                                    if content_type == "text/plain":
                                        content["text"] = body.decode("utf-8", errors="replace")
                                    elif content_type == "text/html":
                                        content["html"] = body.decode("utf-8", errors="replace")
                            except Exception as e:
                                logger.error(f"Error decoding email part: {e}")
                    else:
                        # Get the content type
                        content_type = msg.get_content_type()
                        
                        # Get the content
                        try:
                            body = msg.get_payload(decode=True)
                            if body:
                                if content_type == "text/plain":
                                    content["text"] = body.decode("utf-8", errors="replace")
                                elif content_type == "text/html":
                                    content["html"] = body.decode("utf-8", errors="replace")
                        except Exception as e:
                            logger.error(f"Error decoding email body: {e}")
                    
                    # Check unread status after fetch if not using readonly mode
                    if not PRESERVE_READ_STATUS:
                        status, flags_data = mail.fetch(email_id, '(FLAGS)')
                        if status == "OK":
                            flags_str = flags_data[0].decode("utf-8")
                            is_still_unread = "\\Seen" not in flags_str
                            logger.debug(f"Email {msg_id} unread status after fetch: {is_still_unread}")
                            
                            # Restore unread flag if it changed
                            if is_unread and not is_still_unread:
                                mail.store(email_id, '-FLAGS', '\\Seen')
                                logger.info(f"Restored unread status for email {msg_id}")
                    
                    # Save the original date string and timestamp for sorting
                    date_str = headers["date"]
                    date_obj = date_dict.get(email_id, datetime.min)
                    timestamp = int(date_obj.timestamp()) if date_obj != datetime.min else 0
                    
                    # Create email object
                    email_obj = {
                        "id": msg_id,
                        "thread_id": msg_id,  # Using msg_id as thread_id since IMAP doesn't have this concept
                        "label_ids": ["INBOX", "UNREAD"] if is_unread else ["INBOX"],
                        "snippet": content["text"][:100] if content["text"] else "",
                        "from": headers["from"],
                        "to": headers["to"],
                        "subject": headers["subject"],
                        "date": date_str,
                        "timestamp": timestamp,  # Store timestamp for better sorting
                        "text": content["text"],
                        "html": content["html"]
                    }
                    
                    # Add to cache
                    email_cache[msg_id] = email_obj
                    emails.append(email_obj)
                    
                    logger.info(f"Processed email: {headers['subject']} from {headers['from']} on {date_str}")
                    
                except Exception as e:
                    logger.error(f"Error processing email {email_id}: {e}")
            
            # Close the connection
            mail.close()
            mail.logout()
            
            # Save cache
            self.save_email_cache()
            
            return emails
        except Exception as e:
            logger.error(f"Error fetching emails: {e}")
            if 'mail' in locals() and mail:
                try:
                    mail.close()
                    mail.logout()
                except:
                    pass
            return []

    def load_email_cache(self):
        """Load email cache from disk."""
        global email_cache
        
        try:
            if os.path.exists(EMAIL_CACHE_FILE):
                with open(EMAIL_CACHE_FILE, "r") as f:
                    email_cache = json.load(f)
                logger.info(f"Loaded {len(email_cache)} emails from cache")
            else:
                logger.info(f"No email cache file found at {EMAIL_CACHE_FILE}")
        except Exception as e:
            logger.error(f"Error loading email cache: {e}")
            email_cache = {}

    def save_email_cache(self):
        """Save email cache to disk."""
        try:
            os.makedirs(os.path.dirname(EMAIL_CACHE_FILE), exist_ok=True)
            with open(EMAIL_CACHE_FILE, "w") as f:
                json.dump(email_cache, f)
            logger.info(f"Saved {len(email_cache)} emails to cache")
        except Exception as e:
            logger.error(f"Error saving email cache: {e}")

    def update_email_cache(self):
        """Periodically update email cache."""
        while True:
            try:
                logger.info("Updating email cache")
                self.fetch_emails()
                logger.info(f"Email cache updated, {len(email_cache)} emails in cache")
            except Exception as e:
                logger.error(f"Error updating email cache: {e}")
            
            # Sleep until next update
            logger.info(f"Sleeping for {EMAIL_FETCH_INTERVAL} seconds")
            time.sleep(EMAIL_FETCH_INTERVAL)

# Move Flask route outside of the class
@app.route("/api/emails", methods=["GET"])
def get_emails():
    """API endpoint to get emails."""
    try:
        limit = request.args.get("limit", default=50, type=int)
        emails = list(email_cache.values())
        
        # Sort by timestamp (newest first) or fallback to date string
        emails.sort(key=lambda x: x.get("timestamp", 0) or 0, reverse=True)
        
        # Remove internal timestamp field if it exists
        for email in emails:
            if "timestamp" in email:
                del email["timestamp"]
        
        return jsonify({"emails": emails[:limit]})
    except Exception as e:
        logger.error(f"Error in get_emails API: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Load existing email cache
    email_fetcher = EmailFetcher()
    email_fetcher.load_email_cache()
    
    # Start cache update thread
    import threading
    thread = threading.Thread(target=email_fetcher.update_email_cache, daemon=True)
    thread.start()
    
    # Start the Flask app
    logger.info("Starting email fetcher")
    app.run(host="0.0.0.0", port=5000)
