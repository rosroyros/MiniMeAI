#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMAP email fetcher for retrieving emails from various providers.
"""
import os
import time
import imaplib
import email
import email.utils
import logging
import json
import re
import ssl
from datetime import datetime, timedelta
from email.header import decode_header
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from flask import Flask, jsonify, request
from flask_cors import CORS

# Import BaseFetcher
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.base_fetcher import BaseFetcher
from utils.timing import Timer, timed
# Import the new date utilities
from utils.date_utils import parse_timestamp, get_safe_timestamp, format_timestamp

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

class EmailFetcher(BaseFetcher):
    """Fetcher for retrieving emails from IMAP servers."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the email fetcher.
        
        Args:
            config: Configuration for the email fetcher
        """
        super().__init__(config)
        
        # Load configuration
        self.config = config or {}
        self.email_server = self.config.get("imap_server", "imap.gmail.com")
        self.email_port = self.config.get("imap_port", 993)
        self.email_user = self.config.get("email_user", os.environ.get("EMAIL_USER", ""))
        self.email_password = self.config.get("email_password", os.environ.get("EMAIL_PASSWORD", ""))
        self.email_folder = self.config.get("email_folder", "INBOX")
        self.cache_path = self.config.get("cache_path", "config/email_cache.json")
        self.max_emails = self.config.get("max_emails", 100)
        self.max_days = self.config.get("max_days", 30)
        self.logger = logging.getLogger("email_fetcher")
        
        # Initialize cache
        self.initialize_cache()
        
    def connect_to_imap(self):
        """Connect to IMAP server."""
        try:
            # Connect to the IMAP server
            mail = imaplib.IMAP4_SSL(self.email_server, self.email_port)
            
            # Login
            mail.login(self.email_user, self.email_password)
            
            return mail
        except Exception as e:
            self.logger.error(f"Error connecting to IMAP server: {e}")
            return None

    @timed
    def fetch_emails(self, limit=EMAIL_FETCH_COUNT) -> List[Dict[str, Any]]:
        """Fetch emails from IMAP server while preserving unread status."""
        try:
            # Connect to IMAP
            mail = self.connect_to_imap()
            if not mail:
                self.logger.error("Failed to connect to IMAP server")
                return []
            
            # Select the mailbox (INBOX)
            mail.select("INBOX", readonly=PRESERVE_READ_STATUS)  # Set readonly=True to preserve flags
            
            # Calculate date filter for sync period
            sync_date = (datetime.now() - timedelta(days=EMAIL_SYNC_DAYS)).strftime("%d-%b-%Y")
            search_criteria = f'(SINCE "{sync_date}")'
            
            self.logger.info(f"Searching for emails since {sync_date}")
            
            # Search for emails
            self.logger.info(f"Searching with criteria: {search_criteria}")
            status, messages = mail.search(None, search_criteria)
            if status != "OK":
                self.logger.error(f"Failed to search emails: {status}")
                return []
            # Get email IDs
            email_ids = messages[0].split()
            if not email_ids:
                self.logger.info("No emails found matching criteria")
                return []
            self.logger.info(f"Found {len(email_ids)} emails matching search criteria")
            
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
                            self.logger.error(f"Error parsing date {date_str}: {e}")
                            date_dict[email_id] = datetime.min
                except Exception as e:
                    self.logger.error(f"Error fetching date for email {email_id}: {e}")
                    date_dict[email_id] = datetime.min

            # Sort emails by date (newest first)
            sorted_ids = sorted(date_dict.keys(), key=lambda x: date_dict[x], reverse=True)
            
            # Limit to the specified number of emails
            email_ids = sorted_ids[:limit]
            
            self.logger.info(f"Processing {len(email_ids)} emails, newest first")
            
            # Fetch each email
            emails = []
            for email_id in email_ids:
                try:
                    # Convert ID to string
                    msg_id = email_id.decode("utf-8")
                    
                    # Check if the message is unread before fetching
                    status, flags_data = mail.fetch(email_id, '(FLAGS)')
                    if status != "OK":
                        self.logger.error(f"Failed to fetch flags for email {msg_id}")
                        continue
                    
                    # Parse the flags
                    flags_str = flags_data[0].decode("utf-8")
                    is_unread = "\\Seen" not in flags_str
                    self.logger.debug(f"Email {msg_id} unread status before fetch: {is_unread}")
                    
                    # Fetch the email - using BODY.PEEK to not set the \Seen flag
                    status, data = mail.fetch(email_id, "(BODY.PEEK[] ENVELOPE)")
                    if status != "OK":
                        self.logger.error(f"Failed to fetch email {msg_id}")
                        continue
                    
                    # Parse the email
                    raw_email = data[0][1]
                    msg = email.message_from_bytes(raw_email)
                    
                    # Extract message ID from headers for better deduplication
                    message_id = msg.get("Message-ID", msg_id)
                    
                    # Check if email is already in cache by Message-ID
                    duplicate_found = False
                    if message_id in email_cache:
                        emails.append(email_cache[message_id])
                        duplicate_found = True
                        continue
                      
                    # Also check for duplicates by comparing subjects and dates
                    if not duplicate_found:
                        subject = str(decode_header(msg.get("Subject", ""))[0][0])
                        date_str = msg.get("Date", "")
                        
                        # Look for matching subject+date in cache
                        for cached_id, cached_email in email_cache.items():
                            if (cached_email.get("subject") == subject and 
                                cached_email.get("date") == date_str):
                                emails.append(cached_email)
                                duplicate_found = True
                                break
                    
                    if duplicate_found:
                        continue
                    
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
                                self.logger.error(f"Error decoding email part: {e}")
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
                            self.logger.error(f"Error decoding email body: {e}")
                    
                    # Check unread status after fetch if not using readonly mode
                    if not PRESERVE_READ_STATUS:
                        status, flags_data = mail.fetch(email_id, '(FLAGS)')
                        if status == "OK":
                            flags_str = flags_data[0].decode("utf-8")
                            is_still_unread = "\\Seen" not in flags_str
                            self.logger.debug(f"Email {msg_id} unread status after fetch: {is_still_unread}")
                            
                            # Restore unread flag if it changed
                            if is_unread and not is_still_unread:
                                mail.store(email_id, '-FLAGS', '\\Seen')
                                self.logger.info(f"Restored unread status for email {msg_id}")
                    
                    # Save the original date string and timestamp for sorting
                    date_str = headers["date"]
                    date_obj = date_dict.get(email_id, datetime.min)
                    timestamp = int(date_obj.timestamp()) if date_obj != datetime.min else 0
                    
                    # Create email object
                    email_obj = {
                        "id": msg_id,
                        "message_id": message_id,
                        "thread_id": msg_id,  # Using msg_id as thread_id since IMAP doesn't have this concept
                        "label_ids": ["INBOX", "UNREAD"] if is_unread else ["INBOX"],
                        "snippet": content["text"][:100] if content["text"] else "",
                        "from": headers["from"],
                        "to": headers["to"],
                        "subject": headers["subject"],
                        "date": date_str,
                        "text": content["text"],
                        "html": content["html"]
                    }
                    
                    # Use base class to enrich with metadata (adds source_type and timestamp)
                    email_obj = self.enrich_metadata(email_obj)
                    
                    # Add to cache - use Message-ID when available for better deduplication
                    cache_key = message_id if message_id != msg_id else msg_id
                    email_cache[cache_key] = email_obj
                    emails.append(email_obj)
                    
                    self.logger.info(f"Processed email: {headers['subject']} from {headers['from']} on {date_str}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing email {email_id}: {e}")
            
            # Close the connection
            mail.close()
            mail.logout()
            
            # Save cache
            self.save_email_cache()
            
            return emails
        except Exception as e:
            self.logger.error(f"Error fetching emails: {e}")
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
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r") as f:
                    email_cache = json.load(f)
                self.logger.info(f"Loaded {len(email_cache)} emails from cache")
            else:
                self.logger.info(f"No email cache file found at {self.cache_path}")
        except Exception as e:
            self.logger.error(f"Error loading email cache: {e}")
            email_cache = {}

    def save_email_cache(self):
        """Save email cache to disk."""
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(email_cache, f)
            self.logger.info(f"Saved {len(email_cache)} emails to cache")
        except Exception as e:
            self.logger.error(f"Error saving email cache: {e}")

    def update_email_cache(self):
        """Periodically update email cache."""
        while True:
            try:
                self.logger.info("Updating email cache")
                initial_count = len(email_cache)
                self.fetch_emails()
                final_count = len(email_cache)
                new_emails = final_count - initial_count
                self.logger.info(f"Email cache updated, {len(email_cache)} emails in cache (added {new_emails} new emails)")
            except Exception as e:
                self.logger.error(f"Error updating email cache: {e}")
            
            # Sleep until next update
            self.logger.info(f"Sleeping for {EMAIL_FETCH_INTERVAL} seconds")
            time.sleep(EMAIL_FETCH_INTERVAL)

    # Implement the abstract method from BaseFetcher
    def fetch_data(self, limit: int = EMAIL_FETCH_COUNT) -> List[Dict[str, Any]]:
        """Implement the abstract method from BaseFetcher."""
        return self.fetch_emails(limit=limit)

# Move Flask route outside of the class
@app.route("/api/emails", methods=["GET"])
def get_emails():
    """API endpoint to get emails."""
    try:
        limit = request.args.get("limit", default=50, type=int)
        emails = list(email_cache.values())
        
        # Sort by timestamp (newest first) or fallback to date string
        emails.sort(key=lambda x: x.get("timestamp", 0) or 0, reverse=True)
        
        # No longer removing timestamp field to ensure proper processing
        # We need this field for proper sorting and processing by the processor service
        
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
