#!/usr/bin/env python3
"""
Management script for WhatsApp integration.
Provides command-line utilities for managing the WhatsApp integration.
"""

import argparse
import json
import os
import requests
import shutil
import subprocess
import sys
from datetime import datetime

def check_status():
    """Check the status of the WhatsApp bridge."""
    try:
        response = requests.get("http://localhost:3001/api/status")
        if response.status_code == 200:
            status = response.json()
            print(f"WhatsApp Bridge Status: {status['status']}")
            print(f"Authenticated: {status['authenticated']}")
            return True
        else:
            print(f"Error: Received status code {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Error connecting to WhatsApp bridge: {e}")
        return False

def send_message(to, message):
    """Send a message via WhatsApp."""
    try:
        response = requests.post(
            "http://localhost:3001/api/send",
            json={"to": to, "message": message}
        )
        if response.status_code == 200:
            print(f"Message sent successfully to {to}")
            return True
        else:
            print(f"Error sending message: {response.status_code}")
            print(response.text)
            return False
    except requests.RequestException as e:
        print(f"Error connecting to WhatsApp bridge: {e}")
        return False

def reset_session():
    """Reset the WhatsApp session data."""
    session_dir = os.environ.get("WHATSAPP_SESSION_DIR", "src/whatsapp/.wwebjs_auth")
    
    if not os.path.exists(session_dir):
        print(f"Session directory {session_dir} does not exist")
        return False
    
    # Create a backup before removing
    backup_dir = f"{session_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        shutil.copytree(session_dir, backup_dir)
        print(f"Created backup of session data at {backup_dir}")
        
        # Remove the session data
        shutil.rmtree(session_dir)
        print(f"Removed session data from {session_dir}")
        
        # Recreate the empty directory
        os.makedirs(session_dir, exist_ok=True)
        print("Created new empty session directory")
        
        print("WhatsApp session has been reset. You'll need to scan the QR code again.")
        return True
    except Exception as e:
        print(f"Error resetting session: {e}")
        return False

def restart_service():
    """Restart the WhatsApp bridge service."""
    try:
        print("Restarting WhatsApp bridge service...")
        subprocess.run(["docker-compose", "restart", "whatsapp_bridge"], check=True)
        print("WhatsApp bridge service restarted")
        return True
    except subprocess.SubprocessError as e:
        print(f"Error restarting service: {e}")
        return False

def show_logs(tail=100):
    """Show the logs for the WhatsApp bridge service."""
    try:
        print(f"Showing last {tail} lines of logs:")
        subprocess.run(["docker-compose", "logs", "--tail", str(tail), "whatsapp_bridge"], check=True)
        return True
    except subprocess.SubprocessError as e:
        print(f"Error showing logs: {e}")
        return False

def follow_logs():
    """Follow the logs for the WhatsApp bridge service."""
    try:
        print("Following logs (press Ctrl+C to stop):")
        subprocess.run(["docker-compose", "logs", "--follow", "whatsapp_bridge"], check=True)
        return True
    except subprocess.SubprocessError as e:
        print(f"Error following logs: {e}")
        return False
    except KeyboardInterrupt:
        print("\nStopped following logs")
        return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Manage WhatsApp integration for MiniMeAI")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check WhatsApp bridge status")
    
    # Send message command
    send_parser = subparsers.add_parser("send", help="Send a WhatsApp message")
    send_parser.add_argument("to", help="Recipient phone number (format: 1234567890@c.us)")
    send_parser.add_argument("message", help="Message text")
    
    # Reset session command
    reset_parser = subparsers.add_parser("reset", help="Reset WhatsApp session data")
    
    # Restart service command
    restart_parser = subparsers.add_parser("restart", help="Restart WhatsApp bridge service")
    
    # Show logs command
    logs_parser = subparsers.add_parser("logs", help="Show WhatsApp bridge logs")
    logs_parser.add_argument("--tail", type=int, default=100, help="Number of log lines to show")
    logs_parser.add_argument("--follow", action="store_true", help="Follow the logs")
    
    args = parser.parse_args()
    
    if args.command == "status":
        check_status()
    elif args.command == "send":
        send_message(args.to, args.message)
    elif args.command == "reset":
        reset_session()
    elif args.command == "restart":
        restart_service()
    elif args.command == "logs":
        if args.follow:
            follow_logs()
        else:
            show_logs(args.tail)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 