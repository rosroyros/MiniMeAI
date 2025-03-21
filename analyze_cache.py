import json
import sys

# Load the email cache
with open("config/email_cache.json", "r") as f:
    cache = json.load(f)

# Print basic stats
print(f"Total emails in cache: {len(cache)}")

# Print email IDs format
email_ids = list(cache.keys())
if email_ids:
    print(f"Sample email IDs: {email_ids[:5]}")

# Check for ID format
if "1335" in cache:
    print("Email with ID '1335' found")
else:
    print("Email with ID '1335' NOT found")
    
    # Look for similar IDs
    similar_ids = [id for id in email_ids if "1335" in id]
    if similar_ids:
        print(f"Similar IDs found: {similar_ids}")

# Check for Hebrew content
hebrew_emails = []
for email_id, email_data in cache.items():
    # Check if email contains Hebrew characters (Unicode range)
    text = email_data.get("text", "")
    if any(ord('\u0590') <= ord(c) <= ord('\u05FF') for c in text):
        hebrew_emails.append((email_id, email_data.get("subject", "No subject")))

if hebrew_emails:
    print(f"\nFound {len(hebrew_emails)} emails with Hebrew content:")
    for email_id, subject in hebrew_emails[:10]:  # Show first 10
        print(f"  ID: {email_id}, Subject: {subject}")

# Check if any emails have unusual structures or long content
for email_id, email_data in cache.items():
    text_len = len(email_data.get("text", ""))
    if text_len > 100000:  # Very long emails
        print(f"\nVery long email found: ID {email_id}, length: {text_len}")

