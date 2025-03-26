import os
import time
import logging
import json
import traceback
import uuid
import unicodedata
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pickle
import requests
import numpy as np
import openai
import schedule

# Direct import - no module path
from chroma_client import SimpleChromaClient
from timing import Timer, timed

# Import our new date utilities
from src.utils.date_utils import parse_timestamp, get_safe_timestamp, format_timestamp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/processor.log")
    ]
)
logger = logging.getLogger(__name__)

# Environment variables
DATA_API_HOST = os.getenv("DATA_API_HOST", "email_service")
DATA_API_PORT = int(os.getenv("DATA_API_PORT", "5000"))
VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "vector_db")
VECTOR_DB_PORT = int(os.getenv("VECTOR_DB_PORT", "8000"))
VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "messages_collection")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo" if LLM_PROVIDER == "openai" else "claude-instant-1")
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "/data/processed_messages.pickle")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
PROCESSING_INTERVAL = int(os.getenv("PROCESSING_INTERVAL", "300"))  # 5 minutes
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
# Add WhatsApp service environment variables
WHATSAPP_API_HOST = os.getenv("WHATSAPP_API_HOST", "whatsapp_bridge")
WHATSAPP_API_PORT = int(os.getenv("WHATSAPP_API_PORT", "3001"))

# Initialize OpenAI API for v0.28
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    logger.info("OpenAI API key configured")
else:
    logger.error("OpenAI API key not provided. Embeddings will fail.")


def load_processed_items() -> Dict[str, Any]:
    """Load processed items from disk."""
    try:
        if os.path.exists(PROCESSED_DATA_PATH):
            with open(PROCESSED_DATA_PATH, "rb") as f:
                return pickle.load(f)
        else:
            return {"item_ids": {}, "problematic_items": {}}
    except Exception as e:
        logger.error(f"Error loading processed items: {e}")
        logger.error(traceback.format_exc())
        return {"item_ids": {}, "problematic_items": {}}


def save_processed_items(data: Dict[str, Any]) -> bool:
    """Save processed items to disk."""
    try:
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        with open(PROCESSED_DATA_PATH, "wb") as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        logger.error(f"Error saving processed items: {e}")
        logger.error(traceback.format_exc())
        return False


def connect_to_chroma():
    """Connect to the ChromaDB server."""
    try:
        client = SimpleChromaClient(host=VECTOR_DB_HOST, port=VECTOR_DB_PORT)
        
        # Get or create collection
        try:
            collection = client.get_collection(name=VECTOR_COLLECTION_NAME)
            if not collection:
                collection = client.create_collection(name=VECTOR_COLLECTION_NAME)
        except Exception:
            collection = client.create_collection(name=VECTOR_COLLECTION_NAME)
        
        return collection
    except Exception as e:
        logger.error(f"Error connecting to ChromaDB: {e}")
        return None


def contains_hebrew(text: str) -> bool:
    """Check if text contains Hebrew characters."""
    hebrew_range = range(0x0590, 0x05FF)
    return any(ord(c) in hebrew_range for c in text)


def prepare_text_for_embedding(text: str) -> str:
    """Prepare text for embedding, handling special cases like Hebrew text.
    
    Note: This does NOT remove any Hebrew characters or content.
    It only performs safe normalization to ensure compatibility with the API.
    """
    # Normalize Unicode to ensure consistent representation
    # NFC = canonical composition - maintains complete characters
    normalized_text = unicodedata.normalize('NFC', text)
    
    # Remove zero-width spaces and joiners that can confuse the API
    # These are invisible characters that don't affect meaning
    for char in ['\u200B', '\u200C', '\u200D', '\u2060', '\uFEFF']:
        normalized_text = normalized_text.replace(char, '')
    
    # For Hebrew text, ensure proper right-to-left marks are present
    # This doesn't change content, just helps with proper display
    if contains_hebrew(normalized_text):
        logger.info("Processing text with Hebrew content")
        
    # Ensure the text is properly encoded as UTF-8
    # This is just a check - the text should already be in proper Python str format
    try:
        normalized_text.encode('utf-8').decode('utf-8')
    except UnicodeError:
        logger.warning("Text contains encoding issues, applying fallback normalization")
        # If there's an encoding issue, try a more aggressive normalization
        # that replaces problematic characters but preserves meaning
        normalized_text = ''.join(c for c in normalized_text if unicodedata.category(c)[0] != 'C')
    
    return normalized_text


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts using OpenAI API."""
    try:
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key not provided")
            return []
        
        # Process in batches to avoid timeouts and rate limits
        batch_size = 20
        all_embeddings = []
        
        # First, prepare all texts for embedding
        prepared_texts = [prepare_text_for_embedding(text) for text in texts if text and len(text.strip()) > 0]
        
        for i in range(0, len(prepared_texts), batch_size):
            batch_texts = prepared_texts[i:i + batch_size]
            logger.info(f"Getting embeddings for batch {i//batch_size + 1}/{(len(prepared_texts) + batch_size - 1)//batch_size}")
            
            try:
                # Log if the batch contains Hebrew
                has_hebrew = any(contains_hebrew(text) for text in batch_texts)
                if has_hebrew:
                    logger.info(f"Batch {i//batch_size + 1} contains Hebrew text")
                
                # Use the OpenAI Embeddings API with properly prepared texts
                response = openai.Embedding.create(
                    model=EMBEDDING_MODEL,
                    input=batch_texts
                )
                
                # Extract embeddings from response
                batch_embeddings = [item["embedding"] for item in response["data"]]
                all_embeddings.extend(batch_embeddings)
                
                # Avoid rate limits
                if i + batch_size < len(texts):
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error creating embeddings for batch {i//batch_size + 1}: {e}")
                # Enhanced error logging
                for j, text in enumerate(batch_texts):
                    idx = i + j
                    if idx < len(texts):
                        text_preview = texts[idx][:100].replace("\n", " ") + "..." if len(texts[idx]) > 100 else texts[idx]
                        has_hebrew = contains_hebrew(text)
                        logger.error(f"Problem may be in text {idx}: '{text_preview}' (Has Hebrew: {has_hebrew})")
                
                # Try a more granular approach - one text at a time
                single_embeddings = []
                for j, text in enumerate(batch_texts):
                    try:
                        logger.info(f"Trying individual embedding for text {i+j}")
                        resp = openai.Embedding.create(
                            model=EMBEDDING_MODEL,
                            input=[text]  # Pass as a single-item list
                        )
                        single_embeddings.append(resp["data"][0]["embedding"])
                        time.sleep(0.2)  # Short pause between requests
                    except Exception as inner_e:
                        logger.error(f"Error with individual text embedding {i+j}: {inner_e}")
                        # Add empty embedding as placeholder
                        embedding_dim = 1536  # Standard embedding dimension
                        single_embeddings.append([0.0] * embedding_dim)
                all_embeddings.extend(single_embeddings)
        
        return all_embeddings
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        logger.error(traceback.format_exc())
        return []


def chunk_text(text: str, source_type: str = "message", chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks of approximately chunk_size."""
    if not text:
        return []
    
    # Determine if text contains Hebrew to handle right-to-left text appropriately
    has_hebrew = contains_hebrew(text)
    if has_hebrew:
        logger.info("Chunking text with Hebrew content")
    
    # Use different splitting strategies based on source type
    if source_type == "whatsapp":
        # WhatsApp messages are often naturally chunked by message
        chunks = split_by_messages(text, chunk_size)
    elif source_type == "email":
        # Emails often have structured paragraphs
        chunks = split_by_paragraphs(text, chunk_size, has_hebrew)
    else:
        # Default strategy for other message types
        chunks = split_by_paragraphs(text, chunk_size, has_hebrew)
    
    # Ensure all chunks are prepared for embedding
    prepared_chunks = [prepare_text_for_embedding(chunk) for chunk in chunks]
    
    # Check for very long chunks and split them further if needed
    max_chunk_size = 8000  # OpenAI has a token limit
    final_chunks = []
    for chunk in prepared_chunks:
        if len(chunk) > max_chunk_size:
            logger.warning(f"Very long chunk detected ({len(chunk)} chars), splitting further")
            # Simple splitting for overly long chunks
            for i in range(0, len(chunk), max_chunk_size // 2):
                final_chunks.append(chunk[i:i + max_chunk_size // 2])
        else:
            final_chunks.append(chunk)
    
    return final_chunks


def split_by_messages(text: str, chunk_size: int) -> List[str]:
    """Split text by message boundaries, optimized for chat-like content."""
    # Look for common message patterns like timestamps and sender names
    import re
    
    # Define patterns for different message formats
    whatsapp_pattern = r'\[\d{1,2}/\d{1,2}/\d{2,4},? \d{1,2}:\d{2}(?::\d{2})?(?: [AP]M)?\] [^:]+: '
    
    # Try to split by WhatsApp pattern
    messages = re.split(whatsapp_pattern, text)
    message_headers = re.findall(whatsapp_pattern, text)
    
    # Reconstruct messages with their headers
    complete_messages = []
    for i, message in enumerate(messages):
        if i == 0 and not message.strip():
            # Skip empty first message (happens when text starts with a header)
            continue
        
        header = message_headers[i-1] if i > 0 and i-1 < len(message_headers) else ""
        complete_messages.append(header + message)
    
    # If no WhatsApp patterns found, fallback to line splits
    if len(complete_messages) <= 1:
        lines = text.split('\n')
        chunks = []
        current_chunk = ""
        
        for line in lines:
            if len(current_chunk) + len(line) + 1 <= chunk_size:
                current_chunk += line + "\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    # Group messages into chunks
    chunks = []
    current_chunk = ""
    
    for message in complete_messages:
        if len(current_chunk) + len(message) + 1 <= chunk_size:
            current_chunk += message + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = message + "\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def split_by_paragraphs(text: str, chunk_size: int, has_hebrew: bool) -> List[str]:
    """Split text by paragraphs, with special handling for Hebrew."""
    chunks = []
    paragraphs = text.split("\n\n")
    
    current_chunk = ""
    for paragraph in paragraphs:
        # For very long paragraphs, use different splitting strategy
        if len(paragraph) > chunk_size * 1.5:
            # For Hebrew text, be more careful with sentence splitting
            if has_hebrew:
                # Hebrew uses different punctuation patterns
                sentences = []
                current_sentence = ""
                for char in paragraph:
                    current_sentence += char
                    # Hebrew sentence can end with period, question mark, exclamation mark
                    if char in ['.', '?', '!', '׃', '־'] and len(current_sentence) > 10:
                        sentences.append(current_sentence)
                        current_sentence = ""
                if current_sentence:  # Add any remaining text
                    sentences.append(current_sentence)
            else:
                # Standard sentence splitting for non-Hebrew
                sentences = paragraph.split(". ")
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += sentence + (" " if sentence.endswith(".") else ". ")
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + (" " if sentence.endswith(".") else ". ")
        elif len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def extract_content_and_metadata(item: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """Extract content and metadata from different types of messages."""
    source_type = item.get("source_type", "message")
    item_id = item.get("id", "unknown")
    content = ""
    metadata = {}
    
    try:
        # Extract based on source type
        if source_type == "email":
            # Extract email content
            if "text" in item and item["text"]:
                content = item["text"]
            elif "html" in item and item["html"]:
                # Simple HTML stripping
                content = item["html"]
                content = content.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
                content = content.replace("<p>", "\n").replace("</p>", "\n")
                content = content.replace("<div>", "\n").replace("</div>", "\n")
                content = content.replace("<li>", "\n- ").replace("</li>", "\n")
                
                # Strip remaining HTML tags
                import re
                content = re.sub(r"<[^>]*>", "", content)
            
            # Extract email metadata
            metadata = {
                "id": item_id,
                "source_type": "email",
                "subject": item.get("subject", "No Subject"),
                "from": item.get("from", "Unknown"),
                "to": item.get("to", "Unknown"),
                "date": item.get("date", "Unknown"),
                "timestamp": get_timestamp_from_date(item.get("date", "")),
                "has_hebrew": contains_hebrew(item.get("subject", "")) or contains_hebrew(content)
            }
            
            # Add subject to the content for better context
            title = f"Subject: {metadata['subject']}"
        
        elif source_type == "whatsapp":
            # Extract WhatsApp content
            if "text" in item:
                content = item["text"]
            
            # Extract WhatsApp metadata
            metadata = {
                "id": item_id,
                "source_type": "whatsapp",
                "sender": item.get("sender", "Unknown"),
                "chat": item.get("chat", "Unknown"),
                "date": item.get("date", "Unknown"),
                "timestamp": get_timestamp_from_date(item.get("date", "")),
                "has_hebrew": contains_hebrew(content)
            }
            
            # Add chat name to the content for better context
            title = f"WhatsApp Chat: {metadata['chat']}"
        
        elif source_type == "text":
            # Extract text message content
            if "text" in item:
                content = item["text"]
            
            # Extract text message metadata
            metadata = {
                "id": item_id,
                "source_type": "text",
                "sender": item.get("sender", "Unknown"),
                "recipient": item.get("recipient", "Unknown"),
                "date": item.get("date", "Unknown"),
                "timestamp": get_timestamp_from_date(item.get("date", "")),
                "has_hebrew": contains_hebrew(content)
            }
            
            # Add sender to the content for better context
            title = f"Text Message from: {metadata['sender']}"
        
        else:
            # Generic fallback for unknown types
            if "text" in item:
                content = item["text"]
            elif "content" in item:
                content = item["content"]
            else:
                content = str(item)
            
            # Extract generic metadata
            metadata = {
                "id": item_id,
                "source_type": "message",
                "date": item.get("date", "Unknown"),
                "timestamp": get_timestamp_from_date(item.get("date", "")),
                "has_hebrew": contains_hebrew(content)
            }
            
            # Generic title
            title = f"Message ID: {item_id}"
        
        # Add timestamp if not present
        if "timestamp" not in metadata or not metadata["timestamp"]:
            metadata["timestamp"] = int(time.time())
        
        return content, metadata, title
    
    except Exception as e:
        logger.error(f"Error extracting content from {source_type} {item_id}: {e}")
        return "", {"id": item_id, "source_type": source_type, "error": str(e)}, None


def get_timestamp_from_date(date_str: str) -> Optional[int]:
    """Convert various date formats to a timestamp."""
    return parse_timestamp(date_str)


def fetch_new_items() -> List[Dict[str, Any]]:
    """Fetch new messages from the data service."""
    try:
        # Load previously processed item IDs
        processed_data = load_processed_items()
        processed_ids = processed_data.get("item_ids", {})
        
        # Fetch from multiple data sources
        all_items = []
        
        # Fetch emails with retry logic
        retry_count = 0
        max_retries = 3
        retry_delay = 5  # seconds
        
        while retry_count < max_retries:
            try:
                response = requests.get(
                    f"http://{DATA_API_HOST}:{DATA_API_PORT}/api/emails",
                    params={"limit": 50},
                    timeout=10
                )
                
                if response.status_code == 200:
                    emails = response.json().get("emails", [])
                    logger.info(f"Fetched {len(emails)} emails from data service")
                    
                    for email in emails:
                        if email.get("id") and email.get("id") not in processed_ids.get("email", []):
                            email["source_type"] = "email"
                            all_items.append(email)
                    break  # Success, exit retry loop
                else:
                    logger.warning(f"Failed to fetch emails: {response.status_code}")
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"Retrying in {retry_delay} seconds... ({retry_count}/{max_retries})")
                        time.sleep(retry_delay)
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching emails: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds... ({retry_count}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Max retries exceeded when fetching emails: {e}")
        
        # Fetch WhatsApp messages (if available)
        try:
            logger.info(f"Attempting to fetch WhatsApp messages from {WHATSAPP_API_HOST}:{WHATSAPP_API_PORT}")
            response = requests.get(
                f"http://{WHATSAPP_API_HOST}:{WHATSAPP_API_PORT}/api/whatsapp",
                params={"limit": 50},
                timeout=5
            )
            
            if response.status_code == 200:
                messages = response.json().get("messages", [])
                logger.info(f"Fetched {len(messages)} WhatsApp messages from WhatsApp bridge")
                
                for message in messages:
                    if message.get("id") and message.get("id") not in processed_ids.get("whatsapp", []):
                        # Message is already properly formatted with source_type='whatsapp'
                        all_items.append(message)
            else:
                # WhatsApp endpoint might not be available yet
                logger.warning(f"WhatsApp bridge returned: {response.status_code}, response: {response.text[:100]}")
        except Exception as e:
            # Don't retry for WhatsApp since it's optional
            logger.warning(f"Error fetching WhatsApp messages: {e}")
        
        # Fetch text messages (if available)
        try:
            response = requests.get(
                f"http://{DATA_API_HOST}:{DATA_API_PORT}/api/texts",
                params={"limit": 50},
                timeout=5
            )
            
            if response.status_code == 200:
                texts = response.json().get("messages", [])
                logger.info(f"Fetched {len(texts)} text messages from data service")
                
                for text in texts:
                    if text.get("id") and text.get("id") not in processed_ids.get("text", []):
                        text["source_type"] = "text"
                        all_items.append(text)
            else:
                # Text messages endpoint might not be available yet
                logger.debug(f"Text messages endpoint returned: {response.status_code}")
        except Exception as e:
            # Don't retry for text messages since it's optional
            logger.debug(f"Error fetching text messages (may not be implemented yet): {e}")
        
        logger.info(f"Found {len(all_items)} new items to process")
        return all_items
    except Exception as e:
        logger.error(f"Error fetching new items: {e}")
        logger.error(traceback.format_exc())
        return []


def process_new_items():
    """Process new messages and store in vector database."""
    try:
        # Connect to ChromaDB
        collection = connect_to_chroma()
        if not collection:
            logger.error("Failed to connect to vector database")
            return
        
        # Fetch new items
        new_items = fetch_new_items()
        if not new_items:
            logger.info("No new items to process")
            return
        
        # Process items
        processed_data = load_processed_items()
        processed_ids = processed_data.get("item_ids", {})
        problematic_items = processed_data.get("problematic_items", {})
        
        for item in new_items:
            try:
                item_id = item.get("id")
                source_type = item.get("source_type", "message")
                
                # Initialize the source type in processed_ids if not exists
                if source_type not in processed_ids:
                    processed_ids[source_type] = []
                
                # Skip if already processed
                if item_id in processed_ids[source_type]:
                    continue
                
                # Extract content and metadata
                content, metadata, title = extract_content_and_metadata(item)
                
                if not content:
                    logger.warning(f"No content found for {source_type} {item_id}")
                    continue
                
                # Add title to the content for better context
                if title:
                    full_content = f"{title}\n\n{content}"
                else:
                    full_content = content
                
                # Check content length for very long messages
                content_length = len(full_content)
                if content_length > 100000:
                    logger.warning(f"Very long {source_type} detected: {item_id} with {content_length} characters")
                
                # Chunk content with source-specific chunking
                chunks = chunk_text(full_content, source_type)
                if not chunks:
                    logger.warning(f"No chunks created for {source_type} {item_id}")
                    continue
                
                logger.info(f"Created {len(chunks)} chunks for {source_type} {item_id}")
                
                # Create embeddings
                embeddings = get_embeddings(chunks)
                if not embeddings:
                    logger.error(f"Failed to create any embeddings for {source_type} {item_id}")
                    # Track as problematic but don't stop processing
                    problematic_items[f"{source_type}_{item_id}"] = {
                        "id": item_id,
                        "source_type": source_type,
                        "reason": "Failed to create embeddings",
                        "has_hebrew": metadata.get("has_hebrew", False)
                    }
                    continue
                
                if len(embeddings) != len(chunks):
                    logger.warning(f"Mismatched number of embeddings ({len(embeddings)}) and chunks ({len(chunks)}) for {source_type} {item_id}")
                    # Adjust to match
                    if len(embeddings) < len(chunks):
                        chunks = chunks[:len(embeddings)]
                    else:
                        embeddings = embeddings[:len(chunks)]
                
                # Create metadata for each chunk
                metadatas = []
                for i in range(len(chunks)):
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_id"] = i
                    metadatas.append(chunk_metadata)
                
                # Create IDs
                ids = [f"{source_type}_{item_id}_{i}" for i in range(len(chunks))]
                
                # Add to ChromaDB
                collection.add(
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                
                # Mark as processed
                processed_ids[source_type].append(item_id)
                logger.info(f"Successfully processed {source_type} {item_id}")
                
            except Exception as e:
                item_id = item.get("id", "unknown")
                source_type = item.get("source_type", "message")
                logger.error(f"Error processing {source_type} {item_id}: {e}")
                logger.error(traceback.format_exc())
                
                # Track problematic item
                problematic_items[f"{source_type}_{item_id}"] = {
                    "id": item_id,
                    "source_type": source_type,
                    "reason": str(e)
                }
        
        # Save processed item IDs and problematic items
        processed_data["item_ids"] = processed_ids
        processed_data["problematic_items"] = problematic_items
        save_processed_items(processed_data)
        
        # Log summary
        total_processed = sum(len(ids) for ids in processed_ids.values())
        logger.info(f"Total processed items: {total_processed}")
        if problematic_items:
            logger.warning(f"Encountered issues with {len(problematic_items)} items")
        
    except Exception as e:
        logger.error(f"Error in process_new_items: {e}")
        logger.error(traceback.format_exc())


def main():
    """Main processing loop."""
    logger.info("Starting message processor")
    
    # Wait for dependencies to be ready
    time.sleep(10)
    
    while True:
        try:
            logger.info("Processing new messages")
            process_new_items()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            logger.error(traceback.format_exc())
        
        logger.info(f"Sleeping for {PROCESSING_INTERVAL} seconds")
        time.sleep(PROCESSING_INTERVAL)


if __name__ == "__main__":
    main()

@timed
def generate_embedding(text):
    """Generate embedding for a single text string."""
    try:
        prepared_text = prepare_text_for_embedding(text)
        response = openai.Embedding.create(
            model=EMBEDDING_MODEL,
            input=[prepared_text]
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return []

@timed
def generate_response(query, search_results):
    """Generate an LLM response based on the query and search results."""
    with Timer("LLM prompt construction"):
        # Construct prompt
        context = ""
        for i, result in enumerate(search_results):
            context += f"\n--- Result {i+1} ---\n"
            context += f"Content: {result['document']}\n"
            if 'metadata' in result:
                for key, value in result['metadata'].items():
                    context += f"{key}: {value}\n"
        
        prompt = f"""You are MiniMe AI, a helpful assistant that answers questions based on the user's communications.
        
Below is information from the user's messages that might be relevant to their query:

{context}

Based on the information above, please answer the user's query in a detailed, helpful manner:
{query}
"""
    
    with Timer("LLM API call"):
        # Make API call to LLM
        try:
            if LLM_PROVIDER == "openai":
                response = openai.ChatCompletion.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are MiniMe AI, a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.5
                )
                return response.choices[0].message.content
            else:
                return "LLM provider not configured correctly."
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return f"Error generating response: {str(e)}"
