import os
import time
import json
import logging
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import anthropic
from typing import List, Dict, Any, Optional, Tuple

# Direct import - no module path
from chroma_client import SimpleChromaClient
from timing import Timer, timed
from logging_config import setup_logging

# Initialize logging
logger = setup_logging()

# Environment variables
VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "vector_db")
VECTOR_DB_PORT = int(os.getenv("VECTOR_DB_PORT", "8000"))
VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "emails")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo" if LLM_PROVIDER == "openai" else "claude-instant-1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set up OpenAI API for v0.28
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    logger.info("OpenAI client initialized for embeddings and LLM")
else:
    logger.error("OpenAI API key not provided")

# Set up LLM client based on provider
if LLM_PROVIDER == "openai":
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key not provided")
elif LLM_PROVIDER == "anthropic":
    if not ANTHROPIC_API_KEY:
        logger.error("Anthropic API key not provided")
    else:
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("Anthropic client initialized")
else:
    logger.error(f"Unsupported LLM provider: {LLM_PROVIDER}")


def connect_to_chroma() -> Optional[Any]:
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
        logger.error(traceback.format_exc())
        return None


def get_embeddings(text: str) -> List[float]:
    """Get embeddings for text using OpenAI API."""
    try:
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key not provided")
            return []
        
        # Use the OpenAI Embeddings API with v0.28 syntax
        response = openai.Embedding.create(
            model=EMBEDDING_MODEL,
            input=[text]
        )
        
        # Extract embeddings from response
        embedding = response["data"][0]["embedding"]
        return embedding
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        logger.error(traceback.format_exc())
        return []


def detect_language(text: str) -> str:
    """Detect the language of the text."""
    # Check for Hebrew characters
    hebrew_range = range(0x0590, 0x05FF)
    is_hebrew = any(ord(c) in hebrew_range for c in text)
    
    if is_hebrew:
        return "he"
    
    # Default to English
    return "en"


def get_reliable_timestamp(date_str: str) -> int:
    """Convert various date formats to a reliable timestamp."""
    if not date_str:
        return 0
    
    # Try parsing with email.utils first (most reliable for email dates)
    try:
        import email.utils
        date_tuple = email.utils.parsedate_tz(date_str)
        if date_tuple:
            return int(email.utils.mktime_tz(date_tuple))
    except Exception:
        pass
    
    # Try common date formats
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",  # RFC 2822 format
        "%Y-%m-%dT%H:%M:%S%z",       # ISO 8601
        "%Y-%m-%d %H:%M:%S",         # SQL-like
        "%d/%m/%Y %H:%M:%S",         # Common format
        "%d/%m/%Y %H:%M",            # Common format without seconds
        "%d.%m.%Y %H:%M:%S",         # European format
        "%d.%m.%Y %H:%M",            # European format without seconds
    ]
    
    from datetime import datetime
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return int(dt.timestamp())
        except ValueError:
            continue
    
    # If all else fails, return current time - very old (to put it at the bottom)
    logger.warning(f"Could not parse date: {date_str}")
    from datetime import datetime
    return int(datetime.now().timestamp()) - 31536000  # One year ago


def is_recency_query(query: str, language: str = "en") -> bool:
    """Enhanced detection of recency-based queries."""
    query = query.lower() if language == "en" else query
    
    # Expanded English recency terms
    recency_terms_en = [
        "last", "latest", "recent", "newest", "most recent", 
        "just", "new", "current", "today", "tonight", "this morning",
        "this afternoon", "this evening", "now", "fresh"
    ]
    
    # Hebrew recency terms (keep existing ones)
    recency_terms_he = [
        "אחרון", "אחרונה", "האחרון", "האחרונה", "עדכני", 
        "עדכנית", "היום", "הבוקר", "הערב", "עכשיו"
    ]
    
    # Time-based query patterns (English)
    time_patterns_en = [
        "today", "this week", "this month", "this morning", 
        "this afternoon", "this evening", "just now",
        "in the last hour", "in the last day"
    ]
    
    selected_terms = recency_terms_en if language == "en" else recency_terms_he
    selected_patterns = time_patterns_en if language == "en" else []
    
    # Check for recency terms
    for term in selected_terms:
        if term in query:
            logger.info(f"Recency term matched: '{term}' in '{query}'")
            return True
    
    # Check for time-based patterns
    for pattern in selected_patterns:
        if pattern in query:
            logger.info(f"Time pattern matched: '{pattern}' in '{query}'")
            return True
    
    return False


def apply_recency_sorting(documents, metadatas):
    """Apply enhanced recency-based sorting to retrieved documents."""
    logger.info("Applying enhanced recency-based sorting")
    
    # Create a list of document-metadata pairs
    doc_meta_pairs = list(zip(documents, metadatas))
    
    # Log original timestamps for debugging
    timestamps = [meta.get("timestamp", 0) for meta in metadatas]
    date_strings = [meta.get("date", "") for meta in metadatas]
    logger.info(f"Original timestamps: {timestamps}")
    logger.info(f"Date strings: {date_strings}")
    
    # Convert date strings to reliable timestamps if timestamp is missing or zero
    for i, (doc, meta) in enumerate(doc_meta_pairs):
        timestamp = meta.get("timestamp", 0)
        if not timestamp or timestamp == 0:
            date_str = meta.get("date", "")
            if date_str:
                # Use our reliable timestamp function
                timestamp = get_reliable_timestamp(date_str)
                meta["timestamp"] = timestamp
                logger.info(f"Generated timestamp {timestamp} for date: {date_str}")
    
    # Sort by date if possible (newest first)
    try:
        # Updated sorting that handles missing timestamps better
        doc_meta_pairs.sort(
            key=lambda x: x[1].get("timestamp", 0) 
            if isinstance(x[1].get("timestamp"), (int, float)) and x[1].get("timestamp", 0) > 0
            else get_reliable_timestamp(x[1].get("date", "")), 
            reverse=True
        )
    except Exception as e:
        logger.error(f"Error sorting by date: {e}")
    
    # Log new timestamps for debugging
    new_timestamps = [meta.get("timestamp", 0) for _, meta in doc_meta_pairs]
    logger.info(f"Sorted timestamps: {new_timestamps}")
    
    # Reconstruct sorted lists
    sorted_documents = [doc for doc, _ in doc_meta_pairs]
    sorted_metadatas = [meta for _, meta in doc_meta_pairs]
    
    return sorted_documents, sorted_metadatas


def query_llm(query: str, context: str) -> str:
    """Query LLM based on the provider, responding in the same language as the query."""
    # Detect query language
    language = detect_language(query)
    
    # Create language-specific prompt with enhanced instructions
    if language == "he":
        prompt = f"""אתה MiniMe AI, עוזר אישי ידידותי ומפורט שעוזר למשתמשים למצוא מידע בהודעות התקשורת שלהם (אימייל, ווטסאפ, הודעות טקסט וכו').
        
להלן מידע מההודעות של המשתמש שעשוי להיות רלוונטי לשאילתה שלו:

{context}

בהתבסס על המידע לעיל, אנא ענה על שאילתת המשתמש בצורה מפורטת, מועילה וידידותית:
{query}

הנה כמה הנחיות:
1. תן תשובות מלאות ומפורטות, אבל ממוקדות בשאלה.
2. הצג את המידע בצורה מאורגנת ובהירה.
3. הכלל פרטים רלוונטיים כמו תאריכים, שמות השולחים והנושאים.
4. אם יש מידע נוסף שעשוי להיות מעניין או רלוונטי, כלול אותו גם.
5. השתמש בטון טבעי וידידותי.
6. תמיד ספק את המידע ישירות - אל תשאל אם המשתמש רוצה את המידע.

אם לא ניתן לקבוע את התשובה מההקשר שסופק, פשוט אמור שאין לך את המידע הזה ושאתה יכול לחפש רק בהודעות זמינות.
"""
    else:
        prompt = f"""You are MiniMe AI, a friendly and detailed personal assistant that helps users find information in their communications (emails, WhatsApp, text messages, etc.).
    
Below is information from the user's messages that might be relevant to their query:

{context}

Based on the information above, please answer the user's query in a detailed, helpful, and friendly manner:
{query}

Guidelines:
1. Provide thorough and complete answers, while staying focused on the question.
2. Present information in an organized and clear way.
3. Include relevant details like dates, sender names, and subjects.
4. If there's additional information that might be interesting or relevant, include that as well.
5. Use a natural, conversational tone.
6. When mentioning dates or times, specify them clearly (e.g., "March 13th at 2:15 PM" rather than just "today").
7. Always provide information directly - DO NOT ask if the user wants information.

If the answer cannot be determined from the provided context, politely explain that you don't have that information and can only search through available messages.
"""
    
    try:
        if LLM_PROVIDER == "openai":
            # Use the OpenAI ChatCompletion API with v0.28 syntax
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are MiniMe AI, a helpful and detailed assistant that answers questions based on the user's communications. Provide thorough and informative responses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,  # Increased from 1000 to allow for more detailed responses
                temperature=0.5   # Slightly increased from 0.3 to allow more natural language
            )
            return response.choices[0].message.content
        elif LLM_PROVIDER == "anthropic":
            if not anthropic_client:
                return "Error: Anthropic client not initialized."
                
            response = anthropic_client.completions.create(
                model=MODEL_NAME,
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
                max_tokens_to_sample=1500,  # Increased from 1000
                temperature=0.5             # Slightly increased from 0.3
            )
            return response.completion
        else:
            return "Error: LLM provider not configured correctly."
    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        logger.error(traceback.format_exc())
        if language == "he":
            return f"שגיאה בשאילתה למודל השפה: {str(e)}"
        else:
            return f"Error querying language model: {str(e)}"


def query_llm_with_history(query: str, context: str, conversation_history: List[Dict[str, str]] = None) -> str:
    """Query LLM with conversation history."""
    # Detect query language
    language = detect_language(query)
    
    # Format conversation history for prompt
    formatted_history = ""
    if conversation_history and len(conversation_history) > 0:
        # Get up to last 10 exchanges to keep context manageable
        recent_history = conversation_history[-10:]
        for message in recent_history:
            role = message.get('role', '')
            content = message.get('content', '')
            if role == 'user':
                formatted_history += f"User: {content}\n"
            elif role == 'assistant':
                formatted_history += f"MiniMeAI: {content}\n"
        formatted_history += "\n"
    
    # Create language-specific prompt
    if language == "he":
        prompt = f"""אתה MiniMe AI, עוזר אישי ידידותי ומפורט שעוזר למשתמשים למצוא מידע בהודעות התקשורת שלהם (אימייל, ווטסאפ, הודעות טקסט וכו').

היסטוריית השיחה הקודמת:
{formatted_history}

להלן מידע מההודעות של המשתמש שעשוי להיות רלוונטי לשאילתה הנוכחית:

{context}

בהתבסס על המידע לעיל ועל היסטוריית השיחה, אנא ענה על שאילתת המשתמש בצורה מפורטת, מועילה וידידותית:
{query}

הנחיות:
1. תן תשובות מלאות ומפורטות, אבל ממוקדות בשאלה.
2. הצג את המידע בצורה מאורגנת ובהירה.
3. הכלל פרטים רלוונטיים כמו תאריכים, שמות השולחים והנושאים.
4. אם יש מידע נוסף שעשוי להיות מעניין או רלוונטי, כלול אותו גם.
5. השתמש בטון טבעי, ידידותי ושיחתי.
6. התייחס למידע מהשיחה הקודמת כשרלוונטי.
7. תמיד ספק את המידע ישירות - אל תשאל אם המשתמש רוצה את המידע.

אם לא ניתן לקבוע את התשובה מההקשר שסופק, פשוט אמור שאין לך את המידע הזה ושאתה יכול לחפש רק בהודעות זמינות.
"""
    else:
        prompt = f"""You are MiniMe AI, a friendly and detailed personal assistant that helps users find information in their communications (emails, WhatsApp, text messages, etc.).

Previous conversation history:
{formatted_history}

Below is information from the user's messages that might be relevant to their current query:

{context}

Based on the above information and conversation history, please answer the user's query in a detailed, helpful, and friendly manner:
{query}

Guidelines:
1. Provide thorough and complete answers, while staying focused on the question.
2. Present information in an organized and clear way.
3. Include relevant details like dates, sender names, and subjects.
4. If there's additional information that might be interesting or relevant, include that as well.
5. Use a natural, conversational tone.
6. When mentioning dates or times, specify them clearly (e.g., "March 13th at 2:15 PM" rather than just "today").
7. Reference information from previous messages when relevant.
8. Always provide information directly without asking if the user wants the information.

If the answer cannot be determined from the provided context, politely explain that you don't have that information and can only search through available messages.
"""
    
    try:
        if LLM_PROVIDER == "openai":
            # Create messages array for conversation
            messages = [
                {"role": "system", "content": "You are MiniMe AI, a helpful and detailed assistant that answers questions based on the user's communications. You have access to their emails and messages. Provide thorough, direct, and informative responses."}
            ]
            
            # Add conversation history if available
            if conversation_history and len(conversation_history) > 0:
                # Get up to last 10 exchanges
                recent_history = conversation_history[-10:]
                for message in recent_history:
                    role = message.get('role', '')
                    content = message.get('content', '')
                    if role in ['user', 'assistant']:
                        messages.append({"role": role, "content": content})
            
            # Add the current query with context
            messages.append({"role": "user", "content": prompt})
            
            # Use the OpenAI ChatCompletion API
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=1500,
                temperature=0.5
            )
            return response.choices[0].message.content
        elif LLM_PROVIDER == "anthropic":
            if not anthropic_client:
                return "Error: Anthropic client not initialized."
                
            response = anthropic_client.completions.create(
                model=MODEL_NAME,
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
                max_tokens_to_sample=1500,
                temperature=0.5
            )
            return response.completion
        else:
            return "Error: LLM provider not configured correctly."
    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        logger.error(traceback.format_exc())
        if language == "he":
            return f"שגיאה בשאילתה למודל השפה: {str(e)}"
        else:
            return f"Error querying language model: {str(e)}"


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": time.time()})


@timed
def handle_query(query):
    """Handle a query using the existing query_endpoint implementation."""
    with Timer("Query processing"):
        # Save the original request
        original_request = request
        
        # Create a request-like object with the query
        class MockRequest:
            def __init__(self, query):
                self.json = {"query": query}
                
            def get_json(self):
                return self.json
        
        # Replace Flask's request with our mock
        import flask
        flask.request = MockRequest(query)
        
        try:
            # Use the existing query_endpoint implementation
            response = query_endpoint()
            
            # Extract just the response content
            if isinstance(response, tuple):
                return response[0]  # First element is the response data
            return response
        finally:
            # Restore the original request
            flask.request = original_request


@app.route("/api/query", methods=["POST"])
def query_endpoint():
    """Query endpoint for searching messages."""
    try:
        data = request.json
        if not data or "query" not in data:
            return jsonify({"error": "Query not provided"}), 400
        
        query = data["query"]
        conversation_history = data.get("conversation_history", [])
        use_optimized = data.get("use_optimized", True)  # Default to using optimized query if available
        
        logger.info(f"Received query: {query}")
        logger.info(f"Conversation history length: {len(conversation_history)}")
        
        # Get embeddings for query
        embedding_start = time.time()
        query_embedding = get_embeddings(query)
        embedding_time = time.time() - embedding_start
        logger.info(f"Generated embedding for query: '{query[:50]}...' in {embedding_time:.3f}s")
        if not query_embedding:
            return jsonify({"error": "Failed to generate embeddings"}), 500
        
        # Connect to ChromaDB
        collection = connect_to_chroma()
        logger.info(f"Connected to vector DB, collection: {VECTOR_COLLECTION_NAME}")
        if not collection:
            return jsonify({"error": "Failed to connect to vector database"}), 500
        
        # Check if collection is optimized without triggering optimization
        try:
            status = collection.get_status()
            is_optimized = status.get("is_optimized", False)
            meets_threshold = status.get("meets_threshold", False)
            optimization_coverage = status.get("optimization_coverage", "0%")
            logger.info(f"Collection optimization status: {is_optimized} (coverage: {optimization_coverage}, meets threshold: {meets_threshold})")
            
            # Only use optimized query if the collection meets the optimization threshold
            use_optimized = use_optimized and meets_threshold
        except Exception as e:
            logger.warning(f"Could not check collection optimization status: {e}. Will use standard query.")
            use_optimized = False
        
        # Query ChromaDB
        try:
            logger.info(f"Querying vector DB collection: {VECTOR_COLLECTION_NAME}")
            logger.info(f"Query: '{query}'")
            
            # Track vector DB query time
            vectordb_start = time.time()
            
            # Use optimized query if available and requested
            if use_optimized:
                logger.info(f"Using optimized vector DB query")
                results = collection.query_optimized(
                    query_embeddings=[query_embedding],
                    n_results=5,
                    pre_filter_ratio=0.05  # Check 5% of the embeddings in full resolution
                )
            else:
                logger.info(f"Using standard vector DB query")
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=5
                )
            
            vectordb_time = time.time() - vectordb_start
            
            # Log detailed timing information if available
            if isinstance(results, dict) and "_timing" in results:
                timing_info = results["_timing"]
                logger.info(f"Vector DB query performance:")
                logger.info(f"  - Total query time: {timing_info.get('total_time', vectordb_time):.3f}s")
                logger.info(f"  - Network request time: {timing_info.get('request_time', 'N/A')}s")
                
                # Additional timing info for optimized queries
                if "avg_reduction_time" in timing_info:
                    logger.info(f"  - Dimension reduction time: {timing_info.get('avg_reduction_time', 'N/A')}s")
                    logger.info(f"  - Pre-filtering time: {timing_info.get('avg_pre_filtering_time', 'N/A')}s")
                    logger.info(f"  - Full calculation time: {timing_info.get('avg_full_calc_time', 'N/A')}s")
                
                # Remove timing info from results to avoid sending it to the client
                if "_timing" in results:
                    timing_data = results.pop("_timing")
            else:
                logger.info(f"Vector DB query completed in {vectordb_time:.3f}s")
            
            if not results or not results.get("documents"):
                return jsonify({"response": "No relevant information found."}), 200
            
            # Log results
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            logger.info(f"Query returned {len(documents)} documents")
            
            # Check if query is about recency
            language = detect_language(query)
            logger.info(f"Checking recency for query: '{query}' (language: {language})")

            # Use enhanced recency detection
            is_recency_detected = is_recency_query(query, language)
            logger.info(f"Recency query detected: {is_recency_detected}")

            # For recency queries or time-related queries, apply enhanced sorting
            if is_recency_detected:
                documents, metadatas = apply_recency_sorting(documents, metadatas)
            else:
                # Even for non-recency queries, ensure any documents with timestamps are properly sorted
                # This improves general search experience by promoting newer content when relevant
                doc_timestamps = [meta.get("timestamp", 0) for meta in metadatas]
                if any(ts > 0 for ts in doc_timestamps):
                    logger.info(f"Applying light recency boost to non-recency query results")
                    # Create document-metadata pairs with a timestamp score
                    doc_meta_pairs = list(zip(documents, metadatas))
                    
                    # Add recency score (lower weight than for explicit recency queries)
                    for _, meta in doc_meta_pairs:
                        ts = meta.get("timestamp", 0)
                        if not ts or ts == 0:
                            date_str = meta.get("date", "")
                            if date_str:
                                ts = get_reliable_timestamp(date_str)
                                meta["timestamp"] = ts
                    
                    # Reconstruct lists
                    documents = [doc for doc, _ in doc_meta_pairs]
                    metadatas = [meta for _, meta in doc_meta_pairs]
            
            logger.info(f"Building context from {len(documents)} documents")
            context = ""
            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                # Get source type (email, whatsapp, text, etc.)
                source_type = meta.get("source_type", "message")
                
                context += f"\n--- {source_type.capitalize()} {i+1} ---\n"
                
                # Add metadata based on source type
                if source_type == "email":
                    context += f"From: {meta.get('from', 'Unknown')}\n"
                    context += f"To: {meta.get('to', 'Unknown')}\n"
                    context += f"Subject: {meta.get('subject', 'No Subject')}\n"
                elif source_type == "whatsapp":
                    context += f"From: {meta.get('sender', 'Unknown')}\n"
                    context += f"Chat: {meta.get('chat', 'Unknown')}\n"
                elif source_type == "text":
                    context += f"From: {meta.get('sender', 'Unknown')}\n"
                
                # Common metadata for all types
                context += f"Date: {meta.get('date', 'Unknown')}\n"
                context += f"Content: {doc}\n"
            
            # Query LLM with context and conversation history
            llm_start = time.time()
            logger.info(f"Sending context ({len(context)} chars) to LLM with conversation history")
            if conversation_history and len(conversation_history) > 0:
                response = query_llm_with_history(query, context, conversation_history)
            else:
                response = query_llm(query, context)
            llm_time = time.time() - llm_start
            logger.info(f"LLM response generated in {llm_time:.3f}s: '{response[:100]}...'")
            
            # Log total query processing time breakdown
            total_query_time = time.time() - embedding_start
            logger.info(f"Total query processing time: {total_query_time:.3f}s")
            logger.info(f"  - Embedding generation: {embedding_time:.3f}s ({(embedding_time/total_query_time)*100:.1f}%)")
            logger.info(f"  - Vector DB query: {vectordb_time:.3f}s ({(vectordb_time/total_query_time)*100:.1f}%)")
            logger.info(f"  - LLM generation: {llm_time:.3f}s ({(llm_time/total_query_time)*100:.1f}%)")
            
            return jsonify({"response": response}), 200
        except Exception as e:
            logger.error(f"Error querying vector database: {e}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Error querying vector database: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    # Wait for ChromaDB to be ready
    max_retries = 10
    retry_count = 0
    while retry_count < max_retries:
        try:
            collection = connect_to_chroma()
            if collection:
                logger.info("Successfully connected to vector database")
                break
        except Exception:
            pass
        
        logger.info(f"Waiting for vector database to be ready... ({retry_count+1}/{max_retries})")
        retry_count += 1
        time.sleep(5)
    
    if retry_count == max_retries:
        logger.warning("Could not connect to vector database after maximum retries")
    
    # Start the Flask app
    logger.info("Starting API server")
    app.run(host="0.0.0.0", port=5000)
