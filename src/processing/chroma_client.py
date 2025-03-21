import requests
import json
import time
import logging
from typing import List, Dict, Any, Optional

# Configure logger
logger = logging.getLogger("minimeai.chroma_client")

class SimpleChromaClient:
    """A minimal implementation of the ChromaDB HTTP client for embedding storage."""
    
    def __init__(self, host: str, port: int, max_retries: int = 5, retry_delay: int = 2):
        self.base_url = f"http://{host}:{port}"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    def get_collection(self, name: str):
        """Get a collection by name with retry logic."""
        retry_count = 0
        logger.info(f"Attempting to connect to vector DB at {self.base_url}")
        while retry_count < self.max_retries:
            try:
                logger.info(f"Sending request to {self.base_url}/api/v1/collections/{name}")
                response = requests.get(f"{self.base_url}/api/v1/collections/{name}", timeout=10)
                response.raise_for_status()
                logger.info(f"Successfully connected to collection: {name}")
                return SimpleChromaCollection(self.base_url, name)
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 404:
                    logger.info(f"Collection {name} not found (404)")
                    return None
                retry_count += 1
                if retry_count < self.max_retries:
                    logger.warning(f"HTTP error connecting to collection {name}: {e}. Retrying {retry_count}/{self.max_retries}...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Max retries exceeded for HTTP error: {e}")
                    raise
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                retry_count += 1
                logger.warning(f"Connection error details: {str(e)}")
                if retry_count < self.max_retries:
                    logger.warning(f"Connection error to vector db: {e}. Retrying {retry_count}/{self.max_retries}...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to connect to vector db after {self.max_retries} attempts: {e}")
                    raise
            
    def create_collection(self, name: str, embedding_function=None):
        """Create a new collection with retry logic."""
        data = {"name": name}
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                response = requests.post(f"{self.base_url}/api/v1/collections", json=data, timeout=10)
                response.raise_for_status()
                logger.info(f"Successfully created collection: {name}")
                return SimpleChromaCollection(self.base_url, name)
            except (requests.exceptions.RequestException) as e:
                retry_count += 1
                if retry_count < self.max_retries:
                    logger.warning(f"Error creating collection {name}: {e}. Retrying {retry_count}/{self.max_retries}...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to create collection after {self.max_retries} attempts: {e}")
                    raise

class SimpleChromaCollection:
    """A minimal implementation of the ChromaDB collection for embedding storage."""
    
    def __init__(self, base_url: str, name: str):
        self.base_url = base_url
        self.name = name
        
    def add(self, documents: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Add documents with embeddings to the collection."""
        data = {
            "documents": documents,
            "embeddings": embeddings,
            "metadatas": metadatas,
            "ids": ids
        }
        url = f"{self.base_url}/api/v1/collections/{self.name}/add"
        try:
            response = requests.post(url, json=data, timeout=30)  # Longer timeout for larger data uploads
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error adding documents to collection {self.name}: {e}")
            raise
        
    def query(self, query_embeddings: List[List[float]], n_results: int = 10):
        """Query the collection with embeddings."""
        data = {
            "query_embeddings": query_embeddings,
            "n_results": n_results
        }
        url = f"{self.base_url}/api/v1/collections/{self.name}/query"
        
        # Add timing metrics
        start_time = time.time()
        try:
            # Time the HTTP request specifically
            request_start = time.time()
            response = requests.post(url, json=data, timeout=60)  # Increased timeout from 15s to 60s
            request_time = time.time() - request_start
            
            # Process response
            response.raise_for_status()
            result = response.json()
            
            # Calculate and log total time
            total_time = time.time() - start_time
            logger.info(f"Vector DB query stats: request_time={request_time:.3f}s, total_time={total_time:.3f}s, embeddings_count={len(query_embeddings)}")
            
            # Add timing data to result for debugging
            if isinstance(result, dict):
                result["_timing"] = {
                    "request_time": request_time,
                    "total_time": total_time
                }
            
            return result
        except requests.exceptions.RequestException as e:
            total_time = time.time() - start_time
            logger.error(f"Error querying collection {self.name}: {e} (took {total_time:.3f}s)")
            raise
            
    def query_optimized(self, query_embeddings: List[List[float]], n_results: int = 10, pre_filter_ratio: float = 0.05):
        """Query the collection with embeddings using the optimized endpoint."""
        data = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "pre_filter_ratio": pre_filter_ratio
        }
        url = f"{self.base_url}/api/v1/collections/{self.name}/query_optimized"
        
        # Add timing metrics
        start_time = time.time()
        try:
            # Time the HTTP request specifically
            request_start = time.time()
            response = requests.post(url, json=data, timeout=60)  # Increased timeout from 15s to 60s
            request_time = time.time() - request_start
            
            # Process response
            response.raise_for_status()
            result = response.json()
            
            # Calculate and log total time
            total_time = time.time() - start_time
            logger.info(f"Vector DB optimized query stats: request_time={request_time:.3f}s, total_time={total_time:.3f}s, embeddings_count={len(query_embeddings)}")
            
            # Add timing data to result for debugging
            if isinstance(result, dict):
                result["_timing"] = {
                    "request_time": request_time,
                    "total_time": total_time
                }
            
            return result
        except requests.exceptions.RequestException as e:
            total_time = time.time() - start_time
            logger.error(f"Error in optimized query of collection {self.name}: {e} (took {total_time:.3f}s)")
            # Fallback to regular query if optimized query fails
            logger.info(f"Falling back to regular query...")
            return self.query(query_embeddings, n_results)
            
    def optimize(self, target_dim: int = 128):
        """Optimize the collection for faster searching."""
        data = {
            "target_dim": target_dim
        }
        url = f"{self.base_url}/api/v1/collections/{self.name}/optimize"
        try:
            response = requests.post(url, json=data, timeout=300)  # Long timeout for optimization
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error optimizing collection {self.name}: {e}")
            raise
            
    def get_status(self):
        """Get the optimization status of the collection."""
        url = f"{self.base_url}/api/v1/collections/{self.name}/status"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting status of collection {self.name}: {e}")
            raise
