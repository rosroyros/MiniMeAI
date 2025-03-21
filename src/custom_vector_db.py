from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple, Set
import json
import os
import pickle
import math
import random
import heapq
import functools
import time
import logging
from datetime import datetime, timedelta
import threading
import queue
import concurrent.futures
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Replace external timing imports with local implementations
class Timer:
    """Simple timer utility for measuring execution time."""
    
    def __init__(self, name=None):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.name:
            print(f"{self.name} took {self.interval:.3f}s")

def timed(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.3f}s")
        return result
    return wrapper

# Pure Python Vector Operations
def dot_product(v1, v2):
    """Calculate dot product of two vectors without NumPy."""
    return sum(a * b for a, b in zip(v1, v2))

def vector_magnitude(v):
    """Calculate magnitude (length) of a vector without NumPy."""
    return math.sqrt(sum(x * x for x in v))

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors without NumPy."""
    dot = dot_product(v1, v2)
    mag1 = vector_magnitude(v1)
    mag2 = vector_magnitude(v2)
    
    # Avoid division by zero
    if mag1 * mag2 == 0:
        return 0
        
    return dot / (mag1 * mag2)

def reduce_dimensions(vectors, target_dim=128, chunk_size=100, projection_matrix=None):
    """
    Simple dimension reduction using random projection.
    This is a fallback for systems without NumPy/scikit-learn.
    Added chunking to prevent memory issues.
    
    If projection_matrix is provided, use it instead of generating a new one.
    """
    if not vectors or len(vectors) == 0:
        return [], projection_matrix
        
    # Get original dimension
    input_dim = len(vectors[0])
    
    # Create a random projection matrix - only once
    if projection_matrix is None:
        # For each target dimension, create a random vector in input space
        # and normalize it
        random.seed(42)  # For reproducibility
        projection_matrix = []
        
        for _ in range(target_dim):
            # Create a random vector
            random_vector = [random.uniform(-1, 1) for _ in range(input_dim)]
            # Normalize it
            magnitude = math.sqrt(sum(x*x for x in random_vector))
            normalized_vector = [x/magnitude for x in random_vector]
            projection_matrix.append(normalized_vector)
    
    # Project each vector in chunks to prevent memory issues
    result = []
    
    # Process vectors in chunks
    for i in range(0, len(vectors), chunk_size):
        chunk = vectors[i:i+chunk_size]
        chunk_result = []
        
        for vec in chunk:
            # Project vector onto each basis vector
            projected = [dot_product(vec, basis) for basis in projection_matrix]
            chunk_result.append(projected)
        
        result.extend(chunk_result)
        
        # Log progress
        logger.info(f"Reduced dimensions for {i+len(chunk)}/{len(vectors)} vectors")
        
    return result, projection_matrix

# Initialize FastAPI app at the module level
app = FastAPI(title="Simple Vector DB")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Storage locations
DATA_DIR = "/chroma/chroma"
COLLECTIONS_DIR = os.path.join(DATA_DIR, "collections")

# Initialize directories
os.makedirs(COLLECTIONS_DIR, exist_ok=True)

# In-memory storage (will be persisted to disk)
collections = {}

# Cache for vector norms to avoid recalculating
norm_cache = {}
# Cache for reduced dimension vectors
reduced_vectors_cache = {}
# Cache for cosine similarity calculations
similarity_cache = {}

# Collection Models
class Collection:
    def __init__(self, name):
        self.name = name
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        self.optimized = False
        self.reduced_embeddings = []
        self.projection_matrix = None
        
        # Track new vectors for incremental optimization
        self.new_documents = []
        self.new_embeddings = []
        self.new_metadatas = []
        self.new_ids = []
        self.new_reduced_embeddings = []
        
        # Optimization scheduling
        self.last_hourly_optimization = datetime.now() - timedelta(hours=2)  # Start with need for optimization
        self.last_daily_optimization = datetime.now() - timedelta(days=2)    # Start with need for optimization
        self.next_scheduled_optimization = None
        
    def add(self, documents, embeddings, metadatas, ids):
        # Add to main collection
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        # Also track as new items for incremental optimization
        self.new_documents.extend(documents)
        self.new_embeddings.extend(embeddings)
        self.new_metadatas.extend(metadatas)
        self.new_ids.extend(ids)
        
        # If we have projection matrix, we can immediately reduce dimensions for new vectors
        if self.projection_matrix and self.optimized:
            # Only reduce dimensions for the new vectors
            new_reduced, _ = reduce_dimensions(
                self.new_embeddings, 
                target_dim=len(self.reduced_embeddings[0]) if self.reduced_embeddings else 128,
                projection_matrix=self.projection_matrix
            )
            self.new_reduced_embeddings = new_reduced
            logger.info(f"Reduced dimensions for {len(new_reduced)} new vectors (immediate incremental)")
            # But still need hourly full optimization
            
        # Schedule next optimization times
        self._schedule_next_optimization()
        
    def _schedule_next_optimization(self):
        """Schedule the next optimization time"""
        now = datetime.now()
        
        # Calculate next hourly optimization time
        next_hour = self.last_hourly_optimization + timedelta(hours=1)
        if next_hour < now:
            next_hour = now + timedelta(minutes=5)  # If we're past due, schedule soon
            
        # Calculate next daily optimization (2am)
        tomorrow_2am = datetime.now().replace(hour=2, minute=0, second=0)
        if tomorrow_2am < now:
            tomorrow_2am = tomorrow_2am + timedelta(days=1)
            
        # Set the next optimization time to the earlier of the two
        self.next_scheduled_optimization = min(next_hour, tomorrow_2am)
        logger.info(f"Next optimization for {self.name} scheduled at {self.next_scheduled_optimization}")
    
    def check_needs_optimization(self):
        """Check if optimization is needed based on schedule"""
        now = datetime.now()
        
        # If no schedule is set, set it now
        if self.next_scheduled_optimization is None:
            self._schedule_next_optimization()
            return False
            
        # Check if we've reached the scheduled time
        if now >= self.next_scheduled_optimization:
            # Check if it's time for daily full optimization (between 2am and 3am)
            is_daily = (now.hour == 2 and self.last_daily_optimization.date() < now.date())
            
            # If we need optimization, reschedule next time
            self._schedule_next_optimization()
            
            if is_daily:
                self.last_daily_optimization = now
                logger.info(f"Daily optimization triggered for {self.name}")
                return "full"
            else:
                self.last_hourly_optimization = now
                logger.info(f"Hourly optimization triggered for {self.name}")
                return "incremental"
                
        return False
    
    def optimize_incremental(self):
        """Optimize only the new vectors added since last optimization"""
        if not self.new_embeddings:
            logger.info(f"No new vectors to optimize for {self.name}")
            return True
            
        with Timer(f"Incremental dimension reduction for {len(self.new_embeddings)} vectors"):
            # If we don't have a projection matrix yet, we need to do a full optimization
            if not self.projection_matrix or not self.optimized:
                logger.info(f"No projection matrix exists, doing full optimization for {self.name}")
                return self.optimize()
                
            target_dim = len(self.reduced_embeddings[0]) if self.reduced_embeddings else 128
            
            # Process in smaller chunks to prevent memory issues
            chunk_size = 100
            new_reduced, _ = reduce_dimensions(
                self.new_embeddings,
                target_dim=target_dim,
                chunk_size=chunk_size,
                projection_matrix=self.projection_matrix
            )
            
            # Append the newly reduced vectors to the main reduced embeddings
            if not self.reduced_embeddings:
                self.reduced_embeddings = new_reduced
            else:
                self.reduced_embeddings.extend(new_reduced)
            
            # Clear the new vectors tracking
            self.new_documents = []
            self.new_embeddings = []
            self.new_metadatas = []
            self.new_ids = []
            self.new_reduced_embeddings = []
            
            self.optimized = True
            self.save()
            
            logger.info(f"Incremental dimension reduction complete for {self.name}")
            
        return True
        
    def optimize(self, target_dim=128):
        """Create reduced dimension vectors for faster initial filtering."""
        if not self.embeddings:
            return False
            
        # Define the optimization threshold
        OPTIMIZATION_THRESHOLD = 0.95  # 95% coverage is considered optimized
            
        # Only optimize if not already optimized or if dimensions don't match
        # or if we have new embeddings to process
        if (self.optimized and 
            self.reduced_embeddings and 
            len(self.reduced_embeddings) >= len(self.embeddings) * OPTIMIZATION_THRESHOLD and 
            not self.new_embeddings):
            logger.info(f"Collection {self.name} already meets optimization threshold ({OPTIMIZATION_THRESHOLD:.1%}), skipping")
            return True
            
        with Timer("Dimension reduction"):
            logger.info(f"Starting full dimension reduction for {len(self.embeddings)} vectors")
            
            # Process in smaller chunks to prevent memory issues
            chunk_size = 100
            self.reduced_embeddings, self.projection_matrix = reduce_dimensions(
                self.embeddings, 
                target_dim=target_dim, 
                chunk_size=chunk_size,
                projection_matrix=None  # Always create a new matrix for full optimization
            )
            
            # Clear the new vectors tracking
            self.new_documents = []
            self.new_embeddings = []
            self.new_metadatas = []
            self.new_ids = []
            self.new_reduced_embeddings = []
            
            self.optimized = True
            self.save()
            
            # Calculate and log the optimization coverage
            optimization_coverage = len(self.reduced_embeddings) / len(self.embeddings) if len(self.embeddings) > 0 else 1.0
            logger.info(f"Full dimension reduction complete for {self.name}. Created {len(self.reduced_embeddings)} reduced vectors ({optimization_coverage:.1%} coverage)")
            
        return True
        
    def save(self):
        collection_dir = os.path.join(COLLECTIONS_DIR, self.name)
        os.makedirs(collection_dir, exist_ok=True)
        
        with open(os.path.join(collection_dir, "data.pickle"), "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "embeddings": self.embeddings,
                "metadatas": self.metadatas,
                "ids": self.ids,
                "optimized": self.optimized,
                "reduced_embeddings": self.reduced_embeddings,
                "projection_matrix": self.projection_matrix,
                "new_documents": self.new_documents,
                "new_embeddings": self.new_embeddings,
                "new_metadatas": self.new_metadatas,
                "new_ids": self.new_ids,
                "new_reduced_embeddings": self.new_reduced_embeddings,
                "last_hourly_optimization": self.last_hourly_optimization,
                "last_daily_optimization": self.last_daily_optimization
            }, f)
            
    def load(self):
        collection_dir = os.path.join(COLLECTIONS_DIR, self.name)
        data_path = os.path.join(collection_dir, "data.pickle")
        
        if os.path.exists(data_path):
            with open(data_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data.get("documents", [])
                self.embeddings = data.get("embeddings", [])
                self.metadatas = data.get("metadatas", [])
                self.ids = data.get("ids", [])
                self.optimized = data.get("optimized", False)
                self.reduced_embeddings = data.get("reduced_embeddings", [])
                self.projection_matrix = data.get("projection_matrix", None)
                self.new_documents = data.get("new_documents", [])
                self.new_embeddings = data.get("new_embeddings", [])
                self.new_metadatas = data.get("new_metadatas", [])
                self.new_ids = data.get("new_ids", [])
                self.new_reduced_embeddings = data.get("new_reduced_embeddings", [])
                self.last_hourly_optimization = data.get("last_hourly_optimization", datetime.now() - timedelta(hours=2))
                self.last_daily_optimization = data.get("last_daily_optimization", datetime.now() - timedelta(days=2))
                self._schedule_next_optimization()

# Add the background optimization scheduler
class OptimizationScheduler:
    def __init__(self, check_interval=60):  # Check every minute
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        
    def start(self):
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._schedule_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Optimization scheduler started")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Optimization scheduler stopped")
        
    def _schedule_loop(self):
        while self.running:
            try:
                self._check_collections()
            except Exception as e:
                logger.error(f"Error in optimization scheduler: {e}")
            
            # Sleep until next check
            time.sleep(self.check_interval)
            
    def _check_collections(self):
        """Check all collections for scheduled optimization"""
        for name, collection in collections.items():
            optimization_type = collection.check_needs_optimization()
            
            if optimization_type == "full":
                # Run in a separate thread to avoid blocking
                logger.info(f"Starting full optimization for {name}")
                threading.Thread(
                    target=collection.optimize, 
                    name=f"optimize-{name}-{time.time()}"
                ).start()
            elif optimization_type == "incremental":
                # Run in a separate thread to avoid blocking
                logger.info(f"Starting incremental optimization for {name}")
                threading.Thread(
                    target=collection.optimize_incremental,
                    name=f"optimize-incr-{name}-{time.time()}"
                ).start()

# Create and start the scheduler
scheduler = OptimizationScheduler()

@app.on_event("startup")
async def startup_event():
    # Start the optimization scheduler
    scheduler.start()
    logger.info("Vector DB started with scheduled optimization")

@app.on_event("shutdown")
async def shutdown_event():
    # Stop the optimization scheduler
    scheduler.stop()
    logger.info("Vector DB shutdown, optimization scheduler stopped")

# Request/Response Models
class AddRequest(BaseModel):
    documents: List[str]
    embeddings: List[List[float]]
    metadatas: List[Dict[str, Any]]
    ids: List[str]

class QueryRequest(BaseModel):
    query_embeddings: List[List[float]]
    n_results: Optional[int] = 10

class OptimizedQueryRequest(BaseModel):
    query_embeddings: List[List[float]]
    n_results: Optional[int] = 10
    pre_filter_ratio: Optional[float] = 0.05  # Use top 5% for full search

class OptimizeRequest(BaseModel):
    target_dim: Optional[int] = 128

# Endpoints
@app.post("/api/v1/collections")
async def create_collection(request: dict):
    name = request.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Collection name is required")
        
    if name in collections:
        raise HTTPException(status_code=409, detail="Collection already exists")
        
    collections[name] = Collection(name)
    collections[name].save()
    
    return {"name": name}

@app.get("/api/v1/collections/{name}")
async def get_collection(name: str):
    if name not in collections:
        # Try to load it from disk
        collection_dir = os.path.join(COLLECTIONS_DIR, name)
        if os.path.exists(collection_dir):
            collections[name] = Collection(name)
            collections[name].load()
        else:
            raise HTTPException(status_code=404, detail="Collection not found")
            
    return {"name": name}

@app.post("/api/v1/collections/{name}/add")
async def add_to_collection(name: str, request: AddRequest):
    if name not in collections:
        # Try to load it from disk
        collection_dir = os.path.join(COLLECTIONS_DIR, name)
        if os.path.exists(collection_dir):
            collections[name] = Collection(name)
            collections[name].load()
        else:
            raise HTTPException(status_code=404, detail="Collection not found")
            
    collection = collections[name]
    collection.add(
        request.documents,
        request.embeddings,
        request.metadatas,
        request.ids
    )
    collection.save()
    
    return {"success": True}
    
@app.post("/api/v1/collections/{name}/query")
async def query_collection(name: str, request: QueryRequest):
    with Timer("Total query time"):
        if name not in collections:
            # Try to load it from disk
            collection_dir = os.path.join(COLLECTIONS_DIR, name)
            if os.path.exists(collection_dir):
                collections[name] = Collection(name)
                collections[name].load()
            else:
                raise HTTPException(status_code=404, detail="Collection not found")
                
        collection = collections[name]
        n_results = min(request.n_results, len(collection.embeddings)) if collection.embeddings else 0
        
        if not collection.embeddings or n_results == 0:
            return {
                "ids": [[] for _ in request.query_embeddings],
                "documents": [[] for _ in request.query_embeddings],
                "metadatas": [[] for _ in request.query_embeddings],
                "distances": [[] for _ in request.query_embeddings]
            }
        
        results = []
        
        # Process each query embedding
        with Timer("Search computation"):
            for query_embedding in request.query_embeddings:
                # Calculate similarities and find top n
                similarities = []
                
                for i, embedding in enumerate(collection.embeddings):
                    # Check cache first
                    cache_key = (tuple(query_embedding), tuple(embedding))
                    if cache_key in similarity_cache:
                        similarity = similarity_cache[cache_key]
                    else:
                        similarity = cosine_similarity(query_embedding, embedding)
                        similarity_cache[cache_key] = similarity
                        
                    similarities.append((i, similarity))
                
                # Sort by similarity (higher is better)
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Get top n
                top_n = similarities[:n_results]
                
                # Extract results
                ids = [collection.ids[i] for i, _ in top_n]
                documents = [collection.documents[i] for i, _ in top_n]
                metadatas = [collection.metadatas[i] for i, _ in top_n]
                distances = [1 - sim for _, sim in top_n]  # Convert to distance
                
                results.append({
                    "ids": ids,
                    "documents": documents,
                    "metadatas": metadatas,
                    "distances": distances
                })
        
        # Format final response
        response = {
            "ids": [r["ids"] for r in results],
            "documents": [r["documents"] for r in results],
            "metadatas": [r["metadatas"] for r in results],
            "distances": [r["distances"] for r in results]
        }
        
        return response

@app.post("/api/v1/collections/{name}/query_optimized")
async def query_collection_optimized(name: str, request: OptimizedQueryRequest):
    """Query a collection with embeddings using optimized two-stage approach."""
    if name not in collections:
        # Try to load it from disk
        collection_dir = os.path.join(COLLECTIONS_DIR, name)
        if os.path.exists(collection_dir):
            collections[name] = Collection(name)
            collections[name].load()
        else:
            raise HTTPException(status_code=404, detail="Collection not found")

    collection = collections[name]
    query_embeddings = request.query_embeddings
    n_results = request.n_results
    pre_filter_ratio = request.pre_filter_ratio
    
    # Check if collection is optimized with threshold approach
    OPTIMIZATION_THRESHOLD = 0.95  # Consider "optimized" if 95% or more vectors have reduced dimensions
    
    if not collection.optimized:
        logger.warning(f"Collection {name} is not optimized at all. Falling back to regular query.")
        return await query_collection(name, QueryRequest(
            query_embeddings=query_embeddings,
            n_results=n_results
        ))
    
    # Calculate optimization coverage
    total_vectors = len(collection.embeddings)
    optimized_vectors = len(collection.reduced_embeddings)
    optimization_coverage = optimized_vectors / total_vectors if total_vectors > 0 else 0
    
    if optimization_coverage < OPTIMIZATION_THRESHOLD:
        logger.warning(f"Collection {name} optimization coverage ({optimization_coverage:.1%}) below threshold ({OPTIMIZATION_THRESHOLD:.1%}). Falling back to regular query.")
        return await query_collection(name, QueryRequest(
            query_embeddings=query_embeddings,
            n_results=n_results
        ))
    
    logger.info(f"Using optimized query with {optimization_coverage:.1%} vector coverage (above {OPTIMIZATION_THRESHOLD:.1%} threshold)")

    # Make sure we have data to work with
    if not collection.embeddings or not collection.reduced_embeddings:
        raise HTTPException(status_code=400, detail="Collection has no embeddings")

    results = {
        "ids": [[] for _ in range(len(query_embeddings))],
        "distances": [[] for _ in range(len(query_embeddings))],
        "documents": [[] for _ in range(len(query_embeddings))],
        "metadatas": [[] for _ in range(len(query_embeddings))]
    }
    
    with Timer("Optimized query") as timer:
        # For each query embedding
        for i, query_embedding in enumerate(query_embeddings):
            # First, reduce the query embedding
            query_reduced, _ = reduce_dimensions([query_embedding], target_dim=len(collection.reduced_embeddings[0]), projection_matrix=collection.projection_matrix)
            query_reduced = query_reduced[0]  # Get the first (only) result
            
            # First stage: Use reduced embeddings to find candidates
            # This should be fast even with many vectors
            with Timer("First stage - reduced embedding search"):
                candidate_indices = []
                candidate_scores = []
                
                # Process in chunks to prevent memory issues
                chunk_size = 500
                for j in range(0, len(collection.reduced_embeddings), chunk_size):
                    chunk = collection.reduced_embeddings[j:j+chunk_size]
                    
                    # Calculate similarity scores for this chunk
                    scores = []
                    for k, embedding in enumerate(chunk):
                        # Use cached similarity if available
                        cache_key = (tuple(query_reduced), tuple(embedding))
                        if cache_key in similarity_cache:
                            scores.append((similarity_cache[cache_key], j+k))
                        else:
                            similarity = cosine_similarity(query_reduced, embedding)
                            similarity_cache[cache_key] = similarity
                            scores.append((similarity, j+k))
                    
                    # Add top candidates from this chunk
                    scores.sort(reverse=True)
                    for score, idx in scores[:n_results]:
                        if len(candidate_indices) < n_results:
                            candidate_indices.append(idx)
                            candidate_scores.append(score)
                        elif score > min(candidate_scores):
                            # Replace the lowest score
                            min_idx = candidate_scores.index(min(candidate_scores))
                            candidate_indices[min_idx] = idx
                            candidate_scores[min_idx] = score
            
            # Second stage: Use full embeddings to re-rank candidates
            with Timer("Second stage - full embedding search"):
                candidate_results = []
                
                for idx in candidate_indices:
                    # Use cached similarity if available
                    cache_key = (tuple(query_embedding), tuple(collection.embeddings[idx]))
                    if cache_key in similarity_cache:
                        similarity = similarity_cache[cache_key]
                    else:
                        similarity = cosine_similarity(query_embedding, collection.embeddings[idx])
                        similarity_cache[cache_key] = similarity
                    
                    # Only keep if similarity is positive (related)
                    if similarity > 0:
                        candidate_results.append((similarity, idx))
                
                # Sort by similarity (descending)
                candidate_results.sort(reverse=True)
                
                # Take top n_results
                for similarity, idx in candidate_results[:n_results]:
                    results["ids"][i].append(collection.ids[idx])
                    results["distances"][i].append(1.0 - similarity)  # Convert to distance
                    results["documents"][i].append(collection.documents[idx])
                    results["metadatas"][i].append(collection.metadatas[idx])
    
    return results

@app.post("/api/v1/collections/{name}/optimize")
async def optimize_collection(name: str, request: OptimizeRequest):
    if name not in collections:
        # Try to load it from disk
        collection_dir = os.path.join(COLLECTIONS_DIR, name)
        if os.path.exists(collection_dir):
            collections[name] = Collection(name)
            collections[name].load()
        else:
            raise HTTPException(status_code=404, detail="Collection not found")
            
    collection = collections[name]
    
    if collection.optimize(target_dim=request.target_dim):
        return {"success": True, "message": "Collection optimized successfully"}
    else:
        return {"success": False, "message": "No embeddings to optimize"}

# Add a status endpoint to check optimization
@app.get("/api/v1/collections/{name}/status")
async def get_collection_status(name: str):
    """Get the optimization status of a collection."""
    if name not in collections:
        # Try to load it from disk
        collection_dir = os.path.join(COLLECTIONS_DIR, name)
        if os.path.exists(collection_dir):
            collections[name] = Collection(name)
            collections[name].load()
        else:
            raise HTTPException(status_code=404, detail="Collection not found")
    
    collection = collections[name]
    
    # Calculate optimization metrics
    total_vectors = len(collection.embeddings)
    optimized_vectors = len(collection.reduced_embeddings) if collection.optimized else 0
    pending_vectors = len(collection.new_embeddings)
    optimization_coverage = (optimized_vectors / total_vectors * 100) if total_vectors > 0 else 0
    
    # Calculate if it meets the threshold for optimized queries
    OPTIMIZATION_THRESHOLD = 0.95
    meets_threshold = collection.optimized and (optimized_vectors / total_vectors >= OPTIMIZATION_THRESHOLD if total_vectors > 0 else False)
    
    # Get next scheduled optimization times
    next_hourly = collection.last_hourly_optimization + timedelta(hours=1)
    tomorrow_2am = datetime.now().replace(hour=2, minute=0, second=0)
    if tomorrow_2am < datetime.now():
        tomorrow_2am = tomorrow_2am + timedelta(days=1)
    
    return {
        "name": name,
        "is_optimized": collection.optimized,
        "meets_threshold": meets_threshold,
        "optimization_threshold": f"{OPTIMIZATION_THRESHOLD:.1%}",
        "total_vectors": total_vectors,
        "optimized_vectors": optimized_vectors,
        "pending_vectors": pending_vectors,
        "optimization_coverage": f"{optimization_coverage:.2f}%",
        "last_hourly_optimization": collection.last_hourly_optimization.isoformat(),
        "last_daily_optimization": collection.last_daily_optimization.isoformat(),
        "next_optimization": collection.next_scheduled_optimization.isoformat(),
        "next_hourly": next_hourly.isoformat(),
        "next_daily": tomorrow_2am.isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
