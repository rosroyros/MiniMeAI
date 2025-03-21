#!/usr/bin/env python3
import os
import pickle
import json
import math
import random
from typing import Dict, List, Any

# Storage location - adjust this to match your data path
DATA_DIR = "/media/roy/Seagate/MiniMeAI/data/vectors"
COLLECTIONS_DIR = os.path.join(DATA_DIR, "collections")

def load_collection(name: str):
    """Load a collection from disk."""
    collection_path = os.path.join(COLLECTIONS_DIR, name)
    if not os.path.exists(collection_path):
        print(f"Collection '{name}' not found at {collection_path}")
        return None
    
    try:
        with open(os.path.join(collection_path, "data.pickle"), "rb") as f:
            collection = pickle.load(f)
        print(f"Loaded collection: {name}")
        return collection
    except Exception as e:
        print(f"Error loading collection {name}: {e}")
        return None

def analyze_collection(collection: Dict[str, Any]):
    """Analyze collection and return detailed statistics."""
    if not collection:
        return {"status": "error", "message": "No collection data provided"}
    
    # Basic stats
    total_count = len(collection["documents"])
    if total_count == 0:
        return {
            "count": 0,
            "status": "empty"
        }
    
    print(f"Collection has {total_count} documents")
    
    # Analyze embeddings
    embedding_dim = len(collection["embeddings"][0]) if collection["embeddings"] else 0
    print(f"Embedding dimensions: {embedding_dim}")
    
    # Analyze metadata
    metadata_keys = set()
    metadata_types = {}
    metadata_has_timestamps = False
    
    # Sample up to 100 random items for analysis
    sample_indices = random.sample(range(total_count), min(100, total_count))
    
    for idx in sample_indices:
        metadata = collection["metadatas"][idx]
        for key, value in metadata.items():
            metadata_keys.add(key)
            if key not in metadata_types:
                metadata_types[key] = type(value).__name__
            if key in ["timestamp", "date", "created_at", "modified_at"]:
                metadata_has_timestamps = True
    
    print(f"Metadata keys: {', '.join(metadata_keys)}")
    print(f"Has timestamp fields: {metadata_has_timestamps}")
    
    # Document length stats
    doc_lengths = [len(doc) for doc in collection["documents"]]
    avg_doc_length = sum(doc_lengths) / len(doc_lengths)
    min_doc_length = min(doc_lengths)
    max_doc_length = max(doc_lengths)
    
    print(f"Document lengths - Avg: {avg_doc_length:.1f}, Min: {min_doc_length}, Max: {max_doc_length}")
    
    # Embedding stats
    print("Calculating embedding statistics...")
    
    # Embedding norm stats (for normalization analysis)
    embedding_norms = []
    for emb in collection["embeddings"][:100]:  # Sample up to 100 for performance
        norm = math.sqrt(sum(x*x for x in emb))
        embedding_norms.append(norm)
    
    avg_norm = sum(embedding_norms) / len(embedding_norms)
    min_norm = min(embedding_norms)
    max_norm = max(embedding_norms)
    
    print(f"Embedding norms - Avg: {avg_norm:.4f}, Min: {min_norm:.4f}, Max: {max_norm:.4f}")
    
    # Dimensionality analysis
    zeros_count = 0
    total_elements = 0
    
    for emb in collection["embeddings"][:10]:  # Sample first 10 vectors
        zeros_count += sum(1 for x in emb if abs(x) < 1e-6)
        total_elements += len(emb)
    
    sparsity = zeros_count / total_elements if total_elements > 0 else 0
    print(f"Embedding sparsity: {sparsity:.2%}")
    
    # Sample a few dot products to understand distribution
    sample_similarities = []
    for _ in range(10):  # Sample 10 random pairs
        i = random.randint(0, total_count-1)
        j = random.randint(0, total_count-1)
        if i != j:
            emb1 = collection["embeddings"][i]
            emb2 = collection["embeddings"][j]
            dot_product = sum(a * b for a, b in zip(emb1, emb2))
            norm1 = math.sqrt(sum(a * a for a in emb1))
            norm2 = math.sqrt(sum(b * b for b in emb2))
            similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
            sample_similarities.append(similarity)
    
    avg_similarity = sum(sample_similarities) / len(sample_similarities) if sample_similarities else 0
    print(f"Average random similarity: {avg_similarity:.4f}")
    
    # Performance analysis
    # Time some similarity calculations
    import time
    start_time = time.time()
    emb1 = collection["embeddings"][0]
    
    # Time 100 similarity calculations
    for i in range(1, 101):
        emb2 = collection["embeddings"][i % total_count]
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))
        similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
    
    per_calc_time = (time.time() - start_time) / 100
    print(f"Average time per similarity calc: {per_calc_time:.6f}s")
    print(f"Estimated full scan time for 1 query: {per_calc_time * total_count:.2f}s")
    
    results = {
        "status": "analyzed",
        "count": total_count,
        "embedding_stats": {
            "dimensions": embedding_dim,
            "avg_norm": avg_norm,
            "min_norm": min_norm,
            "max_norm": max_norm,
            "sparsity": sparsity,
            "avg_random_similarity": avg_similarity
        },
        "document_stats": {
            "avg_length": avg_doc_length,
            "min_length": min_doc_length,
            "max_length": max_doc_length
        },
        "metadata": {
            "available_keys": list(metadata_keys),
            "has_timestamps": metadata_has_timestamps,
            "key_types": metadata_types
        },
        "performance": {
            "per_calc_time": per_calc_time,
            "estimated_full_scan": per_calc_time * total_count
        }
    }
    
    return results

def main():
    collection_name = "emails"  # Change this to match your collection name
    collection = load_collection(collection_name)
    
    if collection:
        analysis = analyze_collection(collection)
        print("\nAnalysis results:")
        print(json.dumps(analysis, indent=2))
        
        with open(f"{collection_name}_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\nResults saved to {collection_name}_analysis.json")

if __name__ == "__main__":
    main() 