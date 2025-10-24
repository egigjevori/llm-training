#!/usr/bin/env python3
"""Demo script for vector search with Qdrant."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.qdrant import search_similar, get_embedding_model

def main():
    """Run vector search demo."""
    collection_name = 'uk_companies'
    query = 'technology companies in London'
    top_k = 3

    print(f"\n{'='*60}")
    print(f"  Vector Search Demo")
    print(f"{'='*60}\n")
    print(f"Collection: {collection_name}")
    print(f"Query: '{query}'")
    print(f"Top K: {top_k}\n")
    print("Loading embedding model...")

    # Load embedding model
    embedding_model = get_embedding_model()
    print("✓ Model loaded\n")

    # Generate query embedding
    print("Generating query embedding...")
    query_embedding = embedding_model.encode([query])[0].tolist()
    print(f"✓ Embedding generated ({len(query_embedding)} dimensions)\n")

    # Search for similar documents
    print("Searching Qdrant...")
    results = search_similar(collection_name, query_embedding, top_k)
    print(f"✓ Found {len(results)} results\n")

    # Display results
    print(f"{'='*60}")
    print("  Results")
    print(f"{'='*60}\n")

    for i, result in enumerate(results, 1):
        score = result.get('score', 0)
        metadata = result.get('metadata', {})
        company_name = metadata.get('company_name', 'N/A')

        print(f"{i}. {company_name}")
        print(f"   Score: {score:.3f}")
        print(f"   Relevance: {'⭐⭐⭐ Very High' if score >= 0.8 else '⭐⭐ High' if score >= 0.5 else '⭐ Medium'}")
        print()

    print(f"{'='*60}\n")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
