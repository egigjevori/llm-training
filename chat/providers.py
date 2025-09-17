"""
Response providers for the chat CLI application.
Supports multiple modes: Qdrant search, fine-tuned model, and hybrid RAG.
"""

import logging
import sys
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.qdrant import search_similar, get_embedding_model
from inference import load_model, generate_response

logger = logging.getLogger(__name__)


class ResponseProvider(ABC):
    """Abstract base class for response providers."""
    
    @abstractmethod
    def get_response(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Get response for a query.
        
        Returns:
            Dict with keys: 'response', 'source', 'metadata'
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the provider name."""
        pass


class QdrantProvider(ResponseProvider):
    """Provider that uses Qdrant vector search for responses."""
    
    def __init__(self, collection_name: str = "corporate_data", top_k: int = 3):
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedding_model = None
        
    def _ensure_model(self):
        """Lazy load the embedding model."""
        if self.embedding_model is None:
            self.embedding_model = get_embedding_model()
    
    def get_response(self, query: str, **kwargs) -> Dict[str, Any]:
        """Get response using Qdrant vector search."""
        try:
            self._ensure_model()
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Search for similar documents
            results = search_similar(self.collection_name, query_embedding, self.top_k)
            
            if not results:
                return {
                    'response': "I couldn't find any relevant information in the database.",
                    'source': 'qdrant',
                    'metadata': {'collection': self.collection_name, 'results_found': 0}
                }
            
            # Format response
            response_parts = []
            response_parts.append(f"Based on the documents I found, here's what I can tell you:\n")
            
            for i, result in enumerate(results, 1):
                score = result.get('score', 0)
                metadata = result.get('metadata', {})
                text = result.get('text', '')
                
                # Try to parse JSON text for better formatting
                try:
                    doc_data = json.loads(text)
                    company_name = metadata.get('company_name', 'Unknown')
                    company_id = metadata.get('company_id', 'Unknown')
                    
                    response_parts.append(f"\n**Result {i}** (Relevance: {score:.2f})")
                    response_parts.append(f"Company: {company_name}")
                    response_parts.append(f"Company ID: {company_id}")
                    
                    # Add some key fields from the document
                    if isinstance(doc_data, dict):
                        for key in ['name', 'status', 'incorporation_date', 'address']:
                            if key in doc_data and doc_data[key]:
                                response_parts.append(f"{key.replace('_', ' ').title()}: {doc_data[key]}")
                except:
                    # If not JSON, just show the text
                    response_parts.append(f"\n**Result {i}** (Relevance: {score:.2f})")
                    response_parts.append(f"Content: {text[:200]}...")
            
            return {
                'response': '\n'.join(response_parts),
                'source': 'qdrant',
                'metadata': {
                    'collection': self.collection_name,
                    'results_found': len(results),
                    'top_score': results[0].get('score', 0) if results else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in QdrantProvider: {e}")
            return {
                'response': f"Sorry, I encountered an error while searching: {str(e)}",
                'source': 'qdrant',
                'metadata': {'error': str(e)}
            }
    
    def get_name(self) -> str:
        return "Qdrant Vector Search"


class ModelProvider(ResponseProvider):
    """Provider that uses the fine-tuned model for responses."""
    
    def __init__(self, model_path: str = "../models/finetuned_model", 
                 base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        
    def _ensure_model(self):
        """Lazy load the fine-tuned model."""
        if self.model is None or self.tokenizer is None:
            logger.info("Loading fine-tuned model...")
            self.model, self.tokenizer = load_model(self.model_path, self.base_model)
            logger.info("Fine-tuned model loaded successfully")
    
    def get_response(self, query: str, **kwargs) -> Dict[str, Any]:
        """Get response using the fine-tuned model."""
        try:
            self._ensure_model()
            
            # Extract parameters
            max_new_tokens = kwargs.get('max_new_tokens', 200)
            temperature = kwargs.get('temperature', 0.7)
            
            # Generate response
            response = generate_response(
                self.model, 
                self.tokenizer, 
                query,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            return {
                'response': response,
                'source': 'model',
                'metadata': {
                    'model_path': self.model_path,
                    'base_model': self.base_model,
                    'max_new_tokens': max_new_tokens,
                    'temperature': temperature
                }
            }
            
        except Exception as e:
            logger.error(f"Error in ModelProvider: {e}")
            return {
                'response': f"Sorry, I encountered an error generating a response: {str(e)}",
                'source': 'model',
                'metadata': {'error': str(e)}
            }
    
    def get_name(self) -> str:
        return "Fine-tuned Model"



def get_provider(mode: str, **kwargs) -> ResponseProvider:
    """Factory function to get the appropriate provider."""
    providers = {
        'qdrant': QdrantProvider,
        'model': ModelProvider
    }
    
    if mode not in providers:
        raise ValueError(f"Unknown provider mode: {mode}. Available: {list(providers.keys())}")
    
    return providers[mode](**kwargs)