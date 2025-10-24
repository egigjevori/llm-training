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
from inference import load_model, load_merged_model, generate_response

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
    
    def __init__(self, collection_name: str = "uk_companies", top_k: int = 3):
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
                    company_number = metadata.get('company_number', 'Unknown')

                    response_parts.append(f"\n**Result {i}** (Relevance: {score:.2f})")
                    response_parts.append(f"Company: {company_name}")
                    response_parts.append(f"Company Number: {company_number}")

                    # Add some key fields from UK companies document
                    if isinstance(doc_data, dict):
                        for key in ['company_name', 'status', 'incorporation_date', 'address', 'category', 'sic_codes']:
                            if key in doc_data and doc_data[key]:
                                if key == 'sic_codes' and isinstance(doc_data[key], list):
                                    response_parts.append(f"SIC Codes: {', '.join(doc_data[key])}")
                                elif key == 'address' and isinstance(doc_data[key], dict):
                                    full_address = doc_data[key].get('full_address', '')
                                    if full_address:
                                        response_parts.append(f"Address: {full_address}")
                                else:
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
                 base_model: str = "gpt2",
                 use_merged: bool = False,
                 temperature: float = 0.7,
                 max_new_tokens: int = 200,
                 do_sample: bool = True):
        """
        Initialize model provider.

        Args:
            model_path: Path to the model directory
            base_model: Base model name (only used if use_merged=False)
            use_merged: If True, load as merged model (no PEFT). If False, load with PEFT adapters.
            temperature: Sampling temperature (0.0-1.0)
            max_new_tokens: Maximum tokens to generate
            do_sample: Use sampling vs greedy decoding
        """
        self.model_path = model_path
        self.base_model = base_model
        self.use_merged = use_merged
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.model = None
        self.tokenizer = None

    def _ensure_model(self):
        """Lazy load the fine-tuned model."""
        if self.model is None or self.tokenizer is None:
            if self.use_merged:
                logger.info("Loading merged model (no PEFT)...")
                self.model, self.tokenizer = load_merged_model(self.model_path)
                logger.info("Merged model loaded successfully")
            else:
                logger.info("Loading fine-tuned model with PEFT adapters...")
                self.model, self.tokenizer = load_model(self.model_path, self.base_model)
                logger.info("Fine-tuned model loaded successfully")
    
    def get_response(self, query: str, **kwargs) -> Dict[str, Any]:
        """Get response using the fine-tuned model."""
        try:
            self._ensure_model()

            # Use instance parameters, but allow override via kwargs
            max_new_tokens = kwargs.get('max_new_tokens', self.max_new_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            do_sample = kwargs.get('do_sample', self.do_sample)

            # Generate response
            response = generate_response(
                self.model,
                self.tokenizer,
                query,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample
            )

            return {
                'response': response,
                'source': 'model',
                'metadata': {
                    'model_path': self.model_path,
                    'base_model': self.base_model,
                    'use_merged': self.use_merged,
                    'max_new_tokens': max_new_tokens,
                    'temperature': temperature,
                    'do_sample': do_sample
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


class HybridProvider(ResponseProvider):
    """Hybrid provider that combines RAG retrieval with fine-tuned model generation."""

    def __init__(self,
                 collection_name: str = "uk_companies",
                 top_k: int = 3,
                 model_path: str = "../models/finetuned_model",
                 base_model: str = "gpt2",
                 use_context_threshold: float = 0.5,
                 use_merged: bool = False,
                 temperature: float = 0.7,
                 max_new_tokens: int = 200,
                 do_sample: bool = True):
        """
        Initialize hybrid provider.

        Args:
            collection_name: Qdrant collection name
            top_k: Number of documents to retrieve
            model_path: Path to fine-tuned model
            base_model: Base model name (only used if use_merged=False)
            use_context_threshold: Minimum similarity score to use retrieved context
            use_merged: If True, load merged model (no PEFT). If False, load with PEFT adapters.
            temperature: Sampling temperature (0.0-1.0)
            max_new_tokens: Maximum tokens to generate
            do_sample: Use sampling vs greedy decoding
        """
        # Initialize RAG component
        self.rag_provider = QdrantProvider(collection_name=collection_name, top_k=top_k)

        # Initialize Model component
        self.model_provider = ModelProvider(
            model_path=model_path,
            base_model=base_model,
            use_merged=use_merged,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )

        # Configuration
        self.use_context_threshold = use_context_threshold
        self.collection_name = collection_name
        self.top_k = top_k

    def _format_context(self, rag_results: List[Dict], query: str) -> str:
        """Format retrieved documents as context for the model."""
        if not rag_results:
            return query

        # Extract relevant information from RAG results
        context_parts = []
        context_parts.append("Based on the following company information:")

        for i, result in enumerate(rag_results, 1):
            score = result.get('score', 0)
            text = result.get('text', '')
            metadata = result.get('metadata', {})

            # Only include high-confidence results
            if score >= self.use_context_threshold:
                try:
                    # Try to parse as JSON company data
                    doc_data = json.loads(text)
                    company_name = metadata.get('company_name', doc_data.get('company_name', 'Unknown'))

                    context_parts.append(f"\nCompany {i}: {company_name}")

                    # Add key company details
                    key_fields = ['company_number', 'status', 'incorporation_date', 'category', 'address']
                    for field in key_fields:
                        if field in doc_data and doc_data[field]:
                            if field == 'address' and isinstance(doc_data[field], dict):
                                address_val = doc_data[field].get('full_address', '')
                                if address_val:
                                    context_parts.append(f"- {field.replace('_', ' ').title()}: {address_val}")
                            else:
                                context_parts.append(f"- {field.replace('_', ' ').title()}: {doc_data[field]}")

                except (json.JSONDecodeError, KeyError):
                    # Fallback to text content
                    context_parts.append(f"\nDocument {i}: {text[:200]}...")

        # Combine context with query
        if len(context_parts) > 1:  # We have actual context
            context_str = '\n'.join(context_parts)
            formatted_prompt = f"{context_str}\n\nQuestion: {query}"
            return formatted_prompt
        else:
            # No good context found, use original query
            return query

    def get_response(self, query: str, **kwargs) -> Dict[str, Any]:
        """Get hybrid response combining RAG retrieval and model generation."""
        try:
            # Step 1: Get RAG results
            logger.info(f"Hybrid: Retrieving context for query: {query}")
            rag_response = self.rag_provider.get_response(query, **kwargs)

            # Extract RAG results from metadata or reconstruct
            rag_metadata = rag_response.get('metadata', {})

            # Step 2: Search for similar documents directly to get raw results
            self.rag_provider._ensure_model()
            query_embedding = self.rag_provider.embedding_model.encode([query])[0].tolist()

            from utils.qdrant import search_similar
            raw_results = search_similar(self.collection_name, query_embedding, self.top_k)

            # Step 3: Format context for the model
            context_prompt = self._format_context(raw_results, query)

            # Step 4: Generate response using fine-tuned model
            logger.info("Hybrid: Generating response with fine-tuned model")
            model_response = self.model_provider.get_response(
                context_prompt,
                **kwargs
            )

            # Step 5: Determine response quality and source
            has_good_context = any(r.get('score', 0) >= self.use_context_threshold for r in raw_results)

            response_text = model_response.get('response', '')

            # Step 6: Combine metadata from both sources
            hybrid_metadata = {
                'source': 'hybrid',
                'rag_metadata': rag_metadata,
                'model_metadata': model_response.get('metadata', {}),
                'context_used': has_good_context,
                'context_threshold': self.use_context_threshold,
                'retrieved_docs': len(raw_results),
                'high_confidence_docs': len([r for r in raw_results if r.get('score', 0) >= self.use_context_threshold])
            }

            return {
                'response': response_text,
                'source': 'hybrid',
                'metadata': hybrid_metadata
            }

        except Exception as e:
            logger.error(f"Error in HybridProvider: {e}")

            # Fallback to model-only response
            logger.info("Hybrid: Falling back to model-only response")
            try:
                fallback_response = self.model_provider.get_response(query, **kwargs)
                fallback_response['metadata']['fallback_reason'] = f"Hybrid error: {str(e)}"
                fallback_response['source'] = 'hybrid_fallback'
                return fallback_response
            except Exception as fallback_error:
                return {
                    'response': f"Sorry, I encountered errors in both RAG and model components: {str(e)} | {str(fallback_error)}",
                    'source': 'hybrid_error',
                    'metadata': {'primary_error': str(e), 'fallback_error': str(fallback_error)}
                }

    def get_name(self) -> str:
        return "Hybrid RAG + Fine-tuned Model"



def get_provider(mode: str, **kwargs) -> ResponseProvider:
    """Factory function to get the appropriate provider."""
    providers = {
        'qdrant': QdrantProvider,
        'model': ModelProvider,
        'hybrid': HybridProvider
    }

    if mode not in providers:
        raise ValueError(f"Unknown provider mode: {mode}. Available: {list(providers.keys())}")

    return providers[mode](**kwargs)