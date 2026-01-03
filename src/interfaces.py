"""Defines the abstract RagStrategy class."""

from abc import ABC, abstractmethod


class RagStrategy(ABC):
    """Abstract base class for RAG (Retrieval-Augmented Generation) strategies."""
    
    @abstractmethod
    def retrieve(self, query: str) -> str:
        """Retrieve relevant context for the given query.
        
        Args:
            query: The query string to retrieve context for.
            
        Returns:
            Retrieved context as a string.
        """
        pass
    
    @abstractmethod
    def generate(self, query: str, context: str) -> str:
        """Generate a response using the query and context.
        
        Args:
            query: The query string.
            context: The retrieved context.
            
        Returns:
            Generated response as a string.
        """
        pass

