import os
import logging
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore as LangchainVectorStore
from langchain_core.retrievers import BaseRetriever
import chromadb
from chromadb.config import Settings
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """
    Class for handling vector storage and retrieval using Chroma.
    """
    
    def __init__(
        self, 
        embedding_function: Embeddings,
        persist_directory: str,
        collection_name: str = "default_collection"
    ):
        """
        Initialize the Chroma vector store.
        
        Args:
            embedding_function: Function to convert text to embeddings
            persist_directory: Directory to persist the vector store
            collection_name: Name of the collection
        """
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create the persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize the vector store
        self.vector_store = self._initialize_vector_store()
        logger.info(f"ChromaVectorStore initialized with collection '{collection_name}'")
    
    def _initialize_vector_store(self) -> VectorStore:
        """
        Initialize the vector store with the specified settings.
        
        Returns:
            The initialized vector store
        """
        try:
            # Create vector store
            vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            logger.warning("No documents provided to add to vector store")
            return
        
        try:
            logger.info(f"Adding {len(documents)} documents to collection '{self.collection_name}'")
            self.vector_store.add_documents(documents)
            logger.info(f"Successfully added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        search_type: str = "similarity",
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """
        Perform a similarity search for the query.
        
        Args:
            query: Query string
            k: Number of documents to return
            search_type: Type of search (similarity or mmr)
            fetch_k: Number of documents to fetch for MMR (only used if search_type is mmr)
            lambda_mult: MMR diversity parameter (only used if search_type is mmr)
            
        Returns:
            List of similar documents
        """
        try:
            logger.info(f"Performing {search_type} search for query: '{query}'")
            
            if search_type == "mmr":
                results = self.vector_store.max_marginal_relevance_search(
                    query,
                    k=k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult
                )
            else:  # default to similarity
                results = self.vector_store.similarity_search(query, k=k)
                
            logger.info(f"Found {len(results)} results for query")
            return results
            
        except AttributeError:
            # Fallback to regular similarity search if MMR is not available
            logger.warning("MMR search not available, falling back to similarity search")
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def clear(self) -> None:
        """
        Clear all documents from the vector store.
        """
        try:
            logger.info(f"Clearing all documents from collection '{self.collection_name}'")
            # Get the underlying Chroma client
            client = self.vector_store._client
            
            # Delete the collection
            client.delete_collection(self.collection_name)
            
            # Reinitialize the vector store
            self.vector_store = self._initialize_vector_store()
            
            logger.info(f"Successfully cleared collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        try:
            # Get the underlying Chroma client
            client = self.vector_store._client
            
            # Get the collection
            collection = client.get_collection(self.collection_name)
            
            # Get collection stats
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {str(e)}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "persist_directory": self.persist_directory,
                "error": str(e)
            }

class VectorStore:
    """
    Vector store for document storage and retrieval.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None,
        collection_name: str = "notion_docs"
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding_model_name: Name of the HuggingFace embedding model to use
            persist_directory: Directory to persist the database (if None, use memory)
            collection_name: Name of the collection in the vector database
        """
        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embedding function
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Initialize vector store
        self._init_vector_store()
        
        logger.info(f"Initialized VectorStore with model={embedding_model_name}, collection={collection_name}")
    
    def _init_vector_store(self) -> None:
        """Initialize the vector store with chosen backend."""
        if self.persist_directory:
            os.makedirs(self.persist_directory, exist_ok=True)
            
        self.db = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add to vector store")
            return []
        
        # Generate IDs for documents if they don't have one
        ids = []
        for doc in documents:
            if "id" not in doc.metadata:
                doc.metadata["id"] = str(uuid.uuid4())
            ids.append(doc.metadata["id"])
        
        try:
            self.db.add_documents(documents=documents, ids=ids)
            logger.info(f"Added {len(documents)} documents to vector store")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return []
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Query string
            k: Number of results to return
            filter: Filter to apply to search
            
        Returns:
            List of documents
        """
        try:
            docs = self.db.similarity_search(query=query, k=k, filter=filter)
            logger.info(f"Found {len(docs)} documents for query: {query[:50]}...")
            return docs
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []
    
    def mmr_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform Maximum Marginal Relevance search for diversity.
        
        Args:
            query: Query string
            k: Number of results to return
            fetch_k: Number of documents to consider for diversity
            lambda_mult: Diversity vs relevance tradeoff (0=max diversity, 1=max relevance)
            filter: Filter to apply to search
            
        Returns:
            List of documents
        """
        try:
            docs = self.db.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter
            )
            logger.info(f"Found {len(docs)} documents using MMR search for query: {query[:50]}...")
            return docs
        except Exception as e:
            logger.error(f"Error performing MMR search: {str(e)}")
            return []
    
    def get_retriever(self, search_type: str = "mmr", search_kwargs: Optional[Dict[str, Any]] = None) -> BaseRetriever:
        """
        Get a retriever interface for the vector store.
        
        Args:
            search_type: Type of search to perform ('similarity' or 'mmr')
            search_kwargs: Arguments for the search
            
        Returns:
            Retriever interface
        """
        if search_type == "mmr":
            search_kwargs = search_kwargs or {
                "k": 4,
                "fetch_k": 20,
                "lambda_mult": 0.5
            }
            return self.db.as_retriever(
                search_type="mmr",
                search_kwargs=search_kwargs
            )
        else:
            search_kwargs = search_kwargs or {"k": 4}
            return self.db.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        try:
            self.db.delete_collection()
            self._init_vector_store()
            logger.info(f"Cleared collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
    
    def count(self) -> int:
        """Get the number of documents in the vector store."""
        try:
            return self.db._collection.count()
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            return 0 