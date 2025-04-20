import os
import yaml
import logging
import uuid
import json
import tempfile
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ui.notion_manager import NotionManager
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from ui.document_processor import DocumentProcessor
from ui.vector_store import ChromaVectorStore
from ui.notion_loader import NotionLoader
from ui.notion_processor import NotionProcessor
import streamlit as st

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGApp:
    """
    Retrieval-Augmented Generation (RAG) application that integrates document processing,
    vector storage, and generative AI for question answering.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the RAG application.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.initialize_components()
        
        logger.info("RAG application initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
            else:
                logger.warning(f"Configuration file {config_path} not found. Using default configuration.")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "gemini": {
                "model": "gemini-pro",
                "temperature": 0.2,
                "max_output_tokens": 2048,
                "top_p": 0.95,
                "top_k": 40
            },
            "chroma": {
                "persist_directory": "./chroma_db",
                "collection_name": "rag_collection"
            },
            "embeddings": {
                "model_name": "all-MiniLM-L6-v2",
                "model_kwargs": {"device": "cpu"}
            },
            "document_processor": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "separators": ["\n\n", "\n", " ", ""]
            }
        }
    
    def initialize_components(self):
        """
        Initialize the components of the RAG application.
        """
        try:
            # Get API key from environment variable
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("GOOGLE_API_KEY not found in environment variables")
                api_key = self.config.get("gemini", {}).get("api_key")
            
            # Configure Gemini API
            if api_key:
                genai.configure(api_key=api_key)
                
                # Initialize Gemini model
                model_name = self.config.get("gemini", {}).get("model", "gemini-pro")
                self.generation_config = {
                    "temperature": self.config.get("gemini", {}).get("temperature", 0.2),
                    "max_output_tokens": self.config.get("gemini", {}).get("max_output_tokens", 2048),
                    "top_p": self.config.get("gemini", {}).get("top_p", 0.95),
                    "top_k": self.config.get("gemini", {}).get("top_k", 40)
                }
                self.model = genai.GenerativeModel(model_name, generation_config=self.generation_config)
                logger.info(f"Initialized Gemini model: {model_name}")
            else:
                logger.error("No API key provided for Gemini. Text generation will not work.")
                self.model = None
            
            # Initialize embeddings
            embeddings_config = self.config.get("embeddings", {})
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embeddings_config.get("model_name", "all-MiniLM-L6-v2"),
                model_kwargs=embeddings_config.get("model_kwargs", {"device": "cpu"})
            )
            logger.info(f"Initialized embeddings with model: {embeddings_config.get('model_name')}")
            
            # Initialize vector store
            chroma_config = self.config.get("chroma", {})
            self.vector_store = ChromaVectorStore(
                embedding_function=self.embeddings,
                persist_directory=chroma_config.get("persist_directory", "./chroma_db"),
                collection_name=chroma_config.get("collection_name", "rag_collection")
            )
            logger.info(f"Initialized vector store with collection: {chroma_config.get('collection_name')}")
            
            # Initialize document processor
            doc_processor_config = self.config.get("document_processor", {})
            self.document_processor = DocumentProcessor(
                chunk_size=doc_processor_config.get("chunk_size", 1000),
                chunk_overlap=doc_processor_config.get("chunk_overlap", 200),
                separators=doc_processor_config.get("separators", ["\n\n", "\n", " ", ""])
            )
            logger.info("Initialized document processor")
            
            # Initialize Notion processor
            notion_api_key = os.getenv("NOTION_API_KEY")
            self.notion_processor = NotionProcessor(
                api_key=notion_api_key,
                document_processor=self.document_processor
            )
            logger.info("Initialized Notion processor")
                
        except Exception as e:
            logger.error(f"Error initializing RAG components: {str(e)}")
            raise
    
    def process_notion_pages(self, page_ids: List[str]) -> int:
        """
        Process Notion pages and add them to the vector store.
        
        Args:
            page_ids: List of Notion page IDs
            
        Returns:
            Number of pages processed
        """
        if not self.notion_loader:
            logger.error("Notion loader not initialized. Cannot process Notion pages.")
            return 0
        
        try:
            logger.info(f"Processing {len(page_ids)} Notion pages")
            processed_count = 0
            
            for page_id in page_ids:
                try:
                    # Get the page document
                    document = self.notion_loader.get_page(page_id)
                    
                    # Process the document
                    chunks = self.document_processor.process_document(document)
                    
                    # Add chunks to vector store
                    self.vector_store.add_documents(chunks)
                    
                    processed_count += 1
                    logger.info(f"Successfully processed Notion page: {page_id}")
                except Exception as e:
                    logger.error(f"Error processing Notion page {page_id}: {str(e)}")
            
            return processed_count
            
        except Exception as e:
            logger.error(f"Error processing Notion pages: {str(e)}")
            return 0
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Process documents and add them to the vector store.
        
        Args:
            documents: List of documents with 'text' and 'metadata' fields
            
        Returns:
            Number of documents processed
        """
        try:
            logger.info(f"Processing {len(documents)} documents")
            
            # Convert to Document objects
            doc_objects = [
                Document(
                    page_content=doc.get("text", ""),
                    metadata=doc.get("metadata", {})
                ) for doc in documents
            ]
            
            # Process documents
            chunks = self.document_processor.process_documents(doc_objects)
            
            # Add chunks to vector store
            self.vector_store.add_documents(chunks)
            
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return 0
    
    def get_response(self, query: str, search_type: str = "similarity", k: int = 4) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get a response to a query using RAG.
        
        Args:
            query: User query
            search_type: Type of search ('similarity' or 'mmr')
            k: Number of documents to retrieve
            
        Returns:
            Tuple of (response text, list of source documents)
        """
        if not self.model:
            return "Error: Text generation model not initialized. Please provide a valid API key.", []
        
        try:
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(query, k=k, search_type=search_type)
            
            if not docs:
                return "I couldn't find any relevant information to answer your question.", []
            
            # Format context from documents
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
            
            # Create prompt
            prompt = f"""You are a helpful assistant that answers questions based on the provided context.
            
Context:
{context}

Question: {query}

Instructions:
1. Answer the question based only on the provided context.
2. If the context doesn't contain the information needed to answer the question, say so.
3. Be concise, clear, and helpful in your response.
4. Do not reference the context in your answer (e.g., don't say "According to Document 1...").

Answer:"""
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Format source documents
            source_docs = []
            for i, doc in enumerate(docs):
                source_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            return response.text, source_docs
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}", []
    
    def clear_vector_store(self) -> bool:
        """
        Clear all documents from the vector store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vector_store.clear()
            logger.info("Vector store cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        try:
            return self.vector_store.get_stats()
        except Exception as e:
            logger.error(f"Error getting vector store stats: {str(e)}")
            return {"error": str(e)}

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Process and add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        try:
            # Process documents (chunk, clean, etc.)
            processed_docs = self.document_processor.process_documents(documents)
            
            # Add to vector store
            doc_ids = self.vector_store.add_documents(processed_docs)
            
            # Persist to disk
            self.vector_store.persist()
            
            logger.info(f"Added {len(documents)} documents to vector store, created {len(processed_docs)} chunks")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return []
    
    def load_from_notion(self, page_ids: List[str]) -> List[Document]:
        """
        Load content from Notion pages and add to the vector store.
        
        Args:
            page_ids: List of Notion page IDs
            
        Returns:
            List of loaded documents
        """
        if not self.notion_loader:
            logger.error("Notion loader not initialized. Set NOTION_API_KEY environment variable.")
            return []
        
        try:
            # Load documents from Notion
            documents = self.notion_loader.get_multiple_pages(page_ids)
            
            if documents:
                # Add documents to vector store
                self.add_documents(documents)
                
                logger.info(f"Loaded {len(documents)} documents from Notion")
                return documents
            else:
                logger.warning("No documents loaded from Notion")
                return []
                
        except Exception as e:
            logger.error(f"Error loading from Notion: {str(e)}")
            return []
    
    def check_notion_connection(self) -> Dict[str, Any]:
        """
        Check Notion connection status and list available integrations.
        
        Returns:
            Dictionary with connection status and integrations
        """
        if not self.notion_loader:
            return {
                "status": "not_configured",
                "error": "Notion API key not set",
                "help": "Set the NOTION_API_KEY environment variable"
            }
        
        try:
            return self.notion_loader.list_integrations()
        except Exception as e:
            logger.error(f"Error checking Notion connection: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def clear_documents(self) -> bool:
        """
        Clear all documents from the vector store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vector_store.delete_collection()
            logger.info("Cleared all documents from vector store")
            return True
        except Exception as e:
            logger.error(f"Error clearing documents: {str(e)}")
            return False
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the stored documents.
        
        Returns:
            Dictionary with statistics
        """
        try:
            stats = self.vector_store.get_collection_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}")
            return {
                "error": str(e)
            }
    
    def update_configuration(self, config_updates: Dict[str, Any], save_to_file: bool = True) -> bool:
        """
        Update configuration parameters.
        
        Args:
            config_updates: Dictionary with updated configuration values
            save_to_file: Whether to save the updated configuration to file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update configuration
            self.config.update(config_updates)
            
            # Update document processor if relevant configurations changed
            if any(key in config_updates for key in ["chunk_size", "chunk_overlap"]):
                self.document_processor.update_configuration(
                    chunk_size=self.config.get("chunk_size"),
                    chunk_overlap=self.config.get("chunk_overlap")
                )
            
            # Save to file if requested
            if save_to_file:
                with open("config.yaml", 'w') as file:
                    yaml.dump(self.config, file)
                    logger.info("Saved updated configuration to config.yaml")
            
            logger.info(f"Updated configuration: {config_updates}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            return False

    def index_notion_page(self, page_id: str) -> bool:
        """
        Index a Notion page.
        
        Args:
            page_id: The ID of the Notion page to index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clean page ID if it's a URL
            page_id = self._clean_notion_id(page_id)
            
            # Get page content from Notion
            page_content = self.notion_manager.get_page_content(page_id)
            
            if not page_content:
                logger.error(f"Failed to retrieve content for page: {page_id}")
                return False
            
            # Add metadata
            page_content["metadata"]["source_type"] = "page"
            page_content["metadata"]["indexed_at"] = datetime.now().isoformat()
            page_content["metadata"]["id"] = str(uuid.uuid4())
            
            # Split page content into chunks
            chunks = self.text_splitter.split_text(page_content["text"])
            
            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                # Create metadata for each chunk
                chunk_metadata = {
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "source": "notion",
                    "source_id": page_id,
                    "title": page_content["metadata"]["title"],
                    "url": page_content["metadata"]["url"],
                    "source_type": "page",
                    "indexed_at": page_content["metadata"]["indexed_at"],
                    "id": f"{page_content['metadata']['id']}-{i}"
                }
                
                # Add document to list
                documents.append((chunk, chunk_metadata))
            
            # Add documents to vector store
            ids = [doc[1]["id"] for doc in documents]
            texts = [doc[0] for doc in documents]
            metadatas = [doc[1] for doc in documents]
            
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Indexed Notion page: {page_id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing Notion page {page_id}: {str(e)}")
            return False
    
    def index_notion_database(self, database_id: str) -> bool:
        """
        Index a Notion database.
        
        Args:
            database_id: The ID of the Notion database to index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clean database ID if it's a URL
            database_id = self._clean_notion_id(database_id)
            
            # Get database content from Notion
            database_pages = self.notion_manager.get_database_content(database_id)
            
            if not database_pages:
                logger.error(f"Failed to retrieve content for database: {database_id}")
                return False
            
            # Track success status
            success = True
            
            # Index each page in the database
            for page in database_pages:
                page_success = self.index_notion_page(page["id"])
                if not page_success:
                    success = False
            
            logger.info(f"Indexed Notion database: {database_id} with {len(database_pages)} pages")
            return success
            
        except Exception as e:
            logger.error(f"Error indexing Notion database {database_id}: {str(e)}")
            return False
    
    def _clean_notion_id(self, notion_id: str) -> str:
        """
        Clean a Notion ID from a URL or ID string.
        
        Args:
            notion_id: The Notion ID or URL
            
        Returns:
            Cleaned Notion ID
        """
        # Extract ID from URL if needed
        if "notion.so" in notion_id:
            # Extract the 32-character ID from the URL
            import re
            match = re.search(r"([a-f0-9]{32})", notion_id)
            if match:
                return match.group(1)
        
        # Remove any non-alphanumeric characters
        return ''.join(c for c in notion_id if c.isalnum() or c == '-')
    
    def get_sources(self, docs) -> List[Dict[str, str]]:
        """
        Extract source information from retrieved documents.
        
        Args:
            docs: List of retrieved documents
            
        Returns:
            List of source dictionaries with relevant metadata
        """
        sources = []
        seen_sources = set()
        
        for doc in docs:
            metadata = doc.metadata
            source_id = metadata.get("source_id")
            
            # Skip duplicates
            if source_id in seen_sources:
                continue
            
            seen_sources.add(source_id)
            
            source = {
                "id": source_id,
                "name": metadata.get("title", "Untitled"),
                "url": metadata.get("url", ""),
                "type": metadata.get("source_type", "page")
            }
            
            sources.append(source)
        
        return sources
    
    def search_docs(self, query: str, k: int = 5) -> List:
        """
        Search for relevant documents in the vector store.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        try:
            # Perform similarity search with MMR to ensure diversity
            docs = self.vector_store.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=k*2,  # Fetch more candidates for diversity
                lambda_mult=0.7  # Balance between relevance and diversity
            )
            
            return docs
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_response(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a response based on a user query and relevant documents.
        
        Args:
            query: The user's query
            chat_history: Optional chat history for context
            temperature: Optional temperature override
            max_output_tokens: Optional token limit override
            
        Returns:
            Dictionary containing the answer and source information
        """
        if not self.model:
            return {
                "answer": "No LLM available. Please set the GOOGLE_API_KEY environment variable.",
                "sources": []
            }
        
        try:
            # Search for relevant documents
            docs = self.search_docs(query)
            
            if not docs:
                return {
                    "answer": "I couldn't find any relevant information in your documents. Try indexing more content or rephrasing your question.",
                    "sources": []
                }
            
            # Extract sources
            sources = self.get_sources(docs)
            
            # Create context from documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Format chat history if provided
            history_text = ""
            if chat_history and len(chat_history) > 0:
                for msg in chat_history[-4:]:  # Use last 4 messages for context
                    role = msg["role"]
                    content = msg["content"]
                    history_text += f"{role.capitalize()}: {content}\n"
            
            # Create prompt
            prompt = f"""
You are a helpful AI assistant that answers questions based on the provided context from Notion documents.

CONTEXT:
{context}

CHAT HISTORY:
{history_text}

USER QUERY:
{query}

Answer the user's query based on the provided context. If the answer is not in the context, say that you don't know.
Include relevant information from the context to support your answer. Be concise and to the point.
Do not mention that you're using context or documents in your answer.
"""
            
            # Set generation parameters
            generation_config = {
                "temperature": temperature if temperature is not None else self.config["temperature"],
                "max_output_tokens": max_output_tokens if max_output_tokens is not None else self.config["max_tokens"],
                "top_p": 0.95,
                "top_k": self.config["top_k"],
            }
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Return response with sources
            return {
                "answer": response.text,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your request: {str(e)}",
                "sources": []
            }
    
    def clear_chat_history(self) -> None:
        """Clear conversation chat history."""
        logger.info("Chat history cleared")

    def add_document(self, document: Union[Document, str], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a document to the knowledge base.
        
        Args:
            document: Document to add (either a Document object or a string)
            metadata: Optional metadata for the document
            
        Returns:
            Dictionary with status and information
        """
        try:
            # Convert string to Document if needed
            if isinstance(document, str):
                if metadata is None:
                    metadata = {}
                document = Document(
                    page_content=document,
                    metadata=metadata
                )
            
            # Process document
            chunks = self.document_processor.process_document(document)
            
            # Add to vector store
            add_result = self.vector_store.add_documents(chunks)
            
            return {
                "status": "success",
                "message": f"Added document with {len(chunks)} chunks",
                "chunks_added": len(chunks),
                "document_id": add_result.get("ids", [])[0] if add_result.get("ids") else None
            }
            
        except Exception as e:
            error_msg = f"Error adding document: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }
    
    def set_temperature(self, temperature: float) -> Dict[str, Any]:
        """
        Set the temperature for the model.
        
        Args:
            temperature: Temperature value (0.0 to 1.0)
            
        Returns:
            Dictionary with status and information
        """
        try:
            # Validate temperature
            if not 0.0 <= temperature <= 1.0:
                return {
                    "status": "error",
                    "message": "Temperature must be between 0.0 and 1.0"
                }
            
            # Update temperature
            self.generation_config["temperature"] = temperature
            
            return {
                "status": "success",
                "message": f"Temperature set to {temperature}",
                "temperature": temperature
            }
            
        except Exception as e:
            error_msg = f"Error setting temperature: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }
    
    def load_notion_page(self, page_id: str) -> Dict[str, Any]:
        """
        Load a Notion page and add it to the knowledge base.
        
        Args:
            page_id: The ID or URL of the Notion page to load
            
        Returns:
            Dictionary with status information
        """
        if not page_id:
            return {
                "status": "error",
                "message": "No page ID provided",
                "help": "Please provide a valid Notion page ID or URL"
            }
        
        logger.info(f"Loading Notion page: {page_id}")
        
        try:
            # Process the page using the notion processor
            result = self.notion_processor.process_page(page_id)
            
            if result["status"] == "success":
                # Add chunks to the vector store
                chunks = result.get("chunks", [])
                if chunks:
                    # Add documents to the vector store
                    doc_ids = self.add_documents(chunks)
                    logger.info(f"Added {len(doc_ids)} chunks to vector store")
                    
                    result["doc_ids"] = doc_ids
                
                return result
            else:
                return result
                
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error loading Notion page: {error_message}")
            
            return {
                "status": "error",
                "message": f"Failed to load Notion page: {error_message}",
                "help": "Please check your Notion API key and ensure the page is accessible."
            }
    
    def load_notion_pages(self, page_ids: List[str]) -> Dict[str, Any]:
        """
        Load multiple Notion pages and add them to the knowledge base.
        
        Args:
            page_ids: List of Notion page IDs or URLs
            
        Returns:
            Dictionary with status information
        """
        if not page_ids:
            return {
                "status": "error",
                "message": "No page IDs provided",
                "help": "Please provide at least one Notion page ID or URL"
            }
        
        logger.info(f"Loading {len(page_ids)} Notion pages")
        
        try:
            # Process the pages using the notion processor
            result = self.notion_processor.process_pages(page_ids)
            
            if result["status"] == "success":
                # Add chunks to the vector store
                chunks = result.get("chunks", [])
                if chunks:
                    # Add documents to the vector store
                    doc_ids = self.add_documents(chunks)
                    logger.info(f"Added {len(doc_ids)} chunks to vector store")
                    
                    result["doc_ids"] = doc_ids
                
                return result
            else:
                return result
                
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error loading Notion pages: {error_message}")
            
            return {
                "status": "error",
                "message": f"Failed to load Notion pages: {error_message}",
                "help": "Please check your Notion API key and ensure the pages are accessible."
            }
    
    def update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update application settings.
        
        Args:
            settings: Dictionary with settings to update
            
        Returns:
            Dictionary with update results
        """
        try:
            updated = {}
            
            # Update temperature if provided
            if "temperature" in settings:
                self.temperature = float(settings["temperature"])
                updated["temperature"] = self.temperature
            
            # Update Notion API key if provided
            if "notion_api_key" in settings and settings["notion_api_key"]:
                self.notion_api_key = settings["notion_api_key"]
                os.environ["NOTION_API_KEY"] = self.notion_api_key
                
                # Reinitialize Notion processor
                self.notion_processor = NotionProcessor(
                    api_key=self.notion_api_key,
                    document_processor=self.document_processor
                )
                
                updated["notion_api_key"] = "*****" + self.notion_api_key[-4:] if len(self.notion_api_key) > 4 else "*****"
            
            # Save settings to .env file if required
            if settings.get("save_to_env", False):
                self._save_settings_to_env(settings)
                updated["saved_to_env"] = True
            
            return {
                "success": True,
                "message": "Settings updated successfully",
                "updated": updated
            }
            
        except Exception as e:
            error_msg = f"Error updating settings: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg
            }
    
    def _save_settings_to_env(self, settings: Dict[str, Any]) -> None:
        """
        Save settings to .env file.
        
        Args:
            settings: Dictionary with settings to save
        """
        try:
            # Read existing .env file if it exists
            env_path = Path(".env")
            env_vars = {}
            
            if env_path.exists():
                with open(env_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            key, value = line.split("=", 1)
                            env_vars[key.strip()] = value.strip()
            
            # Update environment variables
            if "notion_api_key" in settings and settings["notion_api_key"]:
                env_vars["NOTION_API_KEY"] = settings["notion_api_key"]
            
            if "google_api_key" in settings and settings["google_api_key"]:
                env_vars["GOOGLE_API_KEY"] = settings["google_api_key"]
            
            # Save to .env file
            with open(env_path, "w") as f:
                for key, value in env_vars.items():
                    f.write(f"{key}={value}\n")
            
            logger.info("Settings saved to .env file")
            
        except Exception as e:
            logger.error(f"Error saving settings to .env file: {str(e)}")
            raise
    
    def clear_vector_store(self) -> Dict[str, Any]:
        """
        Clear all documents from the vector store.
        
        Returns:
            Dictionary with clear results
        """
        try:
            # Delete collection
            self.vector_store.delete_collection()
            
            # Reinitialize vector store
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Clear processed documents
            self.processed_docs = {}
            
            return {
                "success": True,
                "message": "Vector store cleared successfully"
            }
            
        except Exception as e:
            error_msg = f"Error clearing vector store: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg
            } 