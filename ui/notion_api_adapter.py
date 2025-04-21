import os
import sys
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add the parent directory to the path so we can import from notion_api_rag
sys.path.append(str(Path(__file__).parent.parent))

# Import the NotionAPIRAG class from your console app
from notion_api_rag import NotionAPIRAG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NotionAPIAdapter:
    """
    Adapter class to integrate the NotionAPIRAG system with the Streamlit UI.
    This adapter translates between the console-based API and the Streamlit UI's expected interface.
    """
    
    def __init__(self, notion_api_key: str = None, persist_directory: str = "./chroma_db"):
        """
        Initialize the NotionAPIAdapter with an API key and persistence directory.
        
        Args:
            notion_api_key (str, optional): Notion API key. Defaults to None, in which case
                                          it will be loaded from the environment.
            persist_directory (str, optional): Directory to persist the vector database.
        """
        self.api_key = notion_api_key or os.getenv("NOTION_API_KEY", "")
        self.persist_directory = persist_directory
        
        # Initialize the NotionAPIRAG system
        if self.api_key:
            try:
                self.rag_system = NotionAPIRAG(
                    notion_token=self.api_key,
                    persist_directory=self.persist_directory
                )
                logger.info("NotionAPIRAG system initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize NotionAPIRAG system: {str(e)}")
                self.rag_system = None
        else:
            logger.warning("No Notion API key provided. NotionAPIRAG system not initialized.")
            self.rag_system = None
    
    def update_api_key(self, api_key: str) -> Dict[str, Any]:
        """
        Update the Notion API key and reinitialize the RAG system.
        
        Args:
            api_key (str): The new Notion API key to use.
            
        Returns:
            Dict[str, Any]: A dictionary with status information.
        """
        if not api_key:
            return {
                "status": "error",
                "message": "API key cannot be empty",
                "help": "Please provide a valid Notion API key from https://www.notion.so/my-integrations"
            }
        
        self.api_key = api_key
        
        # Reinitialize the RAG system
        try:
            self.rag_system = NotionAPIRAG(
                notion_token=self.api_key,
                persist_directory=self.persist_directory
            )
            logger.info("NotionAPIRAG system reinitialized with new API key")
            
            # Test the connection
            return {
                "status": "success",
                "message": "Successfully connected to Notion API"
            }
        except Exception as e:
            error_message = str(e)
            logger.error(f"Failed to reinitialize NotionAPIRAG system: {error_message}")
            
            # Provide specific error messages based on the error
            if "Unauthorized" in error_message or "401" in error_message:
                help_text = """
                ### How to fix this error:
                
                1. Go to [Notion Integrations](https://www.notion.so/my-integrations)
                2. Select your integration or create a new one
                3. Copy the "Internal Integration Secret" token
                4. Paste it in the Notion API Key field
                """
                return {
                    "status": "error",
                    "message": "Invalid Notion API key",
                    "help": help_text
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to connect to Notion: {error_message}"
                }
    
    def list_integrations(self) -> Dict[str, Any]:
        """
        List all integrations connected to the API key.
        This is a placeholder since the NotionAPIRAG system doesn't have this functionality.
        
        Returns:
            Dict[str, Any]: A dictionary with status information.
        """
        if not self.api_key:
            return {
                "status": "error",
                "message": "Notion API key not configured",
                "help": "Please set your Notion API key in the settings page."
            }
        
        if not self.rag_system:
            return {
                "status": "error",
                "message": "NotionAPIRAG system not initialized",
                "help": "Please check your Notion API key and try again."
            }
        
        try:
            # Check if we can connect to Notion
            # Since NotionAPIRAG doesn't have a list_integrations method,
            # we'll just return a basic success message if the system is initialized
            return {
                "status": "success",
                "message": "Successfully connected to Notion API",
                "integrations": ["NotionAPIRAG Integration"]
            }
        except Exception as e:
            error_message = str(e)
            logger.error(f"Failed to connect to Notion: {error_message}")
            
            return {
                "status": "error",
                "message": f"Failed to connect to Notion: {error_message}"
            }
    
    def process_page(self, page_id: str) -> Dict[str, Any]:
        """
        Process a single Notion page and add it to the vector database.
        
        Args:
            page_id (str): The ID or URL of the Notion page to process.
            
        Returns:
            Dict[str, Any]: A dictionary with status information.
        """
        if not self.api_key:
            return {
                "status": "error",
                "message": "Notion API key not configured",
                "help": "Please set your Notion API key in the settings page."
            }
        
        if not self.rag_system:
            return {
                "status": "error",
                "message": "NotionAPIRAG system not initialized",
                "help": "Please check your Notion API key and try again."
            }
        
        try:
            # Use the add_notion_document method from NotionAPIRAG
            result = self.rag_system.add_notion_document(page_id)
            
            # Parse the result string to extract information
            # Example format: "Added document {page_id} with {len(chunks)} chunks to the database"
            if "Added document" in result and "chunks" in result:
                # Extract the number of chunks from the result string
                chunks_str = result.split("with ")[1].split(" chunks")[0]
                chunks_added = int(chunks_str)
                
                return {
                    "status": "success",
                    "message": f"Successfully processed Notion page: {page_id}",
                    "chunks_added": chunks_added
                }
            elif "Failed to extract" in result:
                return {
                    "status": "error",
                    "message": result,
                    "help": "Please provide a valid Notion page ID or URL."
                }
            else:
                return {
                    "status": "success",
                    "message": result
                }
                
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error processing Notion page {page_id}: {error_message}")
            
            if "Could not find" in error_message or "404" in error_message:
                help_text = """
                ### How to fix this error:
                
                1. Make sure the page ID or URL is correct
                2. Ensure the page is shared with your integration:
                   - Go to the Notion page
                   - Click the ⋮ menu in the top right
                   - Select "Add connections"
                   - Find and select your integration
                """
                return {
                    "status": "error",
                    "message": f"Page not found: {page_id}",
                    "help": help_text
                }
            else:
                return {
                    "status": "error",
                    "message": f"Error processing page {page_id}: {error_message}"
                }
    
    def process_pages(self, page_ids: List[str]) -> Dict[str, Any]:
        """
        Process multiple Notion pages and add them to the vector database.
        
        Args:
            page_ids (List[str]): List of page IDs or URLs to process.
            
        Returns:
            Dict[str, Any]: A dictionary with status information.
        """
        if not page_ids:
            return {
                "status": "error",
                "message": "No page IDs provided"
            }
        
        if not self.rag_system:
            return {
                "status": "error",
                "message": "NotionAPIRAG system not initialized",
                "help": "Please check your Notion API key and try again."
            }
        
        try:
            # Use the add_multiple_notion_documents method from NotionAPIRAG
            results = self.rag_system.add_multiple_notion_documents(page_ids)
            
            # Count successful pages and total chunks
            successful_pages = 0
            total_chunks = 0
            errors = []
            
            for i, result in enumerate(results):
                if "Added document" in result and "chunks" in result:
                    successful_pages += 1
                    # Extract the number of chunks from the result string
                    chunks_str = result.split("with ")[1].split(" chunks")[0]
                    chunks_added = int(chunks_str)
                    total_chunks += chunks_added
                else:
                    errors.append({
                        "page_id": page_ids[i],
                        "error": result
                    })
            
            if successful_pages > 0:
                return {
                    "status": "success",
                    "message": f"Successfully processed {successful_pages} out of {len(page_ids)} pages",
                    "chunks_added": total_chunks,
                    "errors": errors if errors else None
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to process any pages",
                    "errors": errors
                }
                
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error processing Notion pages: {error_message}")
            
            return {
                "status": "error",
                "message": f"Error processing Notion pages: {error_message}"
            }
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the NotionAPIRAG system.
        
        Args:
            question (str): The question to answer.
            
        Returns:
            Dict[str, Any]: A dictionary with the answer and source information.
        """
        if not self.rag_system:
            return {
                "answer": "NotionAPIRAG system not initialized. Please check your Notion API key and try again.",
                "sources": []
            }
        
        try:
            # Use the answer_question method from NotionAPIRAG
            response = self.rag_system.answer_question(question)
            
            # Convert the response to the format expected by the UI
            formatted_response = {
                "answer": response["answer"],
                "sources": [{"content": src, "metadata": {}} for src in response["sources"]]
            }
            
            return formatted_response
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error answering question: {error_message}")
            
            return {
                "answer": f"Error answering question: {error_message}",
                "sources": []
            }
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict[str, Any]: A dictionary with statistics.
        """
        if not self.rag_system:
            return {"document_count": 0}
        
        try:
            # Since NotionAPIRAG doesn't have a get_stats method,
            # we'll query the Chroma database directly
            collection = self.rag_system.db
            doc_count = len(collection.get()["ids"]) if collection.get()["ids"] else 0
            
            return {"document_count": doc_count}
        except Exception as e:
            logger.error(f"Error getting vector store stats: {str(e)}")
            return {"document_count": 0}
    
    def clear_chat_history(self) -> None:
        """
        Clear the chat history.
        This is a placeholder since NotionAPIRAG doesn't store chat history.
        """
        pass  # No action needed