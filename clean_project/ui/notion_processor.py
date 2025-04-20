import os
import re
import logging
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Import custom modules for document processing
from ui.document_processor import DocumentProcessor

# Try to import the NotionManager - we'll use it if available
try:
    from ui.notion_manager import NotionManager
    NOTION_MANAGER_AVAILABLE = True
except ImportError:
    NOTION_MANAGER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NotionProcessor:
    """
    Class to handle loading and processing of Notion content for the RAG system.
    This class provides methods to fetch content from Notion pages and prepare
    them for ingestion into the vector database.
    """
    
    def __init__(self, api_key: str = None, document_processor: DocumentProcessor = None):
        """
        Initialize the NotionProcessor with an API key and document processor.
        
        Args:
            api_key (str, optional): Notion API key. Defaults to None, in which case
                                    it will be loaded from the environment.
            document_processor (DocumentProcessor, optional): Instance of DocumentProcessor
                                                            to process the documents.
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("NOTION_API_KEY", "")
        
        # Initialize the notion manager if the module is available
        if NOTION_MANAGER_AVAILABLE:
            self.notion_manager = NotionManager(api_key=self.api_key)
        else:
            self.notion_manager = None
            logger.warning("NotionManager not available. Some functions may be limited.")
        
        # Use provided document processor or create a new one
        self.document_processor = document_processor or DocumentProcessor()
    
    def update_api_key(self, api_key: str) -> Dict[str, Any]:
        """
        Update the Notion API key and verify it works.
        
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
        
        # Update the notion manager if available
        if self.notion_manager:
            self.notion_manager.api_key = api_key
        
        # Test the connection
        return self.list_integrations()
    
    def list_integrations(self) -> Dict[str, Any]:
        """
        List all integrations connected to the API key.
        
        Returns:
            Dict[str, Any]: A dictionary with status information and a list of integrations.
        """
        if not self.api_key:
            return {
                "status": "error",
                "message": "Notion API key not configured",
                "help": "Please set your Notion API key in the settings page."
            }
        
        if not self.notion_manager:
            return {
                "status": "error",
                "message": "NotionManager not available",
                "help": "The NotionManager module is required for this functionality."
            }
        
        try:
            # Test the connection by listing all integrations
            integrations = self.notion_manager.list_integrations()
            
            return {
                "status": "success",
                "message": f"Successfully connected to Notion. Found {len(integrations)} integrations.",
                "integrations": integrations
            }
        except Exception as e:
            error_message = str(e)
            
            # Provide helpful error messages based on the type of error
            if "Invalid API key" in error_message or "401" in error_message:
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
            elif "403" in error_message:
                help_text = """
                ### How to fix this error:
                
                Your integration doesn't have access to any pages. You need to:
                
                1. Go to the Notion page you want to access
                2. Click the ⋮ menu in the top right
                3. Select "Add connections"
                4. Find and select your integration
                """
                return {
                    "status": "error",
                    "message": "No pages shared with this integration",
                    "help": help_text
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to connect to Notion: {error_message}"
                }
    
    def process_page(self, page_id: str) -> Dict[str, Any]:
        """
        Process a single Notion page and prepare it for ingestion.
        
        Args:
            page_id (str): The ID or URL of the Notion page to process.
            
        Returns:
            Dict[str, Any]: A dictionary with status information and processed chunks.
        """
        if not self.api_key:
            return {
                "status": "error",
                "message": "Notion API key not configured",
                "help": "Please set your Notion API key in the settings page."
            }
        
        if not self.notion_manager:
            return {
                "status": "error",
                "message": "NotionManager not available",
                "help": "The NotionManager module is required for this functionality."
            }
        
        try:
            # Extract page ID from URL if necessary
            if "notion.so" in page_id:
                page_id = self.notion_manager.extract_page_id(page_id)
            
            # Load the page as a document
            logger.info(f"Loading Notion page: {page_id}")
            documents = self.notion_manager.load_page(page_id)
            
            if not documents:
                return {
                    "status": "error",
                    "message": f"No content found in page {page_id}",
                    "help": "Make sure the page has content and is shared with your integration."
                }
            
            # Process the documents
            logger.info(f"Processing {len(documents)} documents from Notion page")
            chunks = self.document_processor.process_documents(documents)
            
            return {
                "status": "success",
                "message": f"Successfully processed Notion page: {documents[0].metadata.get('title', page_id)}",
                "chunks": chunks,
                "chunks_added": len(chunks)
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
            elif "Access denied" in error_message or "403" in error_message:
                # Try to get the list of integrations to provide in the help text
                integrations_info = ""
                try:
                    integrations = self.notion_manager.list_integrations()
                    if integrations:
                        integrations_info = "\nYour available integrations:\n"
                        for i, integration in enumerate(integrations, 1):
                            integrations_info += f"{i}. {integration.get('name', 'Unnamed')}\n"
                except:
                    pass
                
                help_text = f"""
                ### How to fix this error:
                
                Your integration doesn't have access to this page. You need to:
                
                1. Go to the Notion page you want to access
                2. Click the ⋮ menu in the top right
                3. Select "Add connections"
                4. Find and select your integration{integrations_info}
                """
                return {
                    "status": "error",
                    "message": f"Access denied to page: {page_id}",
                    "help": help_text
                }
            else:
                return {
                    "status": "error",
                    "message": f"Error processing page {page_id}: {error_message}"
                }
    
    def process_pages(self, page_ids: List[str]) -> Dict[str, Any]:
        """
        Process multiple Notion pages and prepare them for ingestion.
        
        Args:
            page_ids (List[str]): List of page IDs or URLs to process.
            
        Returns:
            Dict[str, Any]: A dictionary with status information and processed chunks.
        """
        if not page_ids:
            return {
                "status": "error",
                "message": "No page IDs provided"
            }
        
        all_chunks = []
        errors = []
        successful_pages = 0
        
        for page_id in page_ids:
            result = self.process_page(page_id)
            
            if result["status"] == "success":
                all_chunks.extend(result.get("chunks", []))
                successful_pages += 1
            else:
                errors.append({
                    "page_id": page_id,
                    "error": result.get("message", "Unknown error")
                })
        
        if successful_pages > 0:
            return {
                "status": "success",
                "message": f"Successfully processed {successful_pages} out of {len(page_ids)} pages",
                "chunks": all_chunks,
                "chunks_added": len(all_chunks),
                "errors": errors if errors else None
            }
        else:
            return {
                "status": "error",
                "message": "Failed to process any pages",
                "errors": errors
            } 