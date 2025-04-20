from notion_client import Client
import os
import re
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import time
import logging
import requests
from langchain_community.document_loaders import NotionDBLoader
from custom_notion_loader import CustomNotionLoader
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)

class NotionManager:
    """
    Manager for Notion API operations
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the NotionManager with the Notion API key.
        
        Args:
            api_key: The Notion API key. If None, tries to get it from environment variables.
        """
        self.api_key = api_key or os.getenv("NOTION_API_KEY", "")
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        self.logger = logging.getLogger(__name__)
        self.notion = None
        
        if self.api_key:
            self.notion = Client(auth=self.api_key)
            print("✅ Notion client initialized")
        else:
            print("⚠️ Notion API key not provided")
    
    def check_connection(self) -> bool:
        """
        Check if the connection to the Notion API is working.
        
        Returns:
            bool: True if the connection is successful, False otherwise.
        """
        if not self.api_key:
            self.logger.error("Notion API key not provided")
            return False
            
        try:
            response = requests.get(
                f"{self.base_url}/users/me",
                headers=self.headers
            )
            if response.status_code == 200:
                self.logger.info("Successfully connected to Notion API")
                return True
            else:
                self.logger.error(f"Failed to connect to Notion API: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            self.logger.error(f"Error connecting to Notion API: {str(e)}")
            return False
            
    def list_integrations(self) -> Dict[str, Any]:
        """List all integrations connected to the API key"""
        try:
            # Test the connection
            response = requests.get(
                f"{self.base_url}/users/me",
                headers=self.headers
            )
            response.raise_for_status()
            
            # Get accessible pages
            search_response = requests.post(
                f"{self.base_url}/search",
                headers=self.headers,
                json={"page_size": 10}
            )
            search_response.raise_for_status()
            search_results = search_response.json().get("results", [])
            
            return {
                "status": "success",
                "message": "Successfully connected to Notion API",
                "user": response.json(),
                "integrations": [
                    {
                        "id": item.get("id", ""),
                        "name": self._get_page_title(item),
                        "type": item.get("object", "unknown"),
                        "connected": True
                    }
                    for item in search_results
                ]
            }
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Failed to connect to Notion API: {str(e)}"
            }
    
    def _get_page_title(self, page_data: Dict[str, Any]) -> str:
        """Extract page title from page data"""
        if page_data.get("object") != "page":
            return "Non-page item"
            
        properties = page_data.get("properties", {})
        for prop in properties.values():
            if prop.get("type") == "title":
                title_items = prop.get("title", [])
                if title_items:
                    return " ".join([t.get("plain_text", "") for t in title_items])
        
        return "Untitled"
    
    def extract_page_id(self, url_or_id: str) -> str:
        """Extract page ID from URL or return the ID as is"""
        if "notion.so" in url_or_id:
            # Extract ID from URL
            parts = url_or_id.split("/")
            for part in parts:
                if "-" in part and len(part) > 30:
                    return part.split("-")[-1]
            return url_or_id
        return url_or_id
    
    def get_page_content(self, page_id: str) -> Dict:
        """
        Get the content of a Notion page.
        
        Args:
            page_id: The ID of the Notion page.
            
        Returns:
            Dict: A dictionary containing the page content.
        """
        # Clean the page ID (remove any URL parts)
        page_id = self._clean_page_id(page_id)
        
        try:
            response = requests.get(
                f"{self.base_url}/blocks/{page_id}/children?page_size=100",
                headers=self.headers
            )
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to get page content: {response.status_code} - {response.text}")
                if response.status_code == 404:
                    self.logger.error(f"Page not found. Please check if the page exists and the integration has access to it.")
                return {}
        except Exception as e:
            self.logger.error(f"Error getting page content: {str(e)}")
            return {}
    
    def get_database_content(self, database_id: str) -> Dict:
        """
        Get the content of a Notion database.
        
        Args:
            database_id: The ID of the Notion database.
            
        Returns:
            Dict: A dictionary containing the database content.
        """
        # Clean the database ID (remove any URL parts)
        database_id = self._clean_page_id(database_id)
        
        try:
            response = requests.post(
                f"{self.base_url}/databases/{database_id}/query",
                headers=self.headers,
                json={"page_size": 100}
            )
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to get database content: {response.status_code} - {response.text}")
                if response.status_code == 404:
                    self.logger.error(f"Database not found. Please check if the database exists and the integration has access to it.")
                return {}
        except Exception as e:
            self.logger.error(f"Error getting database content: {str(e)}")
            return {}
    
    def load_page_with_langchain(self, page_id: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
        """
        Load a Notion page with LangChain and split it into chunks.
        
        Args:
            page_id: The ID of the Notion page.
            chunk_size: The size of each text chunk.
            chunk_overlap: The overlap between chunks.
            
        Returns:
            List: A list of document chunks.
        """
        # Clean the page ID (remove any URL parts)
        page_id = self._clean_page_id(page_id)
        
        try:
            # Initialize the CustomNotionLoader
            loader = CustomNotionLoader(
                api_key=self.api_key
            )
            
            # Load documents
            documents = loader.load_page(page_id)
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            chunks = text_splitter.split_documents(documents)
            
            # Add source information to metadata
            for chunk in chunks:
                if not chunk.metadata.get("source"):
                    chunk.metadata["source"] = f"notion-page-{page_id}"
                    
            return chunks
        except Exception as e:
            self.logger.error(f"Error loading page with LangChain: {str(e)}")
            return []
    
    def load_database_with_langchain(self, database_id: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
        """
        Load a Notion database with LangChain and split it into chunks.
        
        Args:
            database_id: The ID of the Notion database.
            chunk_size: The size of each text chunk.
            chunk_overlap: The overlap between chunks.
            
        Returns:
            List: A list of document chunks.
        """
        # Clean the database ID (remove any URL parts)
        database_id = self._clean_page_id(database_id)
        
        try:
            # Initialize the NotionDBLoader
            loader = NotionDBLoader(
                integration_token=self.api_key,
                database_id=database_id,
                request_timeout=60
            )
            
            # Load documents
            documents = loader.load()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            chunks = text_splitter.split_documents(documents)
            
            # Add source information to metadata
            for chunk in chunks:
                if not chunk.metadata.get("source"):
                    chunk.metadata["source"] = f"notion-db-{database_id}"
                    
            return chunks
        except Exception as e:
            self.logger.error(f"Error loading database with LangChain: {str(e)}")
            return []
    
    def _clean_page_id(self, page_id: str) -> str:
        """
        Clean the page ID by removing any URL parts.
        
        Args:
            page_id: The page ID, which might be a URL or contain hyphens.
            
        Returns:
            str: The cleaned page ID.
        """
        # Extract the ID from a URL if necessary
        if "notion.so/" in page_id:
            page_id = page_id.split("notion.so/")[1]
            
        # Remove any query parameters
        if "?" in page_id:
            page_id = page_id.split("?")[0]
            
        # Split by hyphens and look for the 32-character hex string
        parts = page_id.split("-")
        for part in parts:
            # Check if this part is a 32-character hex string
            cleaned_part = "".join(c for c in part if c in "0123456789abcdef")
            if len(cleaned_part) == 32:
                return cleaned_part
            
        # If we haven't found a 32-char hex string, try to find it in the combined string
        combined = "".join(parts)
        hex_chars = "".join(c for c in combined if c in "0123456789abcdef")
        if len(hex_chars) >= 32:
            return hex_chars[:32]
            
        # If all else fails, return the cleaned original
        return "".join(c for c in page_id if c.isalnum())
    
    def load_database(self, database_id=None):
        """Load all pages from a Notion database"""
        if not self.notion:
            return {
                "success": False,
                "message": "Notion API key not configured",
                "pages": []
            }
        
        try:
            if not database_id:
                database_id = os.getenv("NOTION_DATABASE_ID")
                if not database_id:
                    # Try to get the first database available
                    databases = self.notion.search(filter={"property": "object", "value": "database"})
                    if databases.get("results"):
                        database_id = databases["results"][0]["id"]
                    else:
                        return {
                            "success": False,
                            "message": "No database ID provided and no databases found",
                            "pages": []
                        }
            
            # Query all pages in the database
            pages = self.notion.databases.query(database_id=database_id)
            
            loaded_pages = []
            
            # Process each page
            for page in pages.get("results", []):
                page_id = page["id"]
                
                # Get page properties
                properties = page.get("properties", {})
                page_title = ""
                
                # Extract title from properties
                for prop in properties.values():
                    if prop.get("type") == "title":
                        title_texts = prop.get("title", [])
                        page_title = "".join([text["plain_text"] for text in title_texts]) if title_texts else "Untitled"
                        break
                
                # Get page content
                page_result = self.get_page_content(page_id)
                
                if page_result["success"]:
                    # Add additional properties to metadata
                    metadata = page_result["metadata"]
                    
                    for prop_name, prop_value in properties.items():
                        if prop_name.lower() != "title":
                            prop_type = prop_value.get("type", "")
                            if prop_type == "rich_text":
                                metadata[prop_name] = "".join([text["plain_text"] for text in prop_value.get("rich_text", [])])
                            elif prop_type == "select":
                                select_data = prop_value.get("select", {})
                                metadata[prop_name] = select_data.get("name") if select_data else ""
                            elif prop_type == "multi_select":
                                multi_select = prop_value.get("multi_select", [])
                                metadata[prop_name] = ", ".join([item.get("name", "") for item in multi_select])
                            elif prop_type == "date":
                                date_data = prop_value.get("date", {})
                                metadata[prop_name] = date_data.get("start") if date_data else ""
                    
                    loaded_pages.append({
                        "id": page_id,
                        "title": page_title,
                        "content": page_result["content"],
                        "metadata": metadata
                    })
                else:
                    # If we couldn't get the content, still add the page with basic info
                    loaded_pages.append({
                        "id": page_id,
                        "title": page_title,
                        "content": "",
                        "metadata": {"error": page_result["message"]}
                    })
            
            return {
                "success": True,
                "message": f"Successfully loaded {len(loaded_pages)} pages from database",
                "database_id": database_id,
                "pages": loaded_pages
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error loading from Notion database: {str(e)}",
                "pages": []
            }
    
    def prepare_documents_for_indexing(self, pages, chunk_size=500, chunk_overlap=50):
        """Prepare documents for indexing by splitting into chunks with metadata"""
        if not pages:
            return []
        
        # Create text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        all_docs = []
        
        # Process each page
        for page in pages:
            if not page.get("content"):
                continue
                
            # Create document with metadata
            doc = Document(
                page_content=page["content"],
                metadata=page.get("metadata", {})
            )
            
            # Split document
            chunks = splitter.split_documents([doc])
            
            # Update chunk metadata
            for i, chunk in enumerate(chunks):
                chunk_metadata = chunk.metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk.metadata = chunk_metadata
                all_docs.append(chunk)
        
        return all_docs
    
    def preprocess_metadata(self, metadata):
        """Process metadata to handle complex types for storage"""
        processed = {}
        for key, value in metadata.items():
            if value is None:
                processed[key] = ''
            elif isinstance(value, list):
                processed[key] = ', '.join(str(item) for item in value)
            elif isinstance(value, dict):
                processed[key] = str(value)
            else:
                processed[key] = value
        return processed 