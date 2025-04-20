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
from langchain_community.document_loaders.notion import NotionDBLoader as NotionPageLoader

class NotionManager:
    def __init__(self, api_key: str = None):
        """
        Initialize the NotionManager with the Notion API key.
        
        Args:
            api_key: The Notion API key. If None, tries to get it from environment variables.
        """
        self.api_key = api_key or os.environ.get("NOTION_API_KEY", "")
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
            
    def list_integrations(self) -> List[Dict]:
        """
        List all Notion integrations connected to the API key.
        
        Returns:
            List[Dict]: A list of dictionaries containing integration information.
        """
        try:
            response = requests.get(
                f"{self.base_url}/users",
                headers=self.headers
            )
            if response.status_code == 200:
                return response.json().get("results", [])
            else:
                self.logger.error(f"Failed to list integrations: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            self.logger.error(f"Error listing integrations: {str(e)}")
            return []
    
    def extract_page_id(self, url: str) -> Optional[str]:
        """Extract page ID from Notion URL"""
        if not url:
            return None
            
        # Remove any text after the URL
        url = url.split()[0] if ' ' in url else url
        
        # Try different patterns for Notion URLs
        # Pattern 1: Standard format with dashes
        match = re.search(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', url)
        if match:
            return match.group(1)
        
        # Pattern 2: Format without dashes
        match = re.search(r'([a-f0-9]{32})', url)
        if match:
            raw_id = match.group(1)
            # Format it properly with dashes
            return f"{raw_id[:8]}-{raw_id[8:12]}-{raw_id[12:16]}-{raw_id[16:20]}-{raw_id[20:]}"
        
        # Pattern 3: Format with page name and ID
        match = re.search(r'notion\.so/(?:[^/]+/)*((?:[^-]+-)([a-f0-9]{32}))', url)
        if match:
            raw_id = match.group(2)
            # Format it properly with dashes
            return f"{raw_id[:8]}-{raw_id[8:12]}-{raw_id[12:16]}-{raw_id[16:20]}-{raw_id[20:]}"
            
        return None
    
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
            # Initialize the NotionPageLoader
            loader = NotionPageLoader(
                integration_token=self.api_key,
                page_id=page_id
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
            
        # Remove any page name at the end
        if "-" in page_id:
            # Check if it's the format with a dash followed by a name at the end
            if len(page_id.split("-")) > 1 and len(page_id.split("-")[-1]) > 20:
                # It's likely just a UUID with hyphens
                pass
            else:
                # It might be a page title at the end
                page_id = page_id.split("-")[0]
                
        # Remove any query parameters
        if "?" in page_id:
            page_id = page_id.split("?")[0]
            
        # Remove any hyphens or other non-alphanumeric characters
        page_id = "".join(c for c in page_id if c.isalnum())
            
        return page_id
    
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