import os
import logging
import json
import re
from typing import List, Dict, Any, Optional, Union
import requests
from datetime import datetime
from langchain_core.documents import Document
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NotionLoader:
    """
    Class for loading content from Notion pages and databases using the Notion API.
    """
    
    BASE_URL = "https://api.notion.com/v1"
    VERSION = "2022-06-28"
    
    def __init__(self, api_key: str):
        """
        Initialize the Notion loader.
        
        Args:
            api_key: Notion API key (Integration Token)
        """
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Notion-Version": self.VERSION,
            "Content-Type": "application/json"
        }
        logger.info("NotionLoader initialized")
        
        # Check connection and list integrations on init to help with debugging
        self.list_integrations()
    
    def list_integrations(self) -> Dict[str, Any]:
        """
        List all available Notion integrations for the API key.
        This helps verify the API key and check what pages the integration has access to.
        
        Returns:
            Dictionary with status and integrations information
        """
        try:
            # Check if API key exists
            if not self.api_key:
                logger.error("No Notion API key provided")
                return {
                    "status": "error",
                    "message": "No Notion API key provided. Please set the NOTION_API_KEY environment variable.",
                    "integrations": []
                }
            
            # Make a test request to the users endpoint to verify the API key
            user_url = f"{self.BASE_URL}/users/me"
            response = requests.get(user_url, headers=self.headers)
            
            # Check if the request was successful
            if response.status_code == 200:
                user_data = response.json()
                
                # Get the user's workspaces
                # The search endpoint can list all pages that have the integration
                search_url = f"{self.BASE_URL}/search"
                search_data = {
                    "filter": {
                        "value": "page",
                        "property": "object"
                    },
                    "page_size": 20  # Increased to show more pages
                }
                
                search_response = requests.post(
                    search_url, 
                    headers=self.headers, 
                    json=search_data
                )
                
                if search_response.status_code == 200:
                    search_results = search_response.json()
                    accessible_pages = search_results.get("results", [])
                    
                    # Format the results
                    integrations_info = {
                        "status": "success",
                        "user": {
                            "id": user_data.get("id"),
                            "name": user_data.get("name"),
                            "type": user_data.get("type")
                        },
                        "connection_status": "connected",
                        "accessible_pages": [
                            {
                                "id": page.get("id"),
                                "title": self._get_page_title(page),
                                "url": page.get("url"),
                                "last_edited": page.get("last_edited_time")
                            }
                            for page in accessible_pages
                        ],
                        "message": f"Successfully connected to Notion API. Found {len(accessible_pages)} accessible pages."
                    }
                    
                    # Log success
                    logger.info(f"Successfully connected to Notion API. Found {len(accessible_pages)} accessible pages.")
                    
                    return integrations_info
                else:
                    # Error with search request
                    error_message = f"Error searching Notion pages: {search_response.status_code} - {search_response.text}"
                    logger.error(error_message)
                    
                    return {
                        "status": "error",
                        "message": error_message,
                        "connection_status": "limited",
                        "user": {
                            "id": user_data.get("id"),
                            "name": user_data.get("name"),
                            "type": user_data.get("type")
                        },
                        "integrations": []
                    }
            else:
                # Error with API key
                error_message = f"Error connecting to Notion API: {response.status_code} - {response.text}"
                logger.error(error_message)
                
                # Check for specific error types
                error_data = response.json()
                error_code = error_data.get("code", "")
                
                help_message = "Please verify your Notion API key is correct."
                
                # Add more specific help messages based on error codes
                if error_code == "unauthorized":
                    help_message = (
                        "Your Notion API key appears to be invalid or expired. Please check:\n"
                        "1. You've copied the correct Integration Token from Notion\n"
                        "2. The token has not expired\n"
                        "3. You've set the NOTION_API_KEY environment variable correctly"
                    )
                
                return {
                    "status": "error",
                    "message": error_message,
                    "help": help_message,
                    "connection_status": "disconnected",
                    "integrations": []
                }
                
        except Exception as e:
            # General exception handling
            error_message = f"Unexpected error connecting to Notion API: {str(e)}"
            logger.error(error_message)
            
            return {
                "status": "error",
                "message": error_message,
                "help": "Please check your internet connection and Notion API key.",
                "connection_status": "error",
                "integrations": []
            }
    
    def _get_page_title(self, page: Dict[str, Any]) -> str:
        """
        Extract the title from a Notion page object.
        
        Args:
            page: Notion page object
            
        Returns:
            Page title string
        """
        title = "Untitled"
        
        try:
            properties = page.get("properties", {})
            if "title" in properties:
                title_content = properties["title"].get("title", [])
                if title_content:
                    title = "".join(item.get("plain_text", "") for item in title_content)
            return title
        except Exception as e:
            logger.warning(f"Error extracting page title: {str(e)}")
            return title
    
    def retrieve_page(self, page_id: str) -> Dict[str, Any]:
        """
        Retrieve a Notion page by ID.
        
        Args:
            page_id: Notion page ID
            
        Returns:
            Dictionary with page data
        """
        # Clean the page ID (remove any URL parts)
        clean_id = self._clean_page_id(page_id)
        
        try:
            # Get page metadata
            page_url = f"{self.BASE_URL}/pages/{clean_id}"
            response = requests.get(page_url, headers=self.headers)
            response.raise_for_status()
            page_data = response.json()
            
            # Get page blocks (content)
            blocks_url = f"{self.BASE_URL}/blocks/{clean_id}/children?page_size=100"
            blocks_response = requests.get(blocks_url, headers=self.headers)
            blocks_response.raise_for_status()
            blocks_data = blocks_response.json()
            
            # Recursively get all nested blocks
            all_blocks = blocks_data.get("results", [])
            all_blocks = self._get_all_nested_blocks(all_blocks)
            
            return {
                "page": page_data,
                "blocks": all_blocks
            }
            
        except requests.HTTPError as e:
            error_msg = f"HTTP error retrieving Notion page {page_id}: {str(e)}"
            logger.error(error_msg)
            
            # Enhanced error handling with specific instructions for common issues
            if e.response.status_code == 404:
                help_text = (
                    "The page could not be found. Please check:\n"
                    "1. The page ID is correct\n"
                    "2. The page exists\n"
                    "3. Your integration has access to the page\n\n"
                    "To share a page with your integration:\n"
                    "1. Open the page in Notion\n"
                    "2. Click 'Share' in the top right\n"
                    "3. Click 'Add people, groups, or emails'\n"
                    "4. In the search box, enter the name of your integration\n"
                    "5. Select your integration and click 'Invite'\n"
                )
                
                # List integrations to help debugging
                integrations = self.list_integrations()
                
                if integrations["status"] == "success":
                    help_text += "\n\nYour integration currently has access to these pages:\n"
                    for page in integrations.get("accessible_pages", []):
                        help_text += f"- {page.get('title', 'Untitled')} (ID: {page.get('id')})\n"
                    
                return {
                    "status": "error",
                    "message": error_msg,
                    "help": help_text
                }
            elif e.response.status_code == 401:
                return {
                    "status": "error",
                    "message": error_msg,
                    "help": "Your Notion API key appears to be invalid. Please check the key and try again."
                }
            else:
                return {
                    "status": "error",
                    "message": error_msg
                }
        except Exception as e:
            error_msg = f"Unexpected error retrieving Notion page {page_id}: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }
    
    def get_page(self, page_id: str) -> Document:
        """
        Get a Notion page as a Document object.
        
        Args:
            page_id: Notion page ID
            
        Returns:
            Document object
        """
        try:
            # Retrieve the page
            page_data = self.retrieve_page(page_id)
            
            # Check if there was an error
            if page_data.get("status") == "error":
                logger.error(f"Error retrieving page: {page_data.get('message')}")
                error_message = page_data.get("help", page_data.get("message", "Unknown error"))
                return Document(
                    page_content=f"Error: {error_message}",
                    metadata={"error": True, "page_id": page_id}
                )
            
            # Convert to Document
            return self.page_to_document(page_data)
            
        except Exception as e:
            error_msg = f"Error getting Notion page {page_id}: {str(e)}"
            logger.error(error_msg)
            return Document(
                page_content=f"Error: {error_msg}",
                metadata={"error": True, "page_id": page_id}
            )
    
    def _get_all_nested_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Recursively get all nested blocks.
        
        Args:
            blocks: List of blocks
            
        Returns:
            List of all blocks, including nested ones
        """
        all_blocks = []
        
        for block in blocks:
            all_blocks.append(block)
            
            # Check if block has children
            if block.get("has_children", False):
                try:
                    # Get block ID
                    block_id = block.get("id")
                    
                    # Get children
                    children_url = f"{self.BASE_URL}/blocks/{block_id}/children?page_size=100"
                    children_response = requests.get(children_url, headers=self.headers)
                    children_response.raise_for_status()
                    children_data = children_response.json()
                    
                    # Get children blocks
                    children_blocks = children_data.get("results", [])
                    
                    # Recursively get nested blocks
                    all_blocks.extend(self._get_all_nested_blocks(children_blocks))
                except Exception as e:
                    logger.warning(f"Error retrieving nested blocks: {str(e)}")
            
        return all_blocks
    
    def _clean_page_id(self, page_id: str) -> str:
        """
        Clean a page ID (remove any URL parts).
        
        Args:
            page_id: Page ID or URL
            
        Returns:
            Cleaned page ID
        """
        if not page_id:
            return ""
            
        # Try to extract page ID from URL
        url_match = re.search(r"([a-f0-9]{32}|[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})", page_id)
        if url_match:
            extracted_id = url_match.group(1)
            
            # If ID doesn't contain dashes, add them
            if "-" not in extracted_id:
                return f"{extracted_id[:8]}-{extracted_id[8:12]}-{extracted_id[12:16]}-{extracted_id[16:20]}-{extracted_id[20:]}"
            return extracted_id
        
        # Return original if no match found (assuming it's already a clean ID)
        return page_id
    
    def page_to_document(self, page_data: Dict[str, Any], include_raw_content: bool = False) -> Document:
        """
        Convert a Notion page to a Document object.
        
        Args:
            page_data: Notion page data
            include_raw_content: Whether to include raw content in metadata
            
        Returns:
            Document object
        """
        try:
            page = page_data.get("page", {})
            blocks = page_data.get("blocks", [])
            
            # Extract page metadata
            page_id = page.get("id", "")
            page_title = self._get_page_title(page)
            page_url = page.get("url", "")
            created_time = page.get("created_time", "")
            last_edited_time = page.get("last_edited_time", "")
            
            # Extract text content from blocks
            text_content = self._extract_text_from_blocks(blocks)
            
            # Create metadata
            metadata = {
                "source": "notion",
                "page_id": page_id,
                "title": page_title,
                "url": page_url,
                "created_time": created_time,
                "last_edited_time": last_edited_time
            }
            
            # Include raw content if requested
            if include_raw_content:
                metadata["raw_content"] = json.dumps(page_data)
            
            # Create Document
            return Document(
                page_content=text_content,
                metadata=metadata
            )
            
        except Exception as e:
            error_msg = f"Error converting Notion page to Document: {str(e)}"
            logger.error(error_msg)
            
            # Return empty Document with error information
            return Document(
                page_content="Error: Failed to process Notion page",
                metadata={
                    "source": "notion",
                    "error": str(e)
                }
            )
    
    def _extract_text_from_blocks(self, blocks: List[Dict[str, Any]]) -> str:
        """
        Extract text content from Notion blocks.
        
        Args:
            blocks: List of Notion blocks
            
        Returns:
            Extracted text
        """
        text_content = []
        
        for block in blocks:
            block_type = block.get("type")
            block_value = block.get(block_type, {})
            
            if block_type == "paragraph":
                text = self._get_rich_text(block_value.get("rich_text", []))
                if text:
                    text_content.append(text)
            
            elif block_type == "heading_1":
                text = self._get_rich_text(block_value.get("rich_text", []))
                if text:
                    text_content.append(f"# {text}")
            
            elif block_type == "heading_2":
                text = self._get_rich_text(block_value.get("rich_text", []))
                if text:
                    text_content.append(f"## {text}")
            
            elif block_type == "heading_3":
                text = self._get_rich_text(block_value.get("rich_text", []))
                if text:
                    text_content.append(f"### {text}")
            
            elif block_type == "bulleted_list_item":
                text = self._get_rich_text(block_value.get("rich_text", []))
                if text:
                    text_content.append(f"• {text}")
            
            elif block_type == "numbered_list_item":
                text = self._get_rich_text(block_value.get("rich_text", []))
                if text:
                    text_content.append(f"1. {text}")  # We don't track actual numbering
            
            elif block_type == "to_do":
                text = self._get_rich_text(block_value.get("rich_text", []))
                checked = block_value.get("checked", False)
                if text:
                    if checked:
                        text_content.append(f"[x] {text}")
                    else:
                        text_content.append(f"[ ] {text}")
            
            elif block_type == "toggle":
                text = self._get_rich_text(block_value.get("rich_text", []))
                if text:
                    text_content.append(f"▶ {text}")
            
            elif block_type == "code":
                text = self._get_rich_text(block_value.get("rich_text", []))
                language = block_value.get("language", "")
                if text:
                    text_content.append(f"```{language}\n{text}\n```")
            
            elif block_type == "quote":
                text = self._get_rich_text(block_value.get("rich_text", []))
                if text:
                    text_content.append(f"> {text}")
            
            elif block_type == "divider":
                text_content.append("---")
            
            elif block_type == "callout":
                text = self._get_rich_text(block_value.get("rich_text", []))
                emoji = block_value.get("icon", {}).get("emoji", "")
                if text:
                    if emoji:
                        text_content.append(f"{emoji} {text}")
                    else:
                        text_content.append(text)
            
            elif block_type == "table":
                # Tables are difficult to properly represent in plain text
                # Just indicate that there is a table
                text_content.append("[Table content]")
            
            elif block_type == "child_page":
                title = block_value.get("title", "Untitled")
                text_content.append(f"[Page: {title}]")
            
            elif block_type == "child_database":
                title = block_value.get("title", "Untitled")
                text_content.append(f"[Database: {title}]")
                
        return "\n\n".join(text_content)
    
    def _get_rich_text(self, rich_text: List[Dict[str, Any]]) -> str:
        """
        Extract plain text from rich text objects.
        
        Args:
            rich_text: List of rich text objects
            
        Returns:
            Plain text
        """
        return "".join(item.get("plain_text", "") for item in rich_text)
    
    def get_multiple_pages(self, page_ids: List[str]) -> List[Document]:
        """
        Get multiple Notion pages as Document objects.
        
        Args:
            page_ids: List of Notion page IDs
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for page_id in page_ids:
            try:
                document = self.get_page(page_id)
                documents.append(document)
                logger.info(f"Successfully loaded Notion page: {page_id}")
            except Exception as e:
                error_msg = f"Error loading Notion page {page_id}: {str(e)}"
                logger.error(error_msg)
                # Add an error document
                documents.append(Document(
                    page_content=f"Error loading page {page_id}: {str(e)}",
                    metadata={"error": True, "page_id": page_id}
                ))
        
        return documents
    
    def search_database(
        self, 
        database_id: str, 
        query: Optional[str] = None,
        filter_properties: Optional[Dict[str, Any]] = None,
        sort_property: Optional[str] = None,
        sort_direction: str = "descending",
        page_size: int = 100
    ) -> Dict[str, Any]:
        """
        Search a Notion database.
        
        Args:
            database_id: Notion database ID
            query: Optional search query
            filter_properties: Optional filter properties
            sort_property: Optional sort property
            sort_direction: Sort direction ("ascending" or "descending")
            page_size: Number of results to return
            
        Returns:
            Dictionary with search results
        """
        # Clean the database ID
        clean_id = self._clean_page_id(database_id)
        
        try:
            # Prepare request data
            request_data = {
                "page_size": page_size
            }
            
            # Add query if provided
            if query:
                request_data["query"] = query
            
            # Add filter if provided
            if filter_properties:
                request_data["filter"] = filter_properties
            
            # Add sort if provided
            if sort_property:
                request_data["sorts"] = [{
                    "property": sort_property,
                    "direction": sort_direction
                }]
            
            # Make request
            response = requests.post(
                f"{self.BASE_URL}/databases/{clean_id}/query",
                headers=self.headers,
                json=request_data
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.HTTPError as e:
            error_msg = f"HTTP error searching Notion database {database_id}: {str(e)}"
            logger.error(error_msg)
            
            # Enhanced error handling
            if e.response.status_code == 404:
                help_text = (
                    "The database could not be found. Please check:\n"
                    "1. The database ID is correct\n"
                    "2. The database exists\n"
                    "3. Your integration has access to the database"
                )
                
                # List integrations to help debugging
                integrations = self.list_integrations()
                
                if integrations["status"] == "success":
                    help_text += "\n\nYour integration currently has access to these pages/databases:\n"
                    for page in integrations.get("accessible_pages", []):
                        help_text += f"- {page.get('title', 'Untitled')} (ID: {page.get('id')})\n"
                
                return {
                    "status": "error",
                    "message": error_msg,
                    "help": help_text
                }
            else:
                return {
                    "status": "error",
                    "message": error_msg
                }
        except Exception as e:
            error_msg = f"Unexpected error searching Notion database {database_id}: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }
    
    def database_to_documents(
        self, 
        database_results: Dict[str, Any],
        include_database_info: bool = True
    ) -> List[Document]:
        """
        Convert database search results to Document objects.
        
        Args:
            database_results: Database search results
            include_database_info: Whether to include database info in metadata
            
        Returns:
            List of Document objects
        """
        documents = []
        
        # Check if there was an error
        if database_results.get("status") == "error":
            error_msg = database_results.get("message", "Unknown error")
            logger.error(f"Error converting database to documents: {error_msg}")
            return [Document(
                page_content=f"Error: {error_msg}",
                metadata={"error": True}
            )]
        
        # Process results
        results = database_results.get("results", [])
        
        for result in results:
            try:
                # Get page content
                page_id = result.get("id")
                page_data = self.retrieve_page(page_id)
                
                # Check if there was an error
                if page_data.get("status") == "error":
                    logger.warning(f"Error retrieving page {page_id}: {page_data.get('message')}")
                    continue
                
                # Convert to Document
                document = self.page_to_document(page_data, include_raw_content=False)
                
                # Add database info if requested
                if include_database_info:
                    document.metadata["database_id"] = database_results.get("database_id", "")
                
                documents.append(document)
                
            except Exception as e:
                logger.warning(f"Error processing database result: {str(e)}")
        
        return documents
    
    def load_documents_from_notion_urls(self, urls: List[str]) -> List[Document]:
        """
        Load documents from a list of Notion URLs.
        
        Args:
            urls: List of Notion URLs
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for url in urls:
            try:
                # Clean the URL and extract page ID
                page_id = self._clean_page_id(url)
                
                # Get the page document
                document = self.get_page(page_id)
                
                documents.append(document)
                logger.info(f"Successfully loaded Notion page from URL: {url}")
            except Exception as e:
                error_msg = f"Error loading Notion page from URL {url}: {str(e)}"
                logger.error(error_msg)
                # Add an error document
                documents.append(Document(
                    page_content=f"Error loading page from URL {url}: {str(e)}",
                    metadata={"error": True, "url": url}
                ))
        
        return documents 