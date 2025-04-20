import os
import logging
import requests
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class CustomNotionLoader:
    """Custom implementation of Notion loader to replace the missing LangChain loader"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("NOTION_API_KEY", "")
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
    
    def extract_page_id(self, url_or_id: str) -> str:
        """Extract page ID from URL or return the ID as is"""
        # If it's already a valid page ID (32 chars with or without hyphens), return it
        cleaned_id = url_or_id.replace("-", "")
        if len(cleaned_id) == 32:
            return url_or_id
            
        # If it's a URL, try to extract the ID
        if "notion.so" in url_or_id:
            parts = url_or_id.split("/")
            for part in parts:
                # Look for the part that contains the page ID
                cleaned_part = part.replace("-", "")
                if len(cleaned_part) == 32:
                    return part  # Return the original part with hyphens
            
        # If no valid ID found, return as is
        return url_or_id
    
    def load_page(self, page_id: str) -> List[Document]:
        """Load a Notion page and convert to Document format"""
        try:
            # Get page info
            page_url = f"{self.base_url}/pages/{page_id}"
            response = requests.get(page_url, headers=self.headers)
            response.raise_for_status()
            page_data = response.json()
            
            # Get page content (blocks)
            blocks_url = f"{self.base_url}/blocks/{page_id}/children?page_size=100"
            response = requests.get(blocks_url, headers=self.headers)
            response.raise_for_status()
            blocks = response.json().get("results", [])
            
            # Extract text from blocks
            text_content = self._process_blocks(blocks)
            
            # Get page title
            title = self._get_page_title(page_data)
            
            # Create metadata
            metadata = {
                "source": "notion",
                "page_id": page_id,
                "title": title,
                "url": f"https://notion.so/{page_id}"
            }
            
            return [Document(page_content=text_content, metadata=metadata)]
            
        except Exception as e:
            logger.error(f"Error loading Notion page {page_id}: {str(e)}")
            return []
    
    def _process_blocks(self, blocks: List[Dict]) -> str:
        """Process blocks into plain text"""
        text_parts = []
        
        for block in blocks:
            block_type = block.get("type")
            if not block_type:
                continue
                
            # Process different block types
            if block_type == "paragraph":
                text = self._get_rich_text_content(block.get("paragraph", {}).get("rich_text", []))
                if text:
                    text_parts.append(text)
            
            elif block_type == "heading_1":
                text = self._get_rich_text_content(block.get("heading_1", {}).get("rich_text", []))
                if text:
                    text_parts.append(f"# {text}")
            
            elif block_type == "heading_2":
                text = self._get_rich_text_content(block.get("heading_2", {}).get("rich_text", []))
                if text:
                    text_parts.append(f"## {text}")
            
            elif block_type == "heading_3":
                text = self._get_rich_text_content(block.get("heading_3", {}).get("rich_text", []))
                if text:
                    text_parts.append(f"### {text}")
                    
            # You can add more block types as needed
                
        return "\n\n".join(text_parts)
    
    def _get_rich_text_content(self, rich_text: List[Dict]) -> str:
        """Extract text from rich text objects"""
        return "".join([rt.get("plain_text", "") for rt in rich_text])
    
    def _get_page_title(self, page_data: Dict) -> str:
        """Extract page title from page data"""
        properties = page_data.get("properties", {})
        
        for prop in properties.values():
            if prop.get("type") == "title":
                title_items = prop.get("title", [])
                if title_items:
                    return "".join([t.get("plain_text", "") for t in title_items])
        
        return "Untitled"
    
    def list_integrations(self) -> Dict[str, Any]:
        """List all available integrations for API key"""
        try:
            # Use search to list all pages accessible to the integration
            search_url = f"{self.base_url}/search"
            response = requests.post(search_url, headers=self.headers, json={})
            response.raise_for_status()
            
            results = response.json().get("results", [])
            
            # Extract basic info about pages
            integrations = []
            for result in results:
                obj_type = result.get("object")
                obj_id = result.get("id")
                
                if obj_type == "page":
                    # Get page title
                    properties = result.get("properties", {})
                    title = "Untitled"
                    for prop in properties.values():
                        if prop.get("type") == "title":
                            title_items = prop.get("title", [])
                            if title_items:
                                title = " ".join([t.get("plain_text", "") for t in title_items])
                                break
                    
                    integrations.append({
                        "id": obj_id,
                        "name": title,
                        "type": "page",
                        "connected": True
                    })
            
            return {
                "status": "success",
                "message": f"Found {len(integrations)} pages accessible to this integration",
                "integrations": integrations
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error listing integrations: {str(e)}"
            }