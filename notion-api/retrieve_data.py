import requests
import json
import unicodedata
import os
import yaml
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv


def format_page_id(page_id):
    """
    Format page ID to include hyphens in the correct positions:
    xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    """
    # Remove any existing hyphens
    clean_id = page_id.replace("-", "")
    
    # Insert hyphens at the correct positions
    if len(clean_id) == 32:
        return f"{clean_id[:8]}-{clean_id[8:12]}-{clean_id[12:16]}-{clean_id[16:20]}-{clean_id[20:]}"
    return page_id


def get_page_properties(page_id, headers):
    """Get page properties including Status and Last Edited Time"""
    url = f"https://api.notion.com/v1/pages/{page_id}"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        page_data = response.json()
        properties = page_data.get("properties", {})
        
        # Get page title
        title = "Untitled"
        for prop in properties.values():
            if prop.get("type") == "title":
                title_texts = prop.get("title", [])
                title = "".join([text["plain_text"] for text in title_texts])
        
        # Get Status property (if exists)
        status = None
        for prop_name, prop in properties.items():
            if prop_name.lower() == "status" and prop.get("type") == "select":
                status_data = prop.get("select", {})
                if status_data:
                    status = status_data.get("name")
        
        # Get Last Edited Time
        last_edited = page_data.get("last_edited_time")
        
        # Get URL
        url = page_data.get("url")
        
        return {
            "title": title,
            "status": status,
            "last_edited_time": last_edited,
            "url": url,
            "page_id": page_id
        }
    else:
        print(f"Error getting page properties: {response.status_code}, {response.text}")
        return {"title": "Untitled", "status": None, "last_edited_time": None, "url": None, "page_id": page_id}


def get_all_blocks(page_id, headers, notion_api_url):
    """
    Get all blcoks from Notion Page
    """
    all_blocks = []
    url = f"{notion_api_url}/{page_id}/children"
    print(f"Fetching blocks from URL: {url}")
    print(f"Using headers: {headers}")
    while url:
        response = requests.get(url, headers=headers)
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")
        if response.status_code == 200:
            data = response.json()
            all_blocks.extend(data.get("results", []))
            
            if data.get("has_more"):
                url = f"{notion_api_url}/{page_id}/children?start_cursor={data.get('next_cursor')}"
            else:
                url = None
        else:
            print(f"Error: {response.status_code}, {response.text}")
            break
    return all_blocks

def normalize_text_data(content_list):
    """
    Normalize text data
    """
    normalized_content_list = [unicodedata.normalize('NFKC', text).strip().lower() for text in content_list]   
    return normalized_content_list 

def get_page_title(page_id, headers):
    """
    Get page title from Notion Page
    """
    url = f"https://api.notion.com/v1/pages/{page_id}"
    print(f"Fetching page title from URL: {url}")
    print(f"Using headers: {headers}")
    response = requests.get(url, headers=headers)
    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.text}")
    if response.status_code == 200:
        page_data = response.json()
        properties = page_data.get("properties", {})
        for prop in properties.values():
            if prop.get("type") == "title":
                title_texts = prop.get("title", [])
                return "".join([text["plain_text"] for text in title_texts])
        return "Untitled"
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def extract_text_from_blocks(blocks, headers, notion_api_url):
    """
    Extract text from blocks and return contents
    """
    content_list = []
    print(f"Processing {len(blocks)} blocks")
    for block in blocks:
        block_type = block.get("type")
        print(f"Processing block type: {block_type}")
        
        # text block
        if block_type in ["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item"]:
            rich_texts = block[block_type].get("rich_text", [])
            content_list.extend([text["text"]["content"] for text in rich_texts if "text" in text])
        
        # table blcok
        elif block_type == "table":
            table_id = block["id"]
            child_blocks = get_all_blocks(table_id, headers, notion_api_url)
            for row in child_blocks:
                if row.get("type") == "table_row":
                    row_cells = row["table_row"]["cells"]
                    row_text = ["".join([cell["text"]["content"] for cell in cell_texts if "text" in cell]) for cell_texts in row_cells]
                    content_list.append("\t".join(row_text)) 
        
        # children block 
        if block.get("has_children"):
            child_id = block["id"]
            child_blocks = get_all_blocks(child_id, headers, notion_api_url)
            content_list.extend(extract_text_from_blocks(child_blocks, headers, notion_api_url))
    
    return content_list


def extract_blocks_with_hierarchy(blocks, headers, notion_api_url, page_metadata, current_h1=None, current_h2=None, current_h3=None):
    """
    Extract blocks with heading hierarchy and metadata
    Returns list of tuples (block_content, metadata)
    """
    extracted_blocks = []
    
    for block in blocks:
        block_id = block.get("id")
        block_type = block.get("type")
        
        # Track heading hierarchy
        if block_type == "heading_1":
            current_h1 = "".join([text["plain_text"] for text in block["heading_1"].get("rich_text", [])])
            current_h2 = None
            current_h3 = None
        elif block_type == "heading_2":
            current_h2 = "".join([text["plain_text"] for text in block["heading_2"].get("rich_text", [])])
            current_h3 = None
        elif block_type == "heading_3":
            current_h3 = "".join([text["plain_text"] for text in block["heading_3"].get("rich_text", [])])
        
        # Extract text content based on block type
        content = None
        
        # Text blocks
        if block_type in ["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item"]:
            rich_texts = block[block_type].get("rich_text", [])
            if rich_texts:
                content = "".join([text["text"]["content"] for text in rich_texts if "text" in text])
        
        # Table blocks
        elif block_type == "table":
            table_id = block["id"]
            child_blocks = get_all_blocks(table_id, headers, notion_api_url)
            table_content = []
            
            for row in child_blocks:
                if row.get("type") == "table_row":
                    row_cells = row["table_row"]["cells"]
                    row_text = ["".join([cell["text"]["content"] for cell in cell_texts if "text" in cell]) for cell_texts in row_cells]
                    table_content.append("\t".join(row_text))
            
            if table_content:
                content = "\n".join(table_content)
        
        # Create metadata for this block
        if content:
            block_metadata = {
                "page_title": page_metadata["title"],
                "h1": current_h1,
                "h2": current_h2,
                "h3": current_h3,
                "last_edited": page_metadata["last_edited_time"],
                "url": page_metadata["url"],
                "page_id": page_metadata["page_id"],
                "block_id": block_id,
                "uuid": str(uuid.uuid4())  # Deterministic UUID
            }
            
            # Add to extracted blocks
            extracted_blocks.append((content, block_metadata))
        
        # Process child blocks and maintain hierarchy
        if block.get("has_children"):
            child_id = block["id"]
            child_blocks = get_all_blocks(child_id, headers, notion_api_url)
            child_extracted = extract_blocks_with_hierarchy(
                child_blocks, 
                headers, 
                notion_api_url, 
                page_metadata,
                current_h1,
                current_h2,
                current_h3
            )
            extracted_blocks.extend(child_extracted)
    
    return extracted_blocks


def normalize_text(text):
    """Normalize text data"""
    return unicodedata.normalize('NFKC', text).strip()


def fetch_all_pages(headers, notion_api_url):
    """Fetch all pages from Notion workspace"""
    url = "https://api.notion.com/v1/search"
    payload = {
        "filter": {
            "property": "object",
            "value": "page"
        }
    }
    
    all_pages = []
    has_more = True
    start_cursor = None
    
    while has_more:
        if start_cursor:
            payload["start_cursor"] = start_cursor
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            all_pages.extend(results)
            
            has_more = data.get("has_more", False)
            start_cursor = data.get("next_cursor")
        else:
            print(f"Error fetching pages: {response.status_code}, {response.text}")
            break
    
    return all_pages


def main():
    # Force reload of environment variables
    os.environ.clear()
    load_dotenv(override=True)

    # Get environment variables
    NOTION_API_KEY = os.getenv('NOTION_API_KEY')
    NOTION_API_URL = os.getenv('NOTION_API_URL', 'https://api.notion.com/v1/blocks')
    NOTION_VERSION = os.getenv('NOTION_VERSION', '2022-06-28')
    PAGE_IDS = os.getenv('NOTION_PAGE_IDS')
    
    print(f"Using Notion API Key: {NOTION_API_KEY[:10]}..." if NOTION_API_KEY else "No API key found")
    
    # Header setting
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json"
    }

    # Load config
    with open('config.yaml') as file:
        config = yaml.safe_load(file.read())

    try:
        # Determine which pages to process
        pages_to_process = []
        
        if PAGE_IDS:
            # Format each page ID with hyphens
            page_ids_list = [format_page_id(page_id.strip()) for page_id in PAGE_IDS.split(",") if page_id.strip()]
            print(f"Processing specific page IDs: {page_ids_list}")
            
            # Get metadata for specified pages
            for page_id in page_ids_list:
                page_metadata = get_page_properties(page_id, headers)
                pages_to_process.append(page_metadata)
        else:
            # Fetch all pages from workspace
            print("No specific pages specified. Fetching all available pages...")
            all_notion_pages = fetch_all_pages(headers, NOTION_API_URL)
            
            for page in all_notion_pages:
                page_id = page.get("id")
                page_metadata = get_page_properties(page_id, headers)
                pages_to_process.append(page_metadata)
        
        # Filter out private pages
        filtered_pages = []
        for page in pages_to_process:
            if page.get("status") == "Private":
                print(f"Skipping private page: {page.get('title')}")
                continue
            filtered_pages.append(page)
        
        print(f"Processing {len(filtered_pages)} non-private pages")
        
        # Process each page
        all_blocks_with_metadata = []
        
        for page in filtered_pages:
            page_id = page.get("page_id")
            print(f"\nProcessing page: {page.get('title')} (ID: {page_id})")
            
            # Get all blocks
            blocks = get_all_blocks(page_id, headers, NOTION_API_URL)
            print(f"Retrieved {len(blocks)} blocks")
            
            # Extract blocks with hierarchy
            extracted_blocks = extract_blocks_with_hierarchy(blocks, headers, NOTION_API_URL, page)
            print(f"Extracted {len(extracted_blocks)} content blocks with metadata")
            
            all_blocks_with_metadata.extend(extracted_blocks)
        
        # Create structured data for saving
        structured_data = []
        for content, metadata in all_blocks_with_metadata:
            structured_data.append({
                "content": normalize_text(content),
                "metadata": metadata
            })
        
        # Save in json
        with open(config["notion"]["content_file"], "w", encoding="utf-8") as file:
            json.dump(structured_data, file, ensure_ascii=False, indent=4)
        
        print(f"Content saved to {config['notion']['content_file']}")

    except Exception as e:
        print(f"Error: {e}")

# run
if __name__ == "__main__":
    main()