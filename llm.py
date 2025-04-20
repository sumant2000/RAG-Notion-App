import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings  # Will be deprecated in future
from langchain_community.vectorstores import Chroma  # Will be deprecated in future
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema import format_document
from operator import itemgetter
import yaml
import json
from notion_client import Client
import re

# Comment to install newer packages when available:
# pip install langchain-huggingface langchain-chroma
# Then update imports to:
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma

class CustomHybridRetriever:
    """Custom Hybrid retriever with MMR reranking and recency tiebreaker"""
    
    def __init__(self, vectorstore, k=8, fetch_k=20, lambda_mult=0.5, score_threshold=0.02):
        self.vectorstore = vectorstore
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.score_threshold = score_threshold
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        # Step 1: Similarity search with MMR
        docs = self.vectorstore.max_marginal_relevance_search(
            query, k=self.k, fetch_k=self.fetch_k, lambda_mult=self.lambda_mult
        )
        
        # Step 2: Check for score ties and apply recency tiebreaker
        if len(docs) > 1:
            # Process for tied scores based on last_edited
            docs_by_score = {}
            for doc in docs:
                score = self.vectorstore.similarity_search_with_score(
                    doc.page_content, k=1
                )[0][1]
                
                # Group by score (with threshold)
                score_key = round(score / self.score_threshold) * self.score_threshold
                if score_key not in docs_by_score:
                    docs_by_score[score_key] = []
                docs_by_score[score_key].append((doc, score))
            
            # Sort tied groups by recency
            sorted_docs = []
            for score_key in sorted(docs_by_score.keys()):
                tied_docs = docs_by_score[score_key]
                if len(tied_docs) > 1:
                    # Sort by last_edited time (if available)
                    sorted_tied_docs = sorted(
                        tied_docs,
                        key=lambda x: x[0].metadata.get("last_edited", ""),
                        reverse=True  # Most recent first
                    )
                    sorted_docs.extend([doc for doc, _ in sorted_tied_docs])
                else:
                    sorted_docs.append(tied_docs[0][0])
            
            return sorted_docs[:self.k]
        
        return docs
    
    def invoke(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


def preprocess_metadata(metadata):
    """Preprocess metadata to handle complex data types and null values"""
    processed_metadata = {}
    for key, value in metadata.items():
        if value is None:
            processed_metadata[key] = ''
        elif isinstance(value, list):
            processed_metadata[key] = ', '.join(str(item) for item in value)
        elif isinstance(value, dict):
            processed_metadata[key] = json.dumps(value)
        else:
            processed_metadata[key] = value
    return processed_metadata


class RAGApp:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        genai.configure(api_key=api_key)
        
        # List available models
        try:
            models = genai.list_models()
            print("Available models:", [model.name for model in models])
            
            # Use gemini-1.5-pro-latest as it's the latest stable model
            model_name = "gemini-1.5-pro-latest"
            self.model = genai.GenerativeModel(model_name)
            print(f"Using model: {model_name}")
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            raise
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config["embedding"]["model"]
        )
        
        # Initialize vector store with persistence
        vector_db_dir = os.path.abspath(self.config["embedding"]["db_dir"])
        os.makedirs(vector_db_dir, exist_ok=True)
        
        # Initialize Chroma with the correct embedding function
        print(f"Initializing Chroma with directory: {vector_db_dir}")
        
        # Make sure to initialize with embedding function always set
        self.vectorstore = Chroma(
            persist_directory=vector_db_dir,
            embedding_function=self.embeddings,
            collection_name="notion_docs",
        )
        
        # Explicitly set the embedding function on the collection to ensure it's available for queries
        if hasattr(self.vectorstore, '_collection'):
            # Try to set at collection level if possible
            try:
                if not hasattr(self.vectorstore._collection, '_embedding_function'):
                    self.vectorstore._collection._embedding_function = self.embeddings
            except:
                print("Could not set embedding function on collection directly")
        
        # Initialize custom retriever
        self.retriever = CustomHybridRetriever(
            vectorstore=self.vectorstore,
            k=self.config["llm"]["k"],
            fetch_k=self.config["llm"]["fetch_k"],
            lambda_mult=0.5,  # Balance between relevance and diversity
            score_threshold=0.02  # Score difference threshold for tiebreaker
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            return_messages=True, 
            output_key="answer", 
            input_key="question"
        )
        
        # Define prompt templates for conversation
        self.condense_question_prompt = PromptTemplate.from_template(
            """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

            Chat History:
            {chat_history}
            Follow Up Input: {question}
            Standalone question:"""
        )
        
        self.answer_prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:
            {context}

            Question: {question}

            Answer (provide specific details from the context and always cite your sources with [[]]):
            """
        )
        
        # Initialize Notion client
        notion_api_key = os.getenv("NOTION_API_KEY")
        if notion_api_key:
            self.notion = Client(auth=notion_api_key)
            # List integrations on initialization
            self.list_integrations()
        else:
            self.notion = None
    
    def list_integrations(self):
        """List all Notion integrations"""
        try:
            # Get current user to verify API key
            user = self.notion.users.me()
            print("\nNotion API Key Status:")
            print(f"✅ Connected as: {user.get('name', 'Unknown User')}")
            print(f"✅ Email: {user.get('person', {}).get('email', 'Unknown')}")
            
            # Get list of integrations using the correct endpoint
            integrations = self.notion.search(
                filter={"property": "object", "value": "page"}
            )
            
            if integrations.get("results"):
                print("\nYour Notion Pages:")
                for page in integrations["results"]:
                    print(f"🔹 {page.get('properties', {}).get('title', {}).get('title', [{}])[0].get('plain_text', 'Untitled Page')}")
                    print(f"   ID: {page.get('id')}")
                    print(f"   URL: {page.get('url')}")
                    print("   ---")
            else:
                print("\nNo pages found. Please ensure your integration has access to the workspace.")
        except Exception as e:
            print(f"\nError listing pages: {str(e)}")
            print("Please check your Notion API key and make sure it's valid.")

    def load_from_notion_database(self, database_id=None):
        """Load documents from a Notion database"""
        try:
            if not self.notion:
                return "Error: Notion API key not configured. Please set NOTION_API_KEY in your environment variables."
            
            if not database_id:
                database_id = os.getenv("NOTION_DATABASE_ID")
                if not database_id:
                    # Try to get the first database available
                    databases = self.notion.search(filter={"property": "object", "value": "database"})
                    if databases.get("results"):
                        database_id = databases["results"][0]["id"]
                    else:
                        return "Error: No database ID provided and no databases found in your Notion workspace."
            
            print(f"\nLoading documents from Notion database: {database_id}")
            
            # Query all pages in the database
            pages = self.notion.databases.query(database_id=database_id)
            
            all_docs = []
            
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
                content = self.get_notion_page_content(page_id)
                if content.startswith("Error"):
                    print(f"Error loading page {page_id}: {content}")
                    continue
                
                # Create metadata
                metadata = {
                    "page_id": page_id,
                    "title": page_title,
                    "url": page.get("url", ""),
                    "last_edited": page.get("last_edited_time", ""),
                }
                
                # Add other properties to metadata
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
                
                # Process metadata
                metadata = preprocess_metadata(metadata)
                
                # Create document
                doc = Document(page_content=content, metadata=metadata)
                all_docs.append(doc)
            
            # Split documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Split and maintain metadata
            split_docs = []
            for doc in all_docs:
                chunks = splitter.split_documents([doc])
                for i, chunk in enumerate(chunks):
                    # Add chunk index to metadata
                    chunk_metadata = chunk.metadata.copy()
                    chunk_metadata["chunk_index"] = i
                    chunk.metadata = chunk_metadata
                    split_docs.append(chunk)
            
            # Add to vector store
            if split_docs:
                self.vectorstore.add_documents(split_docs)
                print(f"Successfully added {len(split_docs)} document chunks from {len(all_docs)} pages")
                return f"Successfully indexed {len(all_docs)} pages from Notion database"
            else:
                return "No documents found in the database"
            
        except Exception as e:
            import traceback
            print(f"Error loading from Notion database: {e}")
            print(traceback.format_exc())
            return f"Error loading from Notion database: {str(e)}"
    
    def get_indexed_pages(self):
        """Return a list of indexed pages"""
        try:
            # First try a simple similarity search to get documents
            try:
                # Use a very generic query that should return results
                docs = self.vectorstore.similarity_search(
                    query="",
                    k=100
                )
                
                # Extract unique pages
                indexed_pages = {}
                for doc in docs:
                    metadata = doc.metadata
                    if metadata and 'page_id' in metadata:
                        page_id = metadata['page_id']
                        # Try different metadata fields for title
                        page_title = metadata.get('page_title', 
                                      metadata.get('title', 
                                      'Untitled'))
                        if page_id and page_id not in indexed_pages:
                            indexed_pages[page_id] = page_title
                
                if indexed_pages:
                    return indexed_pages
            except Exception as e:
                print(f"First attempt failed: {e}")
                # Continue to fallback methods
            
            # Fallback approach - get all ids from collection
            try:
                # Access the collection directly and get all ids
                all_ids = self.vectorstore._collection.get()["ids"]
                if all_ids:
                    # If we have ids but no metadata, return a count
                    return {"unknown": f"Found {len(all_ids)} documents (metadata unavailable)"}
            except Exception as e:
                print(f"Second attempt failed: {e}")
                # Continue to final fallback
            
            # Final fallback
            return {}
        except Exception as e:
            import traceback
            print(f"Error getting indexed pages: {e}")
            print(traceback.format_exc())
            return {}
    
    def extract_page_id(self, url: str) -> str:
        """Extract page ID from Notion URL"""
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
    
    def get_notion_page_content(self, page_id: str) -> str:
        """Fetch content from Notion page"""
        if not self.notion:
            return "Error: Notion API key not configured. Please set NOTION_API_KEY in your environment variables."
        
        try:
            print(f"\nRetrieving content from Notion page: {page_id}")
            
            # Get page content
            page = self.notion.pages.retrieve(page_id=page_id)
            page_title = page.get('properties', {}).get('title', {}).get('title', [{}])[0].get('plain_text', 'Untitled Page')
            print(f"Page title: {page_title}")
            
            blocks = self.notion.blocks.children.list(block_id=page_id)
            print(f"Found {len(blocks.get('results', []))} blocks")
            
            # Extract text content from blocks
            content = []
            for block in blocks.get("results", []):
                block_type = block.get("type")
                rich_text = None
                
                # Get rich text content based on block type
                if block_type == "paragraph":
                    rich_text = block.get("paragraph", {}).get("rich_text", [])
                elif block_type == "heading_1":
                    rich_text = block.get("heading_1", {}).get("rich_text", [])
                    if rich_text:
                        content.append("# " + rich_text[0].get("plain_text", ""))
                elif block_type == "heading_2":
                    rich_text = block.get("heading_2", {}).get("rich_text", [])
                    if rich_text:
                        content.append("## " + rich_text[0].get("plain_text", ""))
                elif block_type == "heading_3":
                    rich_text = block.get("heading_3", {}).get("rich_text", [])
                    if rich_text:
                        content.append("### " + rich_text[0].get("plain_text", ""))
                elif block_type == "bulleted_list_item":
                    rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                    if rich_text:
                        content.append("- " + rich_text[0].get("plain_text", ""))
                elif block_type == "numbered_list_item":
                    rich_text = block.get("numbered_list_item", {}).get("rich_text", [])
                    if rich_text:
                        content.append("1. " + rich_text[0].get("plain_text", ""))
                elif block_type == "to_do":
                    rich_text = block.get("to_do", {}).get("rich_text", [])
                    if rich_text:
                        checked = "✓" if block.get("to_do", {}).get("checked", False) else "☐"
                        content.append(f"{checked} " + rich_text[0].get("plain_text", ""))
                elif block_type == "toggle":
                    rich_text = block.get("toggle", {}).get("rich_text", [])
                    if rich_text:
                        content.append("> " + rich_text[0].get("plain_text", ""))
                elif block_type == "quote":
                    rich_text = block.get("quote", {}).get("rich_text", [])
                    if rich_text:
                        content.append("> " + rich_text[0].get("plain_text", ""))
                elif block_type == "callout":
                    rich_text = block.get("callout", {}).get("rich_text", [])
                    if rich_text:
                        content.append("💡 " + rich_text[0].get("plain_text", ""))
                
                # Handle regular text blocks
                if rich_text and block_type not in ["heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item", "to_do", "toggle", "quote", "callout"]:
                    if rich_text:
                        content.append(rich_text[0].get("plain_text", ""))
            
            if not content:
                return f"Page '{page_title}' is empty or contains no text content."
            
            # Add page title as the first line
            content.insert(0, f"# {page_title}")
            
            return "\n".join(content)
        except Exception as e:
            error_msg = str(e)
            print(f"Error fetching Notion content: {error_msg}")
            if "Could not find page" in error_msg:
                return f"""Error: Could not access the Notion page. Please ensure:
1. The page is shared with your integration
2. Your integration has the correct permissions
3. The page ID is correct

To share a page with your integration:
1. Open the page in Notion
2. Click the 'Share' button
3. Click 'Add people, emails, groups, or integrations'
4. Search for your integration name
5. Select your integration and click 'Invite'

Current integration status:
✅ Connected as: {self.notion.users.me().get('name', 'Unknown User')}
"""
            return f"Error fetching Notion content: {error_msg}"

    def format_context_with_hierarchy(self, docs: List[Document]) -> str:
        """Format context with heading hierarchy and metadata"""
        context_parts = []
        
        for doc in docs:
            metadata = doc.metadata
            # Try to get page title from different possible metadata fields
            page_title = metadata.get("page_title", 
                         metadata.get("title", "Unknown Page"))
            h1 = metadata.get("h1", "")
            h2 = metadata.get("h2", "")
            h3 = metadata.get("h3", "")
            block_id = metadata.get("block_id", "")
            page_id = metadata.get("page_id", "")
            url = metadata.get("url", "")
            
            # Create the heading hierarchy
            heading_hierarchy = page_title
            if h1:
                heading_hierarchy += f" > {h1}"
            if h2:
                heading_hierarchy += f" › {h2}"
            if h3:
                heading_hierarchy += f" › {h3}"
            
            # Use page_id as fallback for block_id if needed
            citation_id = block_id if block_id else page_id
            
            # Add to context with citation for post-processing
            # Include source title for better citation in final output
            context_parts.append(f"- {heading_hierarchy}\n  {doc.page_content}\n  [[{citation_id}|{page_title}]]")
        
        return "\n---\n".join(context_parts)
    
    def post_process_response(self, response: str) -> str:
        """Replace citation markers with canonical Notion URLs and source titles"""
        # Find all citations with format [[id|title]]
        citations = re.findall(r'\[\[([a-f0-9-]+)\|([^\]]+)\]\]', response)
        
        # Replace each citation with its canonical URL and title
        for citation_id, title in citations:
            # Extract page ID for this citation
            page_id = None
            # Check if we have this in our vector store
            query_results = self.vectorstore.similarity_search(
                f"id:{citation_id}", k=1
            )
            if query_results:
                page_id = query_results[0].metadata.get("page_id")
            
            if page_id:
                canonical_url = f"https://www.notion.so/{page_id}#{citation_id}"
                # Replace with formatted citation that includes title
                response = response.replace(f"[[{citation_id}|{title}]]", f"[[{title}]]")
            else:
                # Just keep the title if we can't find the URL
                response = response.replace(f"[[{citation_id}|{title}]]", f"[[{title}]]")
        
        # Also handle old-style citations with just IDs
        block_ids = re.findall(r'\[\[([a-f0-9-]+)\]\]', response)
        for block_id in block_ids:
            # Extract page ID for this block
            page_id = None
            # Check if we have this block in our vector store
            query_results = self.vectorstore.similarity_search(
                f"block_id:{block_id}", k=1
            )
            if query_results:
                page_id = query_results[0].metadata.get("page_id")
                title = query_results[0].metadata.get("page_title", 
                        query_results[0].metadata.get("title", "Source"))
            
            if page_id:
                canonical_url = f"https://www.notion.so/{page_id}#{block_id}"
                response = response.replace(f"[[{block_id}]]", f"[[{title}]]")
            else:
                # If we can't find the page ID, just remove the block ID
                response = response.replace(f"[[{block_id}]]", "[[Source]]")
        
        return response
    
    def get_conversational_response(self, query: str, chat_history=None):
        """Generate a response using conversational memory"""
        try:
            # Load memory
            memory_vars = self.memory.load_memory_variables({})
            chat_history = memory_vars.get("history", [])
            
            # Condense the question if we have chat history
            if chat_history:
                standalone_question = self.condense_question_prompt.format(
                    chat_history=get_buffer_string(chat_history),
                    question=query
                )
                # Use language model to get standalone question
                response = self.model.generate_content(standalone_question)
                standalone_query = response.text
                print(f"Standalone query: {standalone_query}")
            else:
                standalone_query = query
            
            # Get documents
            docs = self.retriever.invoke(standalone_query)
            
            if not docs:
                result = "I don't have any relevant information to answer that question."
                self.memory.save_context({"question": query}, {"answer": result})
                return result, []
            
            # Create context
            context = self.format_context_with_hierarchy(docs)
            
            # Generate answer
            prompt = self.answer_prompt.format(
                context=context,
                question=standalone_query
            )
            
            response = self.model.generate_content(prompt)
            answer = response.text
            
            # Post-process to replace citations
            processed_answer = self.post_process_response(answer)
            
            # Save to memory
            self.memory.save_context({"question": query}, {"answer": processed_answer})
            
            # Extract unique citations
            citations = set(re.findall(r'\[\[([^\]]+)\]\]', processed_answer))
            
            return processed_answer, list(citations)
            
        except Exception as e:
            import traceback
            print(f"Error in get_conversational_response: {e}")
            print(traceback.format_exc())
            return f"Error generating response: {str(e)}", []
    
    def get_response(self, query: str, search_type: str = None, k: int = None, fetch_k: int = None, eval_mode: bool = False) -> str:
        try:
            # Check if this is a database loading request
            if "load database" in query.lower() and "notion" in query.lower():
                database_id = None
                # Try to extract database ID if present
                match = re.search(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', query)
                if match:
                    database_id = match.group(1)
                
                return self.load_from_notion_database(database_id)
            
            # Check if query contains a Notion URL
            if "notion.so" in query:
                page_id = self.extract_page_id(query)
                if not page_id:
                    return "Error: Could not extract a valid page ID from the provided Notion URL. Please check the URL format."
                
                print(f"\nExtracted page ID: {page_id}")
                
                # Check if this page is already indexed
                already_indexed = False
                try:
                    # Use a simpler method to check if page exists
                    existing_docs = self.vectorstore.similarity_search(
                        f"page_id:{page_id}", k=1
                    )
                    if existing_docs:
                        already_indexed = True
                        print(f"Page {page_id} is already indexed. Using existing content.")
                        # Set new query to summarize the indexed content if this is a direct URL request
                        if "notion.so" in query and len(query.strip().split()) < 10:
                            # This appears to be just a URL without a question
                            page_title = existing_docs[0].metadata.get("page_title", 
                                         existing_docs[0].metadata.get("title", "this Notion page"))
                            return f"I've found '{page_title}' in my index. You can now ask me questions about it."
                except Exception as e:
                    print(f"Error checking if page is indexed: {e}")
                
                # Get Notion page content
                notion_content = self.get_notion_page_content(page_id)
                if notion_content.startswith("Error"):
                    return notion_content
                
                # Index this content in our vector store
                print("Indexing Notion page content...")
                try:
                    # Get page metadata
                    page = self.notion.pages.retrieve(page_id=page_id)
                    page_title = page.get('properties', {}).get('title', {}).get('title', [{}])[0].get('plain_text', 'Untitled Page')
                    page_url = page.get('url', '')
                    last_edited = page.get('last_edited_time', '')
                    
                    # Create metadata for this document
                    metadata = {
                        "page_title": page_title,
                        "title": page_title,  # Add both for compatibility
                        "page_id": page_id,
                        "url": page_url,
                        "last_edited": last_edited,
                        "block_id": page_id  # Use page_id as block_id for the entire page
                    }
                    
                    # Add document to vector store using our helper method
                    success = self.add_document_to_vectorstore(notion_content, metadata)
                    
                    if not success:
                        return "Error indexing Notion content. Please try again."
                    
                    # If this is just a URL (not a question about the content)
                    if "notion.so" in query and len(query.strip().split()) < 10:
                        return f"Successfully indexed Notion page '{page_title}'. You can now ask questions about it."
                    
                    # Otherwise, set query to search for the indexed content
                    query = f"Summarize the contents of the Notion page titled '{page_title}'"
                except Exception as e:
                    print(f"Error indexing Notion content: {e}")
                    return f"Error indexing Notion content: {str(e)}"
                
                print(f"\nNew query: {query}")
            
            # Use the conversational response system
            answer, citations = self.get_conversational_response(query)
            return answer
            
        except Exception as e:
            import traceback
            print(f"Error in get_response: {e}")
            print(traceback.format_exc())
            return f"Error generating response: {str(e)}"

    def reset_vector_store(self):
        """Reset the vector store by deleting and reinitializing it"""
        try:
            import shutil
            
            # Get the directory path
            vector_db_dir = os.path.abspath(self.config["embedding"]["db_dir"])
            
            # Delete the directory if it exists
            if os.path.exists(vector_db_dir):
                shutil.rmtree(vector_db_dir)
                print(f"Deleted vector store directory: {vector_db_dir}")
            
            # Recreate the directory
            os.makedirs(vector_db_dir, exist_ok=True)
            
            # Re-initialize the vector store
            self.vectorstore = Chroma(
                persist_directory=vector_db_dir,
                embedding_function=self.embeddings,
                collection_name="notion_docs"
            )
            
            # Re-initialize the retriever
            self.retriever = CustomHybridRetriever(
                vectorstore=self.vectorstore,
                k=self.config["llm"]["k"],
                fetch_k=self.config["llm"]["fetch_k"],
                lambda_mult=0.5,
                score_threshold=0.02
            )
            
            # Reset conversation memory
            self.memory.clear()
            
            print("Vector store reset successfully")
            return True
        except Exception as e:
            import traceback
            print(f"Error resetting vector store: {e}")
            print(traceback.format_exc())
            return False

    def add_document_to_vectorstore(self, document, metadata):
        """Add a document to the vector store with proper error handling"""
        try:
            from langchain.schema import Document
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            # Create document object
            main_doc = Document(page_content=document, metadata=metadata)
            
            # Split into chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Split the document into chunks
            doc_chunks = text_splitter.split_documents([main_doc])
            
            # Make sure each chunk has the same metadata
            for i, chunk in enumerate(doc_chunks):
                # Add the chunk index to differentiate between chunks
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk.metadata = chunk_metadata
            
            print(f"Split document into {len(doc_chunks)} chunks for better retrieval")
            
            # Try to add to vector store
            self.vectorstore.add_documents(doc_chunks)
            print(f"Successfully added {len(doc_chunks)} document chunks to vector store: {metadata.get('page_title', 'Unknown')}")
            return True
        except Exception as e:
            import traceback
            print(f"Error adding document to vector store: {e}")
            print(traceback.format_exc())
            return False


def main():
    # load config
    with open('config.yaml') as file:
        config = yaml.safe_load(file.read())
    
    rag_app = RAGApp()
    
    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        response = rag_app.get_response(query)
        print("\nResponse:")
        print(response)
        print("\n")


if __name__ == "__main__":
    main()

