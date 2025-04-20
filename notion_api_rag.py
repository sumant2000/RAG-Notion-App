import os
from typing import List, Dict, Any
from notion_client import Client
import chromadb
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated to correct package
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline  # Changed to HF Pipeline for local models
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv(override=True)
print("Loading environment variables and initializing the RAG system...")

class NotionAPIRAG:
    def __init__(self, notion_token: str, persist_directory: str = "./chroma_db"):
        """Initialize the NotionRAG system with official Notion API."""
        self.persist_directory = persist_directory
        
        # Initialize Notion client
        self.notion = Client(auth=notion_token)
        
        # Use local sentence transformer model for embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Small, fast model that doesn't require API key
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize ChromaDB
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize LLM with a local TinyLlama model
        print("Loading language model... (this might take a minute)")
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model that runs on CPU
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="auto"
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,  # Enable sampling to use temperature and top_p
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # Initialize QA chain
        self.qa_chain = self._setup_qa_chain()
    
    def _setup_qa_chain(self) -> RetrievalQA:
        """Set up the QA chain with custom prompt."""
        prompt_template = """
        You are an assistant that answers questions based on the provided context from Notion documents.
        
        Context: {context}
        
        Question: {question}
        
        Answer the question based only on the provided context. If you don't know the answer, say "I don't have enough information to answer this question."
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        retriever = self.db.as_retriever(search_kwargs={"k": 3})
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def extract_page_content(self, page_id: str) -> str:
        """Extract content from a Notion page using the official API."""
        try:
            # Get page properties
            page = self.notion.pages.retrieve(page_id)
            
            # Get page blocks (content)
            blocks = self.notion.blocks.children.list(page_id)
            
            content = ""
            
            # Process blocks recursively
            for block in blocks["results"]:
                content += self._process_block(block)
            
            return content
        except Exception as e:
            print(f"Error extracting content from page {page_id}: {e}")
            return ""
    
    def _process_block(self, block: Dict[str, Any], indent: int = 0) -> str:
        """Process a Notion block and extract its text content."""
        block_type = block["type"]
        text = ""
        
        if block_type == "paragraph":
            for rich_text in block["paragraph"]["rich_text"]:
                text += rich_text["plain_text"]
            text += "\n\n"
        
        elif block_type == "heading_1":
            for rich_text in block["heading_1"]["rich_text"]:
                text += rich_text["plain_text"]
            text += "\n\n"
        
        elif block_type == "heading_2":
            for rich_text in block["heading_2"]["rich_text"]:
                text += rich_text["plain_text"]
            text += "\n\n"
        
        elif block_type == "heading_3":
            for rich_text in block["heading_3"]["rich_text"]:
                text += rich_text["plain_text"]
            text += "\n\n"
        
        elif block_type == "bulleted_list_item":
            for rich_text in block["bulleted_list_item"]["rich_text"]:
                text += "• " + rich_text["plain_text"]
            text += "\n"
        
        elif block_type == "numbered_list_item":
            for rich_text in block["numbered_list_item"]["rich_text"]:
                text += "1. " + rich_text["plain_text"]  # Simplified numbering
            text += "\n"
        
        # Process child blocks if they exist
        if "has_children" in block and block["has_children"]:
            try:
                children = self.notion.blocks.children.list(block["id"])
                for child in children["results"]:
                    text += self._process_block(child, indent + 1)
            except Exception as e:
                print(f"Error processing child blocks: {e}")
        
        return text
    
    def page_id_from_url(self, url: str) -> str:
        """Extract page ID from a Notion URL."""
        # Check if the input is already a page ID with hyphens
        if len(url.split("-")) == 5 and all(len(part) in [4, 8, 12] or (i == 0 and len(part) == 8) for i, part in enumerate(url.split("-"))):
            # This appears to be a complete UUID already formatted correctly
            return url
        
        # First, remove any query parameters
        url = url.split("?")[0]
        
        # Extract the last part of the URL
        last_part = url.split("/")[-1]
        
        # Return the full ID as is - don't try to extract just the last segment
        return last_part
    
    def add_notion_document(self, url: str) -> str:
        """Add a Notion document to the vector database using its URL."""
        page_id = self.page_id_from_url(url)
        
        if not page_id:
            return f"Failed to extract page ID from {url}"
        
        # Extract content from the Notion page
        content = self.extract_page_content(page_id)
        
        if not content:
            return f"Failed to extract content from {url}"
        
        # Split the content into chunks
        chunks = self.text_splitter.split_text(content)
        
        # Add metadata to each chunk
        texts = chunks
        metadatas = [{"source": url, "page_id": page_id, "chunk": i} for i in range(len(chunks))]
        
        # Add to ChromaDB
        self.db.add_texts(texts=texts, metadatas=metadatas)
        
        return f"Added document {page_id} with {len(chunks)} chunks to the database"
    
    def add_multiple_notion_documents(self, urls: List[str]) -> List[str]:
        """Add multiple Notion documents to the vector database."""
        results = []
        for url in urls:
            result = self.add_notion_document(url)
            results.append(result)
        return results
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question based on the stored Notion documents."""
        result = self.qa_chain.invoke({"query": question})  # Changed from __call__ to invoke
        
        # Format the response
        response = {
            "answer": result["result"],
            "sources": [doc.metadata["source"] for doc in result["source_documents"]]
        }
        
        return response

# Example usage
if __name__ == "__main__":
    # Get Notion API token from environment variable - try both variable names for compatibility
    notion_token = os.environ.get("NOTION_API_KEY") or os.environ.get("NOTION_TOKEN")
    
    if not notion_token:
        print("Please set the NOTION_API_KEY or NOTION_TOKEN environment variable")
        exit(1)
    
    # Initialize the NotionRAG system
    notion_rag = NotionAPIRAG(notion_token)
    
    # Get Notion page IDs from environment variable
    notion_page_ids = os.environ.get("NOTION_PAGE_IDS", "").split(",")
    
    # Filter out empty strings
    notion_page_ids = [pid.strip() for pid in notion_page_ids if pid.strip()]
    
    if not notion_page_ids:
        print("No Notion page IDs found. Please set the NOTION_PAGE_IDS environment variable.")
        print("You can continue and manually enter page URLs when prompted.")
    
    # Create URLs from the page IDs or use sample URLs if none are provided
    urls = []
    if notion_page_ids:
        for page_id in notion_page_ids:
            # Create a direct URL to the page if it's a UUID
            if "-" in page_id and len(page_id.replace("-", "")) == 32:
                urls.append(f"https://www.notion.so/{page_id}")
            else:
                urls.append(page_id)  # Might already be a full URL
    
    # If no URLs from environment, ask the user
    if not urls:
        print("\nPlease enter a Notion page URL or ID (or leave empty to skip):")
        user_url = input("> ").strip()
        if user_url:
            urls.append(user_url)
    
    if urls:
        print("Adding Notion documents to the database...")
        results = notion_rag.add_multiple_notion_documents(urls)
        for result in results:
            print(result)
    else:
        print("No Notion pages to add. Continuing with existing database content.")
    
    # Example: Answer questions
    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        
        try:
            response = notion_rag.answer_question(question)
            print("\nAnswer:", response["answer"])
            print("\nSources:")
            for source in response["sources"]:
                print(f"- {source}")
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure you have added Notion documents and set up the correct API keys.")