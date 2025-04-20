import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from uuid import uuid4
from pathlib import Path
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Process documents for vector storage and retrieval.
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            separators: Custom separators for text splitting
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
        
        logger.info(f"Initialized DocumentProcessor with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def process_document(self, document: Document) -> List[Document]:
        """
        Process a document by splitting it into chunks and processing metadata.
        
        Args:
            document: The input document
            
        Returns:
            List of processed document chunks
        """
        try:
            # Split document into chunks
            chunks = self.text_splitter.split_documents([document])
            
            # Process metadata for each chunk
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                # Ensure metadata is JSON serializable
                metadata = self._process_metadata(chunk.metadata)
                
                # Add chunk index to metadata
                metadata["chunk_index"] = i
                metadata["chunk_count"] = len(chunks)
                
                # Create new document with processed metadata
                processed_chunk = Document(
                    page_content=chunk.page_content,
                    metadata=metadata
                )
                processed_chunks.append(processed_chunk)
            
            logger.info(f"Split document into {len(processed_chunks)} chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            # Return the original document as a single chunk if processing fails
            return [document]
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process multiple documents.
        
        Args:
            documents: List of input documents
            
        Returns:
            List of processed document chunks
        """
        all_chunks = []
        for doc in documents:
            chunks = self.process_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks
    
    def _process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process document metadata to ensure it's JSON serializable.
        
        Args:
            metadata: Original metadata
            
        Returns:
            Processed metadata
        """
        processed = {}
        
        for key, value in metadata.items():
            # Convert non-serializable types to strings
            if isinstance(value, (dict, list)):
                try:
                    # Test if it's already serializable
                    json.dumps(value)
                    processed[key] = value
                except (TypeError, OverflowError):
                    processed[key] = str(value)
            elif value is None or isinstance(value, (str, int, float, bool)):
                processed[key] = value
            else:
                processed[key] = str(value)
        
        return processed
    
    def update_configuration(
        self, 
        chunk_size: Optional[int] = None, 
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None
    ) -> None:
        """
        Update the configuration of the processor.
        
        Args:
            chunk_size: New chunk size
            chunk_overlap: New chunk overlap
            separators: New separators
        """
        if chunk_size is not None:
            self.chunk_size = chunk_size
        
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
            
        if separators is not None:
            self.separators = separators
        
        # Recreate text splitter with new configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        
        logger.info(f"Updated DocumentProcessor configuration: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
    
    def save_chunks_to_file(self, chunks: List[Document], output_dir: str, filename: Optional[str] = None) -> str:
        """
        Save processed chunks to a JSON file.
        
        Args:
            chunks: List of Document objects
            output_dir: Directory to save the file
            filename: Optional filename (default: generated from timestamp)
            
        Returns:
            Path to the saved file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename if not provided
            if not filename:
                import time
                timestamp = int(time.time())
                filename = f"chunks_{timestamp}.json"
            
            # Create full file path
            file_path = os.path.join(output_dir, filename)
            
            # Convert documents to dictionaries
            chunks_data = []
            for chunk in chunks:
                chunks_data.append({
                    "text": chunk.page_content,
                    "metadata": chunk.metadata
                })
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(chunks)} chunks to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving chunks to file: {str(e)}")
            return ""
    
    def load_chunks_from_file(self, file_path: str) -> List[Document]:
        """
        Load chunks from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of Document objects
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            # Convert dictionaries to Document objects
            chunks = []
            for chunk_data in chunks_data:
                chunk = Document(
                    page_content=chunk_data.get("text", ""),
                    metadata=chunk_data.get("metadata", {})
                )
                chunks.append(chunk)
            
            logger.info(f"Loaded {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading chunks from file: {str(e)}")
            return [] 