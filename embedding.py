import json
import os
import yaml
import uuid
from typing import Dict, List, Any
from dotenv import load_dotenv

# langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document  
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class JSONHandler:
    @staticmethod  # no need to create class instance 
    def load_json_data(file_path):
        """
        Load data from a JSON file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def create_documents_from_structured_data(structured_data):
        """
        Convert structured data with content and metadata into document type.
        """
        documents = []
        for item in structured_data:
            content = item.get("content", "")
            metadata = item.get("metadata", {})
            
            # Ensure deterministic UUID if not already present
            if "uuid" not in metadata:
                # Create a deterministic UUID based on content and metadata
                content_hash = hash(content + str(sorted(metadata.items())))
                metadata["uuid"] = str(uuid.UUID(int=abs(content_hash) % (2**128)))
                
            documents.append(Document(page_content=content, metadata=metadata))
        return documents


class Embedding:
    def __init__(self, embedding_model_name, persist_directory, chunk_size=1024, chunk_overlap=100, separator='\n'):
        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

        load_dotenv()
        HUGGING_FACE_API_KEY = os.getenv('HUGGING_FACE_API_KEY')

        if HUGGING_FACE_API_KEY:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGING_FACE_API_KEY
            print("Embedding: Hugging Face API Key set successfully.")
        else:
            raise ValueError("HUGGING_FACE_API_KEY is not set in the .env file.")

    def create_embeddings(self):
        return HuggingFaceEmbeddings(model_name=self.embedding_model_name)

    def clean_text(self, text):
        """
        Clean text data by stripping empty lines.
        """
        return " ".join(line.strip() for line in text.splitlines() if line.strip())

    def split_and_clean_documents(self, documents):
        """
        Splitting the texts into chunks and cleaning.
        """
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator=self.separator
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Ensure each chunk maintains the original metadata
        cleaned_docs = []
        for doc in split_docs:
            # Copy original metadata
            doc_metadata = doc.metadata.copy()
            
            # Ensure deterministic UUID for each chunk
            if "uuid" in doc_metadata:
                # Append chunk identifier to make unique but deterministic UUIDs for each chunk
                original_uuid = doc_metadata["uuid"]
                content_hash = hash(doc.page_content)
                chunk_uuid = f"{original_uuid}-chunk-{abs(content_hash) % 1000000}"
                doc_metadata["uuid"] = chunk_uuid
            
            cleaned_docs.append(
                Document(
                    page_content=self.clean_text(doc.page_content),
                    metadata=doc_metadata
                )
            )
        return cleaned_docs

    def create_and_persist_chroma_db(self, cleaned_docs, recreate=False):
        """
        Create and save Chroma database.
        """
        embedding_function = self.create_embeddings()

        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
        
        # Check if database should be recreated
        if recreate and os.path.exists(self.persist_directory):
            import shutil
            print(f"Recreating Chroma database at {self.persist_directory}")
            # Create backup
            backup_dir = f"{self.persist_directory}_backup_{int(os.path.getmtime(self.persist_directory))}"
            shutil.copytree(self.persist_directory, backup_dir)
            print(f"Created backup at {backup_dir}")
            # Remove existing
            shutil.rmtree(self.persist_directory)
            os.makedirs(self.persist_directory)

        db = Chroma.from_documents(
            cleaned_docs,
            embedding_function,
            persist_directory=self.persist_directory,
        )
        print(f"Database saved successfully to disk at {self.persist_directory}")
        return db


def main():
    # Load config
    with open('config.yaml') as file:
        config = yaml.safe_load(file.read())

    try:
        # Load and process the structured data
        structured_data = JSONHandler.load_json_data(config["notion"]["content_file"])
        documents = JSONHandler.create_documents_from_structured_data(structured_data)
        print(f"Loaded {len(documents)} documents with metadata")

        # Create EmbeddingHandler instance and process the documents
        embedding_handler = Embedding(
            embedding_model_name=config["embedding"]["model"],
            persist_directory=config["embedding"]["db_dir"],
            chunk_size=config["embedding"]["chunk_size"],
            chunk_overlap=config["embedding"]["overlap"],
        )

        # Check if reindexing is needed
        reindex = config.get("embedding", {}).get("reindex", False)
        
        cleaned_docs = embedding_handler.split_and_clean_documents(documents)
        print(f"Split into {len(cleaned_docs)} chunks with preserved metadata")
        
        db = embedding_handler.create_and_persist_chroma_db(cleaned_docs, recreate=reindex)
        print(f"Created Chroma DB with {len(cleaned_docs)} documents")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
