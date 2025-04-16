from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from chromadb.config import Settings
import uvicorn

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB with server settings
chroma_client = chromadb.Client(Settings(
    chroma_server_host="0.0.0.0",
    chroma_server_http_port=8000,
    allow_reset=True
))

# Create a collection
collection = chroma_client.create_collection(name="notion_docs")

# Add some sample documents
documents = [
    "This is a sample document about artificial intelligence and machine learning.",
    "Here's another document discussing natural language processing and transformers.",
    "This document covers the basics of deep learning and neural networks.",
]

collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 