# RAG Notion Application

A Streamlit-based Retrieval Augmented Generation (RAG) application with Notion integration that answers questions based on content from your Notion pages.

## Features

- Chat interface for question answering with source attribution
- Notion integration for loading content from your Notion pages
- Google Gemini AI model for generating accurate responses
- Vector storage using ChromaDB for efficient document retrieval
- Customizable settings for fine-tuning the RAG system

## Setup

### Prerequisites

- Python 3.9+
- Notion API Integration
- Google Gemini API key

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-notion-app
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with the following variables:
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   NOTION_API_KEY=your_notion_api_key
   NOTION_PAGE_IDS=comma,separated,notion,page,ids
   TEMPERATURE=0.7
   ```

### Notion API Setup

1. Go to [Notion Integrations](https://www.notion.so/my-integrations)
2. Create a new integration
3. Copy the "Internal Integration Secret" and add it to your `.env` file
4. Share your Notion pages with the integration:
   - Open the Notion page you want to use
   - Click the "..." menu in the top right
   - Select "Add connections"
   - Choose your integration

## Usage

Run the Streamlit application:

```bash
streamlit run ui/main.py
```

### Chat Interface

- Ask questions about documents in your knowledge base
- View sources for each answer
- Chat history is preserved during the session

### Settings

- **Model Settings**: Adjust the temperature to control creativity
- **Notion Integration**: Configure your API key and page IDs
- **Load Pages**: Add individual or multiple Notion pages to the knowledge base

## Project Structure

```
rag-notion-app/
├── ui/
│   ├── main.py                 # Streamlit application entry point
│   ├── rag_app.py              # RAG application core functionality
│   ├── document_processor.py   # Document processing utilities
│   ├── notion_processor.py     # Notion integration and processing
│   ├── notion_manager.py       # Notion API wrapper
│   └── vector_store.py         # ChromaDB vector store wrapper
├── config.yaml                 # Application configuration
├── .env                        # Environment variables (not in repository)
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Configuration

The application can be configured by modifying the `config.yaml` file:

- **Gemini**: Model parameters like temperature and token limits
- **Chroma**: Vector store settings
- **Embeddings**: Model selection for embeddings
- **Document Processor**: Chunking parameters
- **Notion**: API settings

## License

[MIT License](LICENSE)

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the UI framework
- [LangChain](https://langchain.readthedocs.io/) for the document processing pipeline
- [ChromaDB](https://www.trychroma.com/) for the vector database
- [Google Gemini](https://ai.google.dev/) for the large language model
- [Notion API](https://developers.notion.com/) for document integration
