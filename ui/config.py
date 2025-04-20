import os
from pathlib import Path

# Define base paths
BASE_DIR = Path(__file__).parent.parent
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Logo paths
SIDEBAR_LOGO_PATH = os.path.join(ASSETS_DIR, "sidebar_logo.png")
MAIN_LOGO_PATH = os.path.join(ASSETS_DIR, "main_logo.png")

# ChromaDB settings
CHROMA_DB_DIR = os.path.join(BASE_DIR, "ui", "chroma_db") 