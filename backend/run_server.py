"""
Run server with logging
"""
import sys
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

# Load env
from dotenv import load_dotenv
load_dotenv()

logger.info("Starting server...")

# Import app
from backend.main import app

# Run server
import uvicorn
logger.info("Running uvicorn...")
uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

