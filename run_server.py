"""Run server with logging"""
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

# Add project to path
sys.path.insert(0, 'c:/Users/spars/Desktop/New folder')
os.chdir('c:/Users/spars/Desktop/New folder')

# Load env
from dotenv import load_dotenv
load_dotenv()

logger.info("Starting server...")

# Import app
from src.app import app

# Run server
import uvicorn
logger.info("Running uvicorn...")
uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

