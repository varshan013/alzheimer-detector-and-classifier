import logging
import os
from datetime import datetime
from src.config.configuration import path_config

os.makedirs(path_config.logs_dir, exist_ok=True)

LOG_FILE = os.path.join(
    path_config.logs_dir,
    f"alz_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def get_logger(name: str):
    return logging.getLogger(name)
