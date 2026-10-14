import logging
from pathlib import Path

"""
This module sets up a logger for the AI Research Assistant application.
It configures the logger to write logs to a file in the "logs" directory, 
with a specific format for log messages.    
"""

# Create a directory for logs if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Function to set up the logger
def setup_logger(name="ai_research_assistant"):

    logger = logging.getLogger(name)

    # Set the logging level to INFO
    logger.setLevel(logging.INFO)

    # Define a log message format
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(
        LOG_DIR / "system.log"
    )

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger if it doesn't already have handlers
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger