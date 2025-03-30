import logging
import os

def setup_logger(name: str, log_file: str = "logs/prim.log", level=logging.INFO):
    """
    Set up a logger for an agent with a specified log file.

    Args:
        name (str): The name of the logger (typically the agent's name).
        log_file (str): Path to the log file.
        level (int): Logging level (default: INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    if not os.path.exists("logs"):
        os.makedirs("logs")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(name)s - %(message)s'))

    # Avoid duplicate handlers
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
