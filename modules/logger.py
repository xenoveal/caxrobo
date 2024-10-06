import logging
import os

def setup_logger(name, log_file, level=logging.DEBUG):
    """Function to set up a logger.
    
    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level: Logging level (default is DEBUG).
    
    Returns:
        logger: Configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create file handler for writing logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Create console handler for outputting logs to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatter to handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
