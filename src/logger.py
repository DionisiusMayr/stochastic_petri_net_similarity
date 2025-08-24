import logging
import sys

def setup_logger(name, log_file: str, level: int=logging.INFO) -> logging.Logger:
    """
    Helper function to set up a logger that logs both to stdout and to a file (if the argument is present).
    @return: a logging.Logger instance.
    """
    logger = logging.getLogger(name)  # Use a module-level logger
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(stream=sys.stdout)  # Create a stream handler (for terminal output)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
