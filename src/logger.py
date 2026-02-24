import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler


# Create logs directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Main log file (rotates when it gets too large)
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Latest log file (always points to current session)
LATEST_LOG_PATH = os.path.join(LOG_DIR, "latest.log")

# Error log file (only errors and critical)
ERROR_LOG_PATH = os.path.join(LOG_DIR, "errors.log")


def setup_logging():
    """
    Setup enhanced logging with multiple handlers
    """

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # Format
    log_format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # 1. Console Handler (for terminal output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. Main File Handler (rotating, max 10MB, keep 5 backups)
    file_handler = RotatingFileHandler(
        LOG_FILE_PATH,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 3. Latest Log Handler (always overwrites - for easy access)
    latest_handler = logging.FileHandler(
        LATEST_LOG_PATH,
        mode='w',  # Overwrite mode
        encoding='utf-8'
    )
    latest_handler.setLevel(logging.INFO)
    latest_handler.setFormatter(formatter)
    logger.addHandler(latest_handler)

    # 4. Error Log Handler (only errors)
    error_handler = RotatingFileHandler(
        ERROR_LOG_PATH,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    logger.info('='*70)
    logger.info('Logging system initialized')
    logger.info('='*70)
    logger.info(f'Main log file: {LOG_FILE_PATH}')
    logger.info(f'Latest log: {LATEST_LOG_PATH}')
    logger.info(f'Error log: {ERROR_LOG_PATH}')
    logger.info('='*70)

    return logger


def get_recent_logs(n_lines=100):
    """
    Get recent log entries

    Args:
        n_lines: Number of recent lines to return

    Returns:
        List of log lines
    """
    try:
        if os.path.exists(LATEST_LOG_PATH):
            with open(LATEST_LOG_PATH, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return lines[-n_lines:]
        return []
    except Exception as e:
        return [f"Error reading logs: {str(e)}"]


def get_error_logs(n_lines=50):
    """
    Get recent error entries

    Args:
        n_lines: Number of recent error lines to return

    Returns:
        List of error log lines
    """
    try:
        if os.path.exists(ERROR_LOG_PATH):
            with open(ERROR_LOG_PATH, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return lines[-n_lines:]
        return []
    except Exception as e:
        return [f"Error reading error logs: {str(e)}"]


def get_all_log_files():
    """
    Get list of all log files

    Returns:
        List of log file names
    """
    try:
        if os.path.exists(LOG_DIR):
            files = [f for f in os.listdir(LOG_DIR) if f.endswith('.log')]
            files.sort(reverse=True)  # Most recent first
            return files
        return []
    except Exception as e:
        return [f"Error listing log files: {str(e)}"]


def get_log_file_content(filename, n_lines=None):
    """
    Get content of a specific log file

    Args:
        filename: Name of log file
        n_lines: Number of lines to return (None for all)

    Returns:
        Content of log file
    """
    try:
        filepath = os.path.join(LOG_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if n_lines:
                    return lines[-n_lines:]
                return lines
        return [f"Log file not found: {filename}"]
    except Exception as e:
        return [f"Error reading log file: {str(e)}"]


# Initialize logging when module is imported
logger = setup_logging()


if __name__ == "__main__":
    # Test logging
    logging.info("Testing enhanced logging system")
    logging.info("This is an info message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")

    print("\n✓ Logging test complete")
    print(f"✓ Check logs in: {LOG_DIR}/")
    print(f"  - Main log: {LOG_FILE}")
    print(f"  - Latest log: latest.log")
    print(f"  - Error log: errors.log")
