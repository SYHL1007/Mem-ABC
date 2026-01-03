import logging
import config
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import APITimeoutError, APIConnectionError, RateLimitError

# --- Logging Configuration ---
def setup_logging(quiet: bool = False):
    """
    Configure global logging.
    
    Args:
        quiet: If True, console output is restricted to WARNING/ERROR; 
               file output always records DEBUG and above.
    """
    # File handler: logs everything
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    
    # Console handler: adjusts based on quiet mode
    console_handler = logging.StreamHandler()
    if quiet:
        console_handler.setLevel(logging.WARNING)
    else:
        console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    
    # Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set root to capture everything
    root_logger.handlers = []            # Clear existing handlers to prevent duplication
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress httpx logging noise
    logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Tenacity Async Retry Decorator ---
# Retry strategy for transient LLM API errors
retry_async_llm_call = retry(
    wait=wait_exponential(multiplier=1, min=2, max=60), # Exponential backoff
    stop=stop_after_attempt(5),                         # Max 5 attempts
    retry=retry_if_exception_type((APITimeoutError, APIConnectionError, RateLimitError)),
    reraise=True
)