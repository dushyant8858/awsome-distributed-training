"""Simple colored logging utility."""

import logging
import sys


class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    
    # Color mapping for different log levels
    LEVEL_COLORS = {
        logging.DEBUG: Colors.DIM + Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD + Colors.RED,
    }
    
    def __init__(self, fmt: str | None = None, datefmt: str | None = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        # Store original levelname
        orig_levelname = record.levelname
        
        if self.use_colors:
            # Add color to levelname
            color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
            record.levelname = f"{color}{record.levelname}{Colors.RESET}"
        
        # Format the message
        result = super().format(record)
        
        # Restore original levelname
        record.levelname = orig_levelname
        
        return result


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a logger with colored output.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create colored formatter
    formatter = ColoredFormatter(
        fmt="%(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    # Don't propagate to root logger
    logger.propagate = False
    
    return logger


# Convenience functions for direct use
def info(msg: str) -> None:
    """Log info message."""
    get_logger("k8s").info(msg)


def debug(msg: str) -> None:
    """Log debug message."""
    get_logger("k8s").debug(msg)


def warning(msg: str) -> None:
    """Log warning message."""
    get_logger("k8s").warning(msg)


def error(msg: str) -> None:
    """Log error message."""
    get_logger("k8s").error(msg)


def success(msg: str) -> None:
    """Print success message in green."""
    if sys.stdout.isatty():
        print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")
    else:
        print(f"✓ {msg}")


def failure(msg: str) -> None:
    """Print failure message in red."""
    if sys.stdout.isatty():
        print(f"{Colors.RED}✗ {msg}{Colors.RESET}")
    else:
        print(f"✗ {msg}")


if __name__ == "__main__":
    # Test the logger
    logger = get_logger("test")
    
    print("Testing logger levels:")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print("\nTesting convenience functions:")
    success("Operation completed successfully")
    failure("Operation failed")
    
    print("\nTesting colors directly:")
    print(f"{Colors.RED}Red{Colors.RESET} {Colors.GREEN}Green{Colors.RESET} {Colors.YELLOW}Yellow{Colors.RESET}")
