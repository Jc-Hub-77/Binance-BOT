import sys
from pathlib import Path
from loguru import logger
from trading_bot.config.config_loader import LoggingConfig # Assuming this path is correct

# Store the default logger configuration to potentially reset or inspect
_default_logger_config = {
    "handlers": []
}

def setup_logger(config: LoggingConfig):
    """
    Configure Loguru logger based on the provided LoggingConfig.
    Removes existing handlers and adds new ones for console and file.
    """
    global _default_logger_config
    _default_logger_config['handlers'] = logger._core.handlers.copy() # type: ignore

    logger.remove()  # Remove all existing handlers, including the default one

    # Console Handler
    # A more readable format for console
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=console_format,
        level=config.level.upper(), # Ensure level is uppercase
        colorize=True,
        enqueue=True # Make logging calls non-blocking
    )

    # File Handler
    try:
        log_file_path = Path(config.file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure log directory exists

        # A more detailed format for file logs
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS Z} | {level: <8} | "
            "{process.id} | {thread.name: <15} | " # Process and Thread info
            "{name}.{function}:{line} - {message}"
        )
        logger.add(
            log_file_path,
            format=file_format,
            level=config.level.upper(),
            rotation=config.rotation,
            retention=config.retention,
            compression="zip", # Compressing old log files
            enqueue=True, # Make logging calls non-blocking
            # encoding="utf-8" # Explicitly set encoding if needed
        )

        # Optional: Add a separate error log file
        error_log_path = log_file_path.parent / f"{log_file_path.stem}_errors.log"
        logger.add(
            error_log_path,
            format=file_format, # Or a specific error format
            level="ERROR", # Only log ERROR and CRITICAL messages
            rotation="1 week", # Or as per config if extended
            retention="1 month", # Or as per config if extended
            compression="zip",
            enqueue=True
        )
        logger.info(f"Logging initialized. Level: {config.level}. File: {log_file_path}. Error File: {error_log_path}")

    except Exception as e:
        # Fallback to console if file logging setup fails
        logger.remove() # Clean up any partially added handlers
        logger.add(sys.stderr, level="WARNING", colorize=True)
        logger.error(f"Failed to set up file logger at {config.file}. Reason: {e}. Falling back to stderr.")

    return logger # Return the configured logger instance

if __name__ == '__main__':
    # Example Usage:
    # Create a dummy LoggingConfig for testing
    class DummyLoggingConfig(LoggingConfig):
        level: str = "DEBUG"
        file: str = "logs/test_bot.log"
        rotation: str = "100 KB"
        retention: str = "2 days"

    test_config = DummyLoggingConfig()

    # Setup the logger
    setup_logger(test_config)

    # Test log messages
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.success("This is a success message (custom level if configured or Loguru default).")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("An exception occurred:")

    print(f"\nLog files should be in: {Path(test_config.file).parent.resolve()}")
    print(f"Default logger handlers before setup: {_default_logger_config['handlers']}")
    print(f"Current logger handlers: {logger._core.handlers}") # type: ignore

    # Clean up dummy log files if necessary
    # log_dir = Path("logs")
    # if log_dir.exists():
    #     for f in log_dir.glob("test_bot*"):
    #         f.unlink()
    #     if not any(log_dir.iterdir()): # if directory is empty
    #         log_dir.rmdir()
