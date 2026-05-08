import logging


def setup_logger(
    name: str = __name__,
    level: int = logging.DEBUG
) -> logging.Logger:
    """Setup-ol egy loggert az folyamat nyomonkövetéséhez

    Parameters:
        name: Logger neve, általában (__name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Duplikáció elkerülése miatt
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        datefmt="%Y/%m/%d - %H:%M:%S"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
