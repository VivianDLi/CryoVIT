import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    logger = logging.getLogger()
    logger.setLevel(level)

    if not logger.hasHandlers():
        rich_handler = RichHandler(rich_tracebacks=True)
        rich_handler.setLevel(level)
        formatter = logging.Formatter("%(message)s", datefmt="[%X]")
        rich_handler.setFormatter(formatter)
        logger.addHandler(rich_handler)
