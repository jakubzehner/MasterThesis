import logging
import pickle
from typing import Any

logger = logging.getLogger(__name__)


def load_pickle(file_path: str) -> Any | None:
    """
    Load an object from a pickle file.
    Args:
        file_path: The file path from which the object will be loaded.
    Returns:
        The loaded object, or None if loading fails.
    """
    loaded_data = None

    logger.info(f"Loading data from {file_path}...")
    try:
        with open(file_path, "rb") as f:
            loaded_data = pickle.load(f)
        logger.info(f"Data loaded successfully. Number of entries: {len(loaded_data)}")
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}. Error: {e}")
    finally:
        return loaded_data
