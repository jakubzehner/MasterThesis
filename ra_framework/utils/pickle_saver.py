import logging
import os
import pickle
from typing import Any

logger = logging.getLogger(__name__)


def save_pickle(obj: Any, path: str):
    """Save an object to a pickle file.
    Args:
        obj: The object to be saved.
        path: The file path where the object will be saved.
    """

    logger.info(f"Saving data to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Data saved successfully to {path}.")
    except Exception as e:
        logger.error(f"Failed to save data to {path}. Error: {e}")
