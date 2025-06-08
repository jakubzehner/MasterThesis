import logging

from readers.code_change_model import CodeChangeModel
from readers.generic_reader import GenericReader
from utils.pickle_reader import load_pickle

logger = logging.getLogger(__name__)


class APCAReader(GenericReader):
    def read(self, path: str) -> list[CodeChangeModel]:
        """
        Read data from the specified path and return a list of CodeChangeModel instances.
        Args:
            path: File path to read data from.
        Returns:
            A list of CodeChangeModel instances read from the specified path.
        """
        loaded_data = load_pickle(path)
        if loaded_data is None:
            logger.error(f"Failed to load data from {path}.")
            return []

        logger.info("Processing loaded apca data...")
        result = []
        for entry in loaded_data:
            result.append(self._process_entry(entry))

        logger.info(f"Processed {len(result)} entries from {path}.")
        return result

    def _process_entry(self, entry: dict) -> CodeChangeModel:
        added_code = []
        removed_code = []

        for diff in entry["diff"]:
            added_code.append(self._list_to_text(diff["added_code"]))
            removed_code.append(self._list_to_text(diff["removed_code"]))

        joined_added_code = self._list_to_text(added_code)
        joined_removed_code = self._list_to_text(removed_code)

        return CodeChangeModel(
            added_code=joined_added_code,
            removed_code=joined_removed_code,
            code=joined_added_code + " " + joined_removed_code,
            label=entry["label"],
        )
