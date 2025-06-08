from abc import ABC, abstractmethod

from readers.code_change_model import CodeChangeModel


class GenericReader(ABC):
    @abstractmethod
    def read(self, path: str) -> list[CodeChangeModel]:
        """
        Read data from the specified paths and return a list of CodeChangeModel instances.

        Args:
            path: File paths to read data from.

        Returns:
            A list of CodeChangeModel instances read from the specified paths.
        """
        pass

    def _list_to_text(self, text_list: list[str], separator: str = " ") -> str:
        """
        Convert a list of strings to a single string with each element separated by a separator.

        Args:
            text_list: A list of strings to be converted.
            separator: The separator used to join the strings.

        Returns:
            A single string with each element from the list (stripped) separated by the separator.
        """
        return separator.join(s.strip() for s in text_list)
