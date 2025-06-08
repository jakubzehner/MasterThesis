from dataclasses import dataclass
from typing import Optional


@dataclass
class CodeChangeModel:
    """
    Represents a code change
    """

    added_code: str
    removed_code: str
    code: str
    label: int
    comment: Optional[str] = None

    def tokenized_data(self) -> dict:
        """
        Return data that can be used tokenized.

        Returns:
            A dictionary containing the code change data.
        """
        data = {
            "added_code": self.added_code,
            "removed_code": self.removed_code,
            "code": self.code,
        }
        if self.comment is not None:
            data["comment"] = self.comment

        return data

    def not_tokenized_data(self) -> dict:
        """
        Return data that can not be tokenized.

        Returns:
            A dictionary containing the non-tokenized code change data.
        """
        return {
            "label": self.label,
        }
