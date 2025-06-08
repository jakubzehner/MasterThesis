import logging
from typing import Literal

import torch

from core.retrieval_augmented.retrieval_methods.retriever_base import RetrieverBase

logger = logging.getLogger(__name__)


class RetrievalAugmentedDB:
    def __init__(self, retrieval_method: RetrieverBase, k: int = 5):
        """
        Initialize the RetrievalAugmentedDB with a retrieval method and number of k top items to retrieve.
        Args:
            retrieval_method: An instance of a RetrieverBase subclass that defines the retrieval method.
            k: The number of top items to retrieve for each query.
        """

        self.retrieval_method = retrieval_method
        self.k = k
        self.db = None

    def set_db(self, db: torch.Tensor):
        """
        Set the database for the retrieval method.
        Args:
            db: A tensor representing the database items, shape (num_items, item_dim).
        """
        self.db = torch.unique(db, dim=0)

    def retrieve(
        self, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Literal["cosine", "euclidean", "manhattan"]]:
        """
        Retrieve the top-k items from the database based on the query.
        Args:
            query: A tensor representing the query, shape (batch_size, item_dim).
        Returns:
            A tuple containing:
                - retrieved_items: Tensor of shape (batch_size, k, item_dim) with the top-k retrieved items.
                - relevance_scores: Tensor of shape (batch_size, k, 1) with the relevance scores for each retrieved item.
                - retrieval_method: A string indicating the retrieval method used.
        """
        if self.db is None:
            error_msg = (
                "Database is not set. Please set the database using set_db() method."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        retrieved, scores, method = self.retrieval_method(query, self.db, k=self.k)
        return retrieved, scores, method
