from abc import ABC, abstractmethod
from typing import Literal

import torch


class RetrieverBase(ABC):
    def __init__(self, retrieval_method_name: str, largest: bool):
        self.retrieval_method = retrieval_method_name
        self.largest = largest

    @abstractmethod
    def retrieve(
        self, query: torch.Tensor, db: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Abstract method to retrieve the top-k items from the database based on the query.
        Args:
            query: A tensor representing the query, shape (batch_size, item_dim).
            db: A tensor representing the database items, shape (num_items, item_dim).
            k: The number of top items to retrieve.
        Returns:
            A tuple containing:
                - retrieved_items: Tensor of shape (batch_size, k, item_dim) with the top-k retrieved items.
                - relevance_scores: Tensor of shape (batch_size, k, 1) with the relevance scores for each retrieved item.
        """
        pass

    def __call__(
        self, query: torch.Tensor, db: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor, Literal["cosine", "euclidean", "manhattan"]]:
        """
        Method to retrieve the top-k items from the database based on the query.
        Args:
            query: A tensor representing the query, shape (batch_size, item_dim).
            db: A tensor representing the database items, shape (num_items, item_dim).
            k: The number of top items to retrieve.
        Returns:
            A tuple containing:
                - retrieved_items: Tensor of shape (batch_size, k, item_dim) with the top-k retrieved items.
                - relevance_scores: Tensor of shape (batch_size, k, 1) with the relevance scores for each retrieved item.
                - retrieval_method: A string indicating the retrieval method used.
        """
        retrieved_items, relevance_scores = self.retrieve(query, db, k + 1)

        items, scores = self._remove_query_from_retrieved_if_exists(
            query, retrieved_items, relevance_scores, k
        )

        return items, scores, self.retrieval_method

    def _remove_query_from_retrieved_if_exists(
        self,
        query: torch.Tensor,
        retrieved_items: torch.Tensor,
        scores: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Remove the query from the retrieved items if it exists.
        """

        query_expanded = query.unsqueeze(1)
        matches = torch.all(retrieved_items == query_expanded, dim=2)

        scores = scores.masked_fill(matches, float("-inf" if self.largest else "inf"))
        sorted_scores, sorted_indices = torch.topk(
            scores, k=k, dim=1, largest=self.largest
        )

        items = torch.gather(
            retrieved_items,
            1,
            sorted_indices.unsqueeze(-1).expand(-1, -1, retrieved_items.size(-1)),
        )

        return items, sorted_scores
