import torch
import torch.nn.functional as F

from core.retrieval_augmented.retrieval_methods.retriever_base import RetrieverBase


class CosineSimilarityRetriever(RetrieverBase):
    def __init__(self):
        super().__init__(retrieval_method_name="cosine", largest=True)

    def retrieve(
        self, query: torch.Tensor, db: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the top-k items from the database based on cosine similarity with the query.
        Args:
            query: A tensor representing the query, shape (batch_size, item_dim).
            db: A tensor representing the database items, shape (num_items, item_dim).
            k: The number of top items to retrieve.
        Returns:
            A tuple containing:
                - retrieved_items: Tensor of shape (batch_size, k, item_dim) with the top-k retrieved items.
                - scores: Tensor of shape (batch_size, k) with the cosine similarity scores for each retrieved item.
        """

        db_norm = F.normalize(db, p=2, dim=1)
        query_norm = F.normalize(query, p=2, dim=1)

        sim = torch.matmul(query_norm, db_norm.T)

        scores, indices = torch.topk(sim, k=k, dim=1, largest=self.largest)

        return db[indices], scores
