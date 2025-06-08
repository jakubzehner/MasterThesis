import torch

from core.retrieval_augmented.retrieval_methods.retriever_base import RetrieverBase


class ManhattanRetriever(RetrieverBase):
    def __init__(self, batch_size: int = 16):
        super().__init__(retrieval_method_name="manhattan", largest=False)

        self.batch_size = batch_size

    def retrieve(
        self, query: torch.Tensor, db: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the top-k items from the database based on Manhattan distance with the query.
        Args:
            query: A tensor representing the query, shape (batch_size, item_dim).
            db: A tensor representing the database items, shape (num_items, item_dim).
            k: The number of top items to retrieve.
        Returns:
            A tuple containing:
                - retrieved_items: Tensor of shape (batch_size, k, item_dim) with the top-k retrieved items.
                - scores: Tensor of shape (batch_size, k) with the cosine similarity scores for each retrieved item.
        """
        real_batch_size = query.shape[0]

        all_scores = []
        all_indices = []

        for start in range(0, real_batch_size, self.batch_size):
            end = min(start + self.batch_size, real_batch_size)
            batch_query = query[start:end]

            db_expanded = db.unsqueeze(0)
            query_expanded = batch_query.unsqueeze(1)

            distances = torch.norm(query_expanded - db_expanded, p=1, dim=2)

            scores, indices = torch.topk(distances, k=k, dim=1, largest=self.largest)

            all_scores.append(scores)
            all_indices.append(indices)

        all_scores = torch.cat(all_scores, dim=0)
        all_indices = torch.cat(all_indices, dim=0)

        return db[all_indices], all_scores
