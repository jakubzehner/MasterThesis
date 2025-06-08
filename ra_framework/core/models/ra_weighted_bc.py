import logging
from typing import Literal

import torch
from torch import nn

from core.models.binary_classifier_base import BinaryClassifierBase

logger = logging.getLogger(__name__)


class RaWeightedClassifier(BinaryClassifierBase):
    def __init__(self, input_dim: int):
        super(RaWeightedClassifier, self).__init__(classifier_dim=input_dim * 2)

        self.input_norm = nn.LayerNorm(input_dim)
        self.combined_norm = nn.LayerNorm(input_dim * 2)

    def model_layers(
        self,
        inputs: list[
            torch.Tensor
            | tuple[
                torch.Tensor, torch.Tensor, Literal["cosine", "euclidean", "manhattan"]
            ]
        ],
    ) -> torch.Tensor:
        """
        Define the model layers for the simple binary classifier.
        Args:
            inputs: List of input tensors, expected to contain two tensors:
                - inputs[0]: Tensor of shape (batch_size, input_dim) representing the model input
                - inputs[1]: tuple containing:
                    - Tensor of shape (batch_size, k, input_dim) representing the top-k retrieved items
                    - Tensor of shape (batch_size, k, 1) representing the relevance scores
                    - Literal of "cosine", "euclidean" or "manhattan" representing the retrieval method
        Returns:
            classifer_input: Tensor of shape (batch_size, classifier_dim) that is fed into the classifier
        """
        assert len(inputs) == 2, "Expected two input tensors for RaAverageClassifier"
        assert isinstance(inputs[0], torch.Tensor), "First input must be a tensor"
        assert isinstance(inputs[1], tuple) and len(inputs[1]) == 3, (
            "Second input must be a tuple of (retrieved_items, relevance_scores, retrieval_method)"
        )

        retrieved_items, scores, method = inputs[1]

        normalized_input = self.input_norm(inputs[0])
        averaged_retrieved = self._calculate_weighted_average(
            retrieved_items, scores, method
        )

        combined_inputs = torch.cat((normalized_input, averaged_retrieved), dim=1)
        normalized_combined = self.combined_norm(combined_inputs)

        return normalized_combined

    def _calculate_weighted_average(
        self,
        retrieved_items: torch.Tensor,
        scores: torch.Tensor,
        method: Literal["cosine", "euclidean", "manhattan"],
    ) -> torch.Tensor:
        """
        Calculate the weighted average of retrieved items based on relevance scores.
        Args:
            retrieved_items: Tensor of shape (batch_size, k, input_dim) representing the top-k retrieved items
            scores: Tensor of shape (batch_size, k, 1) representing the relevance scores
            method: str representing the retrieval method
        Returns:
            averaged_retrieved: Tensor of shape (batch_size, input_dim) representing the weighted average
        """
        EPS = 1e-8  # Small value to avoid division by zero

        if torch.isnan(scores).any() or torch.isinf(scores).any():
            logger.warning(
                "NaN or Inf values found in scores. Replacing with zeros for `cosine` method or large values for others."
            )
            scores = torch.nan_to_num(scores, nan=0.0, posinf=100000.0, neginf=0.0)

        weights = None
        match method:
            case "cosine":
                weights = torch.softmax(scores, dim=1)
            case "euclidean" | "manhattan":
                inv_distances = 1.0 / (scores + EPS)
                sum_inv_distances = inv_distances.sum(dim=1, keepdim=True)
                sum_inv_distances = torch.clamp(sum_inv_distances, min=EPS)
                weights = inv_distances / sum_inv_distances
            case _:
                error_msg = f"Unsupported retrieval method: {method}. Supported methods are 'cosine', 'euclidean', and 'manhattan'."
                logger.error(error_msg)
                raise ValueError(error_msg)

        weights = weights.unsqueeze(-1)
        averaged_retrieved = (retrieved_items * weights).sum(dim=1)
        return averaged_retrieved
