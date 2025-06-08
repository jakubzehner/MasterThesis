import torch
from torch import nn

from core.models.binary_classifier_base import BinaryClassifierBase


class RaAverageClassifier(BinaryClassifierBase):
    def __init__(self, input_dim: int):
        super(RaAverageClassifier, self).__init__(classifier_dim=input_dim * 2)

        self.input_norm = nn.LayerNorm(input_dim)
        self.combined_norm = nn.LayerNorm(input_dim * 2)

    def model_layers(
        self, inputs: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor, str]]
    ) -> torch.Tensor:
        """
        Define the model layers for the simple binary classifier.
        Args:
            inputs: List of input tensors, expected to contain two tensors:
                - inputs[0]: Tensor of shape (batch_size, input_dim) representing the model input
                - inputs[1]: tuple containing:
                    - Tensor of shape (batch_size, k, input_dim) representing the top-k retrieved items
                    - Tensor of shape (batch_size, k, 1) representing the relevance scores (not used in this implementation)
                    - str representing the retrieval method (not used in this implementation)
        Returns:
            classifer_input: Tensor of shape (batch_size, classifier_dim) that is fed into the classifier
        """
        assert len(inputs) == 2, "Expected two input tensors for RaAverageClassifier"
        assert isinstance(inputs[0], torch.Tensor), "First input must be a tensor"
        assert isinstance(inputs[1], tuple) and len(inputs[1]) == 3, (
            "Second input must be a tuple of (retrieved_items, relevance_scores, retrieval_method)"
        )

        retrieved_items, _, _ = inputs[1]

        normalized_input = self.input_norm(inputs[0])
        averaged_retrieved = retrieved_items.mean(dim=1)

        combined_inputs = torch.cat((normalized_input, averaged_retrieved), dim=1)
        normalized_combined = self.combined_norm(combined_inputs)

        return normalized_combined
