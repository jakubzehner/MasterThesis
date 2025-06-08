import torch

from core.models.binary_classifier_base import BinaryClassifierBase


class SimpleBinaryClassifier(BinaryClassifierBase):
    def __init__(self, input_dim: int):
        super(SimpleBinaryClassifier, self).__init__(classifier_dim=input_dim)

    def model_layers(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Define the model layers for the simple binary classifier.
        Args:
            inputs: List of input tensors, expected to contain a single tensor of shape (batch_size, input_dim)
        Returns:
            classifer_input: Tensor of shape (batch_size, classifier_dim) that is fed into the classifier
        """
        return inputs[0]
