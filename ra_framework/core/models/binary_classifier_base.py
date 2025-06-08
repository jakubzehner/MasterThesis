from abc import ABC, abstractmethod

import torch
from torch import nn


class BinaryClassifierBase(nn.Module, ABC):
    def __init__(self, classifier_dim: int):
        super(BinaryClassifierBase, self).__init__()

        self.classifier_layer = nn.Linear(classifier_dim, 1)
        self.probabilities = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

    @abstractmethod
    def model_layers(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Abstract method to define the model layers.
        Args:
            inputs: List of input tensors
        Returns:
            classifer_input: Tensor of shape (batch_size, classifier_dim) that is fed into the classifier
        """
        pass

    def _calculate_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to probabilities using the sigmoid function.
        Args:
            logits: Tensor of shape (batch_size, 1)
        Returns:
            probabilities: Tensor of shape (batch_size, 1) with probabilities
        """
        return self.probabilities(logits)

    def _calculate_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the binary cross-entropy loss.
        Args:
            logits: Tensor of shape (batch_size, 1)
            targets: Tensor of shape (batch_size, 1) with binary labels (0 or 1)
        Returns:
            loss: Tensor representing the binary cross-entropy loss
        """
        return self.loss(logits.reshape(-1), targets)

    def forward(
        self, inputs: list[torch.Tensor], targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        Args:
            targets: Tensor of shape (batch_size, 1) with binary labels (0 or 1)
            inputs: List of input tensors
        Returns:
            probabilities: Tensor of shape (batch_size, 1) with predicted probabilities
            loss: Tensor representing the binary cross-entropy loss
        """
        model_output = self.model_layers(inputs)
        logits = self.classifier_layer(model_output)

        probabilities = self._calculate_probabilities(logits)
        loss = self._calculate_loss(logits, targets)
        return probabilities, loss
