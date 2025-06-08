from abc import ABC, abstractmethod

import numpy


class MetricBase(ABC):
    def __init__(self, name: str):
        """
        Initialize the metric with a name.

        Args:
            name (str): The name of the metric.
        """
        self.name = name

    @abstractmethod
    def _compute_metric(
        self,
        true_labels: numpy.ndarray,
        predicted_labels: numpy.ndarray,
        labels_probs: numpy.ndarray,
    ) -> float:
        """
        Abstract method to compute the metric value.

        Args:
            true_labels (numpy.ndarray): The ground truth labels.
            predicted_labels (numpy.ndarray): The predicted labels.
            labels_probs (numpy.ndarray): The probabilities of the predicted labels.

        Returns:
            float: The computed metric value.
        """
        pass

    def compute(
        self, true_labels: numpy.ndarray, labels_probs: numpy.ndarray
    ) -> tuple[str, float]:
        """
        Compute the metric based on true labels and label probabilities.

        Args:
            true_labels (numpy.ndarray): The ground truth labels.
            labels_probs (numpy.ndarray): The probabilities of the predicted labels.

        Returns:
            tuple[str, float]: A tuple containing the metric name and its computed value.
        """
        value = self._compute_metric(
            true_labels, (labels_probs > 0.5).astype(int), labels_probs
        )
        return self.name, value
