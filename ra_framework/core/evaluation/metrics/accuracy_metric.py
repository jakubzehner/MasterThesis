import numpy
from sklearn.metrics import accuracy_score

from core.evaluation.metrics.metric_base import MetricBase


class AccuracyMetric(MetricBase):
    def __init__(self):
        super().__init__("Acc.")

    def _compute_metric(
        self,
        true_labels: numpy.ndarray,
        predicted_labels: numpy.ndarray,
        labels_probs: numpy.ndarray,
    ) -> float:
        return accuracy_score(true_labels, predicted_labels)
