import numpy
from sklearn.metrics import matthews_corrcoef

from core.evaluation.metrics.metric_base import MetricBase


class MCCMetric(MetricBase):
    def __init__(self):
        super().__init__("MCC")

    def _compute_metric(
        self,
        true_labels: numpy.ndarray,
        predicted_labels: numpy.ndarray,
        labels_probs: numpy.ndarray,
    ) -> float:
        return matthews_corrcoef(true_labels, predicted_labels)
