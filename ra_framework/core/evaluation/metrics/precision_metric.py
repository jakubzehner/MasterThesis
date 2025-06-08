import numpy
from sklearn.metrics import precision_score

from core.evaluation.metrics.metric_base import MetricBase


class PrecisionMetric(MetricBase):
    def __init__(self):
        super().__init__("Prec.")

    def _compute_metric(
        self,
        true_labels: numpy.ndarray,
        predicted_labels: numpy.ndarray,
        labels_probs: numpy.ndarray,
    ) -> float:
        return precision_score(true_labels, predicted_labels)
