import argparse
import logging

from core.evaluation.metrics.accuracy_metric import AccuracyMetric
from core.evaluation.metrics.auc_metric import AUCMetric
from core.evaluation.metrics.f1_score_metric import F1ScoreMetric
from core.evaluation.metrics.mcc_metric import MCCMetric
from core.evaluation.metrics.precision_metric import PrecisionMetric
from core.evaluation.metrics.recall_metric import RecallMetric
from runners import apca_main, jitdp_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ],
)

logger = logging.getLogger(__name__)


DEVICE = "cuda"
PT_MODEL = "Salesforce/codet5p-770m"
EMBEDDING_DIM = 1024  # 2x for apca, 3x for jitdp

DATASET = "apca/small"

MAX_EPOCHS = 200
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 5
EARRLY_STOPPING_DELTA = 0.00001


TOP_K = 5
MODEL = "simple"  # available: "simple", "ra_average", "ra_weighted", "ra_attention"
RETRIEVER = "cosine"  # available: "cosine", "euclidean", "manhattan"

METRICS = [
    AccuracyMetric(),
    PrecisionMetric(),
    RecallMetric(),
    F1ScoreMetric(),
    MCCMetric(),
    AUCMetric(),
]


def main(
    dataset: str,
    model: str,
    retriever: str,
    top_k: int,
    max_epochs: int,
    batch_size: int,
    early_stopping_patience: int,
    early_stopping_delta: float,
    pt_model: str,
    embedding_dim: int,
    device: str,
):
    if dataset.startswith("apca"):
        apca_main(
            dataset=dataset,
            model=model,
            retriever=retriever,
            top_k=top_k,
            max_epochs=max_epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            early_stopping_delta=early_stopping_delta,
            pt_model=pt_model,
            embedding_dim=embedding_dim * 2,
            device=device,
            metrics=METRICS,
        )
    elif dataset.startswith("jitdp"):
        jitdp_main(
            dataset=dataset,
            model=model,
            retriever=retriever,
            top_k=top_k,
            max_epochs=max_epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            early_stopping_delta=early_stopping_delta,
            pt_model=pt_model,
            embedding_dim=embedding_dim * 3,
            device=device,
            metrics=METRICS,
        )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Run the model with specified parameters."
    )
    arg_parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=DATASET,
        help="Dataset to use.",
        choices=["apca/small", "jitdp/go", "jitdp/jdt", "jitdp/gerrit"],
    )
    arg_parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=MODEL,
        help="Model to use.",
        choices=["simple", "ra_average", "ra_weighted", "ra_attention"],
    )
    arg_parser.add_argument(
        "-r",
        "--retriever",
        type=str,
        default=RETRIEVER,
        help="Retriever to use.",
        choices=["cosine", "euclidean", "manhattan"],
    )
    arg_parser.add_argument(
        "-k", "--top_k", type=int, default=TOP_K, help="Top K for retrieval."
    )
    arg_parser.add_argument(
        "-e",
        "--max_epochs",
        type=int,
        default=MAX_EPOCHS,
        help="Maximum number of epochs.",
    )
    arg_parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training.",
    )
    arg_parser.add_argument(
        "-p",
        "--early_stopping_patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Early stopping patience.",
    )
    arg_parser.add_argument(
        "-l",
        "--early_stopping_delta",
        type=float,
        default=EARRLY_STOPPING_DELTA,
        help="Early stopping delta.",
    )
    arg_parser.add_argument(
        "-t",
        "--pt_model",
        type=str,
        default=PT_MODEL,
        help="Pretrained model to use for encoding.",
    )
    arg_parser.add_argument(
        "-i",
        "--embedding_dim",
        type=int,
        default=EMBEDDING_DIM,
        help="Embedding dimension.",
    )
    arg_parser.add_argument(
        "-v",
        "--device",
        type=str,
        default=DEVICE,
        help="Device to use for training (e.g., 'cuda' or 'cpu').",
    )

    args = arg_parser.parse_args()

    main(
        dataset=args.dataset,
        model=args.model,
        retriever=args.retriever,
        top_k=args.top_k,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_delta=args.early_stopping_delta,
        pt_model=args.pt_model,
        embedding_dim=args.embedding_dim,
        device=args.device,
    )
