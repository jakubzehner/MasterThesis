import logging
import os

import torch

from core.early_stopper import EarlyStopper
from core.evaluation.evaluator import Evaluator
from core.evaluation.metrics.metric_base import MetricBase
from core.retrieval_augmented.db import RetrievalAugmentedDB
from core.trainer import Trainer
from utils.model_run_utils import (
    calculate_average_metrics,
    get_model,
    get_retriever,
    get_x_and_y_tensors,
    pretty_print_metrics,
    save_metrics_to_file,
)
from utils.pickle_reader import load_pickle
from utils.plot_utils import save_loss_plot

logger = logging.getLogger(__name__)


def apca_main(
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
    metrics: list[MetricBase],
):
    logger.info("Starting cross-validation...")
    results_path = f"results/{pt_model}/{dataset}/{model}_{retriever}_k{top_k}/"
    os.makedirs(results_path, exist_ok=True)

    all_metrics = []
    for fold in range(5):
        logger.info(f"Processing fold {fold + 1}/5...")
        logger.info("Loading dataset...")

        train_data = load_pickle(
            f"data/embeddings/{pt_model}/{dataset}/cv{fold + 1}_train.pkl"
        )
        test_data = load_pickle(
            f"data/embeddings/{pt_model}/{dataset}/cv{fold + 1}_test.pkl"
        )

        x_train, y_train = get_x_and_y_tensors(train_data, device)
        x_test, y_test = get_x_and_y_tensors(test_data, device)

        logger.info("Initializing fold...")
        early_stopper = EarlyStopper(
            early_stopping_patience,
            early_stopping_delta,
        )
        model_instance = get_model(model, embedding_dim).to(device)
        retriever_instance = get_retriever(retriever)
        ra_db = None
        if model.startswith("ra"):
            ra_db = RetrievalAugmentedDB(retriever_instance, k=top_k)
            ra_db.set_db(x_train)

        optimizer = torch.optim.AdamW(model_instance.parameters(), lr=1e-5)
        trainer = Trainer(
            model_instance,
            optimizer,
            early_stopper,
            (x_train, y_train),
            (x_test, y_test),
            max_epochs,
            batch_size,
            device,
            ra_db,
        )

        logger.info("Starting training...")
        train_loss, val_loss = trainer.train()

        logger.info("Evaluating model...")
        evaluator = Evaluator(
            model_instance,
            (x_test, y_test),
            metrics,
            device,
            ra_db,
        )
        metrics_res = evaluator.evaluate()
        all_metrics.append(metrics_res)
        save_metrics_to_file(results_path + f"fold_{fold + 1}_metrics.txt", metrics_res)
        save_loss_plot(
            results_path + f"fold_{fold + 1}_loss",
            train_loss,
            val_loss,
        )

        pretty_print_metrics(metrics_res)

    avg_metrics = calculate_average_metrics(all_metrics)
    pretty_print_metrics(avg_metrics)
    save_metrics_to_file(results_path + "average_metrics.txt", avg_metrics)


def jitdp_main(
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
    metrics: list[MetricBase],
):
    logger.info("Starting cross-validation...")
    results_path = f"results/{pt_model}/{dataset}/{model}_{retriever}_k{top_k}/"
    os.makedirs(results_path, exist_ok=True)

    logger.info("Loading dataset...")
    train_data = load_pickle(f"data/embeddings/{pt_model}/{dataset}/train.pkl")
    test_data = load_pickle(f"data/embeddings/{pt_model}/{dataset}/test.pkl")

    x_train, y_train = get_x_and_y_tensors(train_data, device)
    x_test, y_test = get_x_and_y_tensors(test_data, device)

    logger.info("Initializing...")
    early_stopper = EarlyStopper(
        early_stopping_patience,
        early_stopping_delta,
    )
    model_instance = get_model(model, embedding_dim).to(device)
    retriever_instance = get_retriever(retriever)
    ra_db = None
    if model.startswith("ra"):
        ra_db = RetrievalAugmentedDB(retriever_instance, k=top_k)
        ra_db.set_db(x_train)

    optimizer = torch.optim.AdamW(model_instance.parameters(), lr=1e-5)
    trainer = Trainer(
        model_instance,
        optimizer,
        early_stopper,
        (x_train, y_train),
        (x_test, y_test),
        max_epochs,
        batch_size,
        device,
        ra_db,
    )

    logger.info("Starting training...")
    train_loss, val_loss = trainer.train()

    logger.info("Evaluating model...")
    evaluator = Evaluator(
        model_instance,
        (x_test, y_test),
        metrics,
        device,
        ra_db,
    )
    metrics_res = evaluator.evaluate()

    save_metrics_to_file(results_path + "metrics.txt", metrics_res)
    save_loss_plot(
        results_path + "loss",
        train_loss,
        val_loss,
    )

    pretty_print_metrics(metrics_res)
