import argparse
import os.path as osp
import optuna
import wandb
from optuna_integration.wandb import WeightsAndBiasesCallback
from types import SimpleNamespace
from config import *
import joblib
import torch
import torch.optim as optim
from torch_geometric.datasets import FB15k_237
from torch_geometric.nn import TransE
from torch_geometric.nn.kge import KGEModel
import math
import torch.nn.functional as F
from torch import Tensor
import os
import numpy as np
import random
import sys
sys.path.append("../")
from config import *
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"


wandb.login(key=WANDB_API_KEY)


wandb_kwargs = {"project": WANDB_PROJECT, "group": "PyG_tranE"}
wandbc = WeightsAndBiasesCallback(
    metric_name="mrr", wandb_kwargs=wandb_kwargs, as_multirun=True
)


@wandbc.track_in_wandb()
def train_main_optuna(trial: optuna.Trial):

    path = os.path.join(".", "data", "FB15k")
    loss_type = trial.suggest_categorical(
        "loss_type", ["CrossEntropyLoss", "BCEWithLogitsLoss", "MarginRankingLoss"]
    )
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "Adagrad"])
    batch_size = trial.suggest_categorical(
        "batch_size", [int(128), int(256), int(512), int(1024)]
    )
    embedding_dim = trial.suggest_categorical(
        "embedding_dim", [int(50), int(128), int(256), int(512), int(1024)]
    )
    lr = trial.suggest_float("lr", 1e-4, 1e-0, log=True)
    weights = trial.suggest_float("weights", 1e-4, 1e-1, log=True)
    margin = trial.suggest_float("margin", 1e-4, 1e-1, log=True)

    train_data = FB15k_237(path, split="train")[0].to(device)
    val_data = FB15k_237(path, split="val")[0].to(device)
    test_data = FB15k_237(path, split="test")[0].to(device)
    model = TransE(
        num_nodes=train_data.num_nodes,
        num_relations=train_data.num_edge_types,
        hidden_channels=embedding_dim,
    ).to(device)

    loader = model.loader(
        head_index=train_data.edge_index[0],
        rel_type=train_data.edge_type,
        tail_index=train_data.edge_index[1],
        batch_size=batch_size,
        shuffle=True,
    )
    if optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adagrad(model.parameters(), lr=lr)

    if loss_type == "CrossEntropyLoss":
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_type == "BCEWithLogitsLoss":
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.MarginRankingLoss(margin=margin)

    def train():
        model.train()
        total_loss = total_examples = 0
        for head_index, rel_type, tail_index in loader:
            optimizer.zero_grad()
            loss = model.loss(head_index, rel_type, tail_index)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * head_index.numel()
            total_examples += head_index.numel()
        return total_loss / total_examples

    @torch.no_grad()
    def test(data):
        model.eval()
        return model.test(
            head_index=data.edge_index[0],
            rel_type=data.edge_type,
            tail_index=data.edge_index[1],
            batch_size=batch_size,
            k=10,
        )

    best_mrr = 0
    patience = 10
    patience_counter = 0

    import time

    start_training = time.time()
    for epoch in range(1, 1000):
        loss = train()
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
        if epoch % 5 == 0:
            rank, mrr, hits = test(val_data)
            print(
                f"Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}"
            )

            # Early stopping condition
            if mrr > best_mrr:
                best_mrr = mrr
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered. No improvement in MRR.")
                    break
    end_training = time.time()
    start_test = time.time()
    rank, mrr, hits_at_10 = test(test_data)
    print(
        f"Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, "
        f"Test Hits@10: {hits_at_10:.4f}"
    )
    end_test = time.time()
    print("training time", end_training - start_training)
    print("testing time", end_test - start_test)
    values_to_log = dict()
    values_to_log["test_mrr"] = mrr
    values_to_log["test_hits@10"] = hits_at_10
    values_to_log["test_mean_rank"] = rank
    values_to_log["training_time"] = end_training - start_training
    values_to_log["testing_time"] = end_test - start_test

    wandb.log(values_to_log)

    return mrr


if __name__ == "__main__":
    # wandb.agent(sweep_id, function=train_main, count=NUM_TRIALS)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name=f"{MODEL_NAME}_HPO", direction="maximize", sampler=sampler
    )
    study.optimize(
        train_main_optuna, n_trials=NUM_TRIALS, catch=(ValueError,), callbacks=[wandbc]
    )
    print(study.best_params)
    print(study.trials)
    joblib.dump(study, f"{MODEL_NAME}_STUDY")
