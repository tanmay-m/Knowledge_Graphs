import torch
import torch.optim as optim
import os
from torch_geometric.datasets import FB15k_237
from torch_geometric.nn import TransE
from torch_geometric.nn.kge import KGEModel
import math
import torch.nn.functional as F
from torch import Tensor
import pandas as pd
import numpy as np
import random
import sys
sys.path.append("../")
from config import *
import yaml
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_main():

    path = os.path.join(".", "data", "FB15k")
    with open(r"../best_params.yaml") as stream:
        try:
            pyg_row = yaml.safe_load(stream)['pyGeometric']
        except yaml.YAMLError as exc:
            print(exc)

    loss_type = pyg_row["Loss"]
    optimizer = pyg_row["optimizer"]
    batch_size = pyg_row["batchSize"]
    embedding_dim = pyg_row["embeddingDimension"]
    lr = pyg_row["lr"]
    weights = pyg_row["weights"]
    margin = pyg_row["margin"]

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
    print("Test MRR:", mrr)
    print("Test Hits@10", hits_at_10)


train_main()
