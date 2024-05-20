from torch import cuda
from torch.optim import Adam, Adagrad, SGD
import torchkge
from torchkge.models.bilinear import ComplExModel
from torchkge.models import TransEModel, TransRModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.models.bilinear import DistMultModel, AnalogyModel
from torchkge.models.deep import ConvKBModel
from torchkge.utils import MarginLoss, DataLoader, BinaryCrossEntropyLoss
from tqdm import tqdm
import pandas as pd
from torchkge.utils.datasets import load_fb15k237
import wandb
from config import *
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import joblib
import argparse
import os.path as osp
from types import SimpleNamespace
import sys
sys.path.append("../")
from config import *

wandb.login(key=WANDB_API_KEY)

wandb_kwargs = {"project": WANDB_PROJECT, "group": MODEL_NAME}
wandbc = WeightsAndBiasesCallback(
    metric_name="mrr", wandb_kwargs=wandb_kwargs, as_multirun=True
)


def train_one_epoch(epoch, dataloader, optimizer, sampler, criterion, model, iterator):
    runningLoss = 0.0
    for batch in dataloader:
        head, tail, relation = batch[0], batch[1], batch[2]
        numHead, numTail = sampler.corrupt_batch(head, tail, relation)
        optimizer.zero_grad()
        pos, neg = model(head, tail, relation, numHead, numTail)
        loss = criterion(pos, neg)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()
    iterator.set_description(
        "Epoch %d, loss %.5f" % (epoch, runningLoss / len(dataloader))
    )
    print("Epoch %d, loss %.5f" % (epoch, runningLoss / len(dataloader)))
    return {"train_loss": runningLoss / len(dataloader)}


def test_loop(model, kg_test, epoch, batchSize):
    print("Epoch: ", epoch)
    print(f"Performing Test after epoch {epoch}")
    eval = torchkge.evaluation.LinkPredictionEvaluator(model, kg_test)
    eval.evaluate(b_size=batchSize)
    hits_at_10 = eval.hit_at_k()
    mrr = eval.mrr()
    mr = eval.mean_rank()

    print(f"Epoch: {epoch} --- Test Mean Rank (MR): {mr}")
    print(f"Epoch: {epoch} --- Test Mean Reciprocal Rank (MRR): {mrr}")
    print(f"Epoch: {epoch} --- Test Hits@10: {hits_at_10}")

    return {"hits_10": hits_at_10[1], "mrr": mrr[1], "mr": mr[1]}


@wandbc.track_in_wandb()
def train_main_optuna(trial: optuna.Trial):
    kg_train, kg_val, kg_test = load_fb15k237()
    embeddingDimension = trial.suggest_categorical(
        "embeddingDimension", [int(64), int(128), int(256), int(512), int(1024)]
    )
    batchSize = trial.suggest_categorical(
        "batchSize", [int(128), int(256), int(512), int(1024)]
    )
    optimizer = trial.suggest_categorical("optimizer", ["adam", "adagrad", "sgd"])
    margin = trial.suggest_categorical("margin", [0.3, 0.5, 0.7])
    dissimilarity = trial.suggest_categorical("dissimilarity", ["L1", "L2"])
    loss_type = trial.suggest_categorical("loss_type", ["bce", "margin_loss"])
    learningRate = trial.suggest_float("learningRate", 1e-4, 1e-0, log=True)

    model = TransEModel(
        embeddingDimension,
        kg_train.n_ent,
        kg_train.n_rel,
        dissimilarity_type=dissimilarity,
    )
    # elif MODEL_NAME == "transR":
    #     model = TransRModel(
    #         entEmbDimension, relEmbDimension, kg_train.n_ent, kg_train.n_rel
    #     )
    # elif MODEL_NAME == "complex":
    #     model = ComplExModel(
    #         emb_dim=embeddingDimension,
    #         n_entities=kg_train.n_ent,
    #         n_relations=kg_train.n_rel,
    #     )
    # elif MODEL_NAME == "distmult":
    #     model = DistMultModel(
    #         emb_dim=embeddingDimension,
    #         n_entities=kg_train.n_ent,
    #         n_relations=kg_train.n_rel,
    #     )
    # elif MODEL_NAME == "analogy":
    #     scaler_s = trial.suggest_categorical("scaler_share", [0.3, 0.5, 0.7])
    #     model = AnalogyModel(
    #         emb_dim=embeddingDimension,
    #         n_entities=kg_train.n_ent,
    #         n_relations=kg_train.n_rel,
    #         scalar_share=scaler_s,
    #     )

    # elif MODEL_NAME == "conve":

    #     n_filt = trial.suggest_categorical("n_filters", [1, 3, 5, 7])
    #     model = ConvKBModel(
    #         emb_dim=embeddingDimension,
    #         n_filters=n_filt,
    #         n_entities=kg_train.n_ent,
    #         n_relations=kg_train.n_rel,
    #     )
    if loss_type == "bce":
        criterion = BinaryCrossEntropyLoss()
    elif loss_type == "margin_loss":
        criterion = MarginLoss(margin)
    if cuda.is_available():
        cuda.empty_cache()
        model.cuda()
        criterion.cuda()
    if optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=learningRate, weight_decay=1e-5)
    elif optimizer == "adagrad":
        optimizer = Adagrad(model.parameters(), lr=learningRate, weight_decay=1e-5)
    elif optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=learningRate, weight_decay=1e-5)

    sampler = BernoulliNegativeSampler(kg_train)
    dataloader = DataLoader(kg_train, batch_size=batchSize, use_cuda="all")
    iterator = tqdm(range(400), unit="epoch")

    for epoch in range(1, 400 + 1):
        train_loss_dict = train_one_epoch(
            epoch, dataloader, optimizer, sampler, criterion, model, iterator
        )
        wandb.log(train_loss_dict)
        if epoch % 400 == 0:
            test_dict = test_loop(model, kg_test, epoch, batchSize)
            wandb.log(test_dict)
            return test_dict["mrr"]


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
