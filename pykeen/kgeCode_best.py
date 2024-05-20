from pykeen.regularizers import LpRegularizer
from pykeen.losses import CrossEntropyLoss, BCEWithLogitsLoss, MarginRankingLoss
from pykeen.evaluation import RankBasedEvaluator
from pykeen.datasets import FB15k237
from pykeen.pipeline import pipeline
from pykeen.models import TransE
from torch.optim import Adam, Adagrad, SGD
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
from pykeen.stoppers.early_stopping import EarlyStopper
import pandas as pd
import sys
import yaml
sys.path.append("../")
from config import *

def train():
    dataset = FB15k237()
    training, validation, testing = (
        dataset.training,
        dataset.validation,
        dataset.testing,
    )
    evaluator = RankBasedEvaluator()
    # read the hyperparams file
    with open(r"../best_params.yaml") as stream:
        try:
            pykeen_row = yaml.safe_load(stream)['pykeen']
        except yaml.YAMLError as exc:
            print(exc)
            
    training_triples_factory = dataset.training
    loss_type = pykeen_row["Loss"]
    optimizer = pykeen_row["optimizer"]
    batch_size = pykeen_row["batchSize"]
    embedding_dim = pykeen_row["embeddingDimension"]
    lr = pykeen_row["lr"]
    weights = pykeen_row["weights"]
    num_negs_per_pos = pykeen_row["num_negs_per_pos"]
    entity_initializer = pykeen_row["entity_initializer"]
    relation_initializer = pykeen_row["relation_initializer"]
    regularizer = pykeen_row["regularizer"]

    model = TransE(
        triples_factory=training_triples_factory,
        embedding_dim=embedding_dim,
        scoring_fct_norm=1,
        entity_initializer=entity_initializer,
        relation_initializer=relation_initializer,
        regularizer=LpRegularizer(weight=weights, p=regularizer),
        loss=loss_type,
    ).to("cuda")

    if optimizer == "Adam":
        optimizer = Adam(params=model.get_grad_params())
    elif optimizer == "SGD":
        optimizer = SGD(params=model.get_grad_params())
    else:
        optimizer = Adagrad(params=model.get_grad_params())

    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=training_triples_factory,
        optimizer=optimizer,
        negative_sampler="basic",
        negative_sampler_kwargs=dict(num_negs_per_pos=num_negs_per_pos),
    )
    early_stopper = EarlyStopper(
        model=model,
        evaluator=evaluator,
        training_triples_factory=dataset.training,
        evaluation_triples_factory=dataset.validation,
        frequency=20,
        patience=10,
        relative_delta=0.01,
        metric="mrr",
    )

    _ = training_loop.train(
        triples_factory=training_triples_factory,
        num_epochs=400,
        batch_size=batch_size,
        stopper=early_stopper,
    )
    mapped_triples = dataset.validation.mapped_triples

    # Evaluate
    results = evaluator.evaluate(
        model=model,
        mapped_triples=mapped_triples,
        batch_size=batch_size,
        additional_filter_triples=[
            dataset.training.mapped_triples,
        ],
    )
    results_test = evaluator.evaluate(
        model=model,
        mapped_triples=dataset.testing.mapped_triples,
        batch_size=batch_size,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
    )

    print("Test MRR:", results_test.get_metric("mrr"))
    print("Test Hits@10", results_test.get_metric("hits@10"))


train()
