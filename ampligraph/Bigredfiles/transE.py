import optuna
import wandb
from optuna_integration.wandb import WeightsAndBiasesCallback
from types import SimpleNamespace
from config_transE import *
import joblib
from pykeen.regularizers import LpRegularizer
from pykeen.losses import CrossEntropyLoss, BCEWithLogitsLoss, MarginRankingLoss
from pykeen.evaluation import RankBasedEvaluator
from pykeen.datasets import FB15k237
from pykeen.pipeline import pipeline
from pykeen.models import TransE
from torch.optim import Adam, Adagrad
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
from pykeen.stoppers.early_stopping import EarlyStopper

############

wandb.login(key=WANDB_API_KEY)


wandb_kwargs = {"project": WANDB_PROJECT, "group": "pykeen_tranE"}
wandbc = WeightsAndBiasesCallback(
    metric_name="mrr", wandb_kwargs=wandb_kwargs, as_multirun=True
)

##############


##############
@wandbc.track_in_wandb()
def train_main_optuna(trial: optuna.Trial):

    ##################
    dataset = FB15k237()
    training, validation, testing = (
        dataset.training,
        dataset.validation,
        dataset.testing,
    )
    evaluator = RankBasedEvaluator()
    training_triples_factory = dataset.training
    ##################
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
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    num_negs_per_pos = trial.suggest_int("num_negs_per_pos", 1, 45, step=5)
    entity_initializer = trial.suggest_categorical(
        "entity_initializer",
        [
            "xavier_uniform_",
            "xavier_uniform_norm_",
            "xavier_normal_",
            "xavier_normal_norm_",
        ],
    )
    relation_initializer = trial.suggest_categorical(
        "relation_initializer",
        [
            "xavier_uniform_",
            "xavier_uniform_norm_",
            "xavier_normal_",
            "xavier_normal_norm_",
        ],
    )
    regularizer = trial.suggest_categorical("regularizer", [int(1), int(2)])

    model = TransE(
        triples_factory=training_triples_factory,
        embedding_dim=embedding_dim,
        scoring_fct_norm=1,
        entity_initializer=entity_initializer,
        relation_initializer=relation_initializer,
        regularizer=LpRegularizer(p=regularizer),
        loss=loss_type,
    ).to("cuda")

    if optimizer == "Adam":
        optimizer = Adam(params=model.get_grad_params())
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
        patience=5,
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
    values_to_log = dict()
    values_to_log["mrr"] = results.get_metric("mrr")
    values_to_log["hits@10"] = results.get_metric("hits@10")
    values_to_log["test_mrr"] = results_test.get_metric("mrr")
    values_to_log["test_hits@10"] = results_test.get_metric("hits@10")

    wandb.log(values_to_log)
    wandb.log(results.to_dict())
    return results.get_metric("mrr")


##############

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


###############
