import numpy as np
from ampligraph.datasets import load_fb15k_237
from ampligraph.latent_features import ScoringBasedEmbeddingModel
from ampligraph.evaluation import mrr_score, hits_at_n_score
from ampligraph.latent_features.loss_functions import get as get_loss
from ampligraph.latent_features.regularizers import get as get_regularizer
import tensorflow as tf
import ampligraph
import numpy as np
from ampligraph.datasets import load_fb15k_237
import numpy as np
from ampligraph.datasets import load_fb15k_237
from ampligraph.latent_features import ScoringBasedEmbeddingModel
from ampligraph.evaluation import mrr_score, hits_at_n_score
from ampligraph.latent_features.loss_functions import get as get_loss
from ampligraph.latent_features.regularizers import get as get_regularizer
import tensorflow as tf
from torch.optim import Adam, Adagrad
import optuna
import wandb
from optuna_integration.wandb import WeightsAndBiasesCallback
import joblib

# load Wordnet18 dataset:

WANDB_PROJECT = "Ampligraph_sweeps"
WANDB_ENTITY = "tanmay-mandy"
MODEL_NAME = "TransE"
WANDB_API_KEY = "e339d692c6d0e5d0c84b303e0919755ad67635e8"
NUM_TRIALS = 30

wandb.login(key=WANDB_API_KEY)


wandb_kwargs = {"project": WANDB_PROJECT, "group": "ampligraph_"+MODEL_NAME}
wandbc = WeightsAndBiasesCallback(
    metric_name="mrr", wandb_kwargs=wandb_kwargs, as_multirun=True
)

# load Wordnet18 dataset:
@wandbc.track_in_wandb()
def train_main_optuna(trial: optuna.Trial):

    X = load_fb15k_237()

    # Initialize a ComplEx neural embedding model: the embedding size is k,
    # eta specifies the number of corruptions to generate per each positive,
    # scoring_type determines the scoring function of the embedding model.
    model = ScoringBasedEmbeddingModel(k=150,
                                    eta=10,
                                    scoring_type=MODEL_NAME)

    # Optimizer, loss and regularizer definition
    loss_type = trial.suggest_categorical(
        "loss_type", ["pairwise", "nll", "self_adversarial"]
    )
    batch_size = trial.suggest_categorical(
        "batch_size", [int(128), int(256), int(512), int(1024)]
    )
    optimizer = trial.suggest_categorical("optimizer", ["adam","adagrad"])

    learningRate = trial.suggest_categorical("learningRate",[4e-3,4e-5,4e-4])
    batch_size = trial.suggest_categorical(
        "batch_size", [int(128), int(256), int(512), int(1024)]
    )


    
    if optimizer == "adam":
        optim = tf.keras.optimizers.Adam(learning_rate=learningRate)
    else:
        optim = tf.keras.optimizers.Adagrad(learning_rate=learningRate)
    loss = get_loss(loss_type, {'margin': 0.5})
    regularizer = get_regularizer('LP', {'p': 2, 'lambda': 1e-5})

    # Compilation of the model
    model.compile(optimizer=optim, loss=loss, entity_relation_regularizer=regularizer)

    # For evaluation, we can use a filter which would be used to filter out
    # positives statements created by the corruption procedure.
    # Here we define the filter set by concatenating all the positives
    filter = {'test' : np.concatenate((X['train'], X['valid'], X['test']))}

    # Early Stopping callback
    checkpoint = tf.keras.callbacks.EarlyStopping(
        monitor='val_{}'.format('hits10'),
        min_delta=0,
        patience=5,
        verbose=1,
        mode='max',
        restore_best_weights=True
    )

    # Fit the model on training and validation set
    model.fit(X['train'],
            batch_size=batch_size,
            epochs=4,                    # Number of training epochs
            validation_freq=20,           # Epochs between successive validation
            validation_burn_in=100,       # Epoch to start validation
            validation_data=X['valid'],   # Validation data
            validation_filter=filter,     # Filter positives from validation corruptions
            callbacks=[checkpoint],       # Early stopping callback (more from tf.keras.callbacks are supported)
            verbose=True                  # Enable stdout messages
            )


    # Run the evaluation procedure on the test set (with filtering)
    # To disable filtering: use_filter=None
    # Usually, we corrupt subject and object sides separately and compute ranks
    ranks = model.evaluate(X['valid'],
                        use_filter=filter,
                        corrupt_side='s,o')

    # compute and print metrics:
    mrr = mrr_score(ranks)
    hits_10 = hits_at_n_score(ranks, n=10)
    print("MRR: %f, Hits@10: %f" % (mrr, hits_10))
    # Output: MRR: 0.884418, Hits@10: 0.935500

    values_to_log = dict()
    values_to_log["test_mrr"] = mrr
    values_to_log["test_hits_10"] = hits_at_n_score(ranks, n=10)
    wandb.log(values_to_log)
    # wandb.log(ranks)
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
    


    

