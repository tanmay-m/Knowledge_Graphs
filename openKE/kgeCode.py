import optuna
import os, time
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import wandb
from optuna_integration.wandb import WeightsAndBiasesCallback
from types import SimpleNamespace
from config import *
import joblib
import sys
sys.path.append("../")
from config import *
wandb.login(key=WANDB_API_KEY)

wandb_kwargs = {"project": WANDB_PROJECT, "group": "openKE_" + MODEL_NAME}
wandbc = WeightsAndBiasesCallback(
    metric_name="mrr", wandb_kwargs=wandb_kwargs, as_multirun=True)

def get_validation_metrics(model, test_dataloader, use_gpu):
    tester = Tester(model=model, data_loader=test_dataloader, use_gpu=use_gpu)
    metrics = tester.run_link_prediction(type_constrain=False)
    return metrics

train_dataloader = TrainDataLoader(
    in_path = "./benchmarks/FB15K237/", 
    nbatches = 100,
    threads = 8, 
    sampling_mode = "normal", 
    bern_flag = 1, 
    filter_flag = 1, 
    neg_ent = 25,
    neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")


@wandbc.track_in_wandb()
def train_main_optuna(trial: optuna.Trial):
    embeddingDimension = trial.suggest_categorical("embeddingDimension",[int(64),int(128),int(256),int(512), int(1024)])
    batchSize = trial.suggest_categorical("batchSize",[int(128),int(256),int(512), int(1024)])
    optimizer = trial.suggest_categorical("optimizer",["adam","adagrad","adadelta","sgd"])
    margin = trial.suggest_categorical("margin",[0.3,0.5,0.7])
    alpha = trial.suggest_float("alpha", 1e-4, 1e-0, log=True)
    
    print("batchSize = 2721")
    print("optimizer = ", optimizer)
    print("margin = ", margin)
    print("alpha = ", alpha)
    # define the model
    transx = TransE(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = embeddingDimension, 
        p_norm = 1, 
        norm_flag = False
        )

    print("batch size = ",train_dataloader.get_batch_size())
    # define the loss function
    model = NegativeSampling(
        model = transx, 
        loss = MarginLoss(margin = margin),
#        batch_size = batchSize
       batch_size = train_dataloader.get_batch_size()
    )

    # Define the early stopping parameters
    patience = 5  # number of epochs to wait before stopping
    best_val_metric = None  # initialize with None
    epochs_without_improvement = 0

    # train the model
    # trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
    # trainer.run()
    # if not os.path.exists("./result"):
    #     os.makedirs("./result")

    # start time
#    start_time = time.time()

    for epoch in range(400):
        # Train for one epoch
        print(f"Epoch #: {epoch}")
        start_train_time = time.time()  # Record the start time for training
        trainer = Trainer(model=model, data_loader=train_dataloader, train_times=1, alpha=alpha, use_gpu=True, opt_method=optimizer)
        trainer.run()
        end_train_time = time.time()  # Record the end time for training
        train_time = end_train_time - start_train_time
        print(f"Training time for epoch {epoch}: {train_time:.2f} seconds")
        
        start_eval_time = time.time()  # Record the start time for evaluation
        val_metrics = get_validation_metrics(transx, test_dataloader, use_gpu=True)
        end_eval_time = time.time()  # Record the end time for evaluation
        eval_time = end_eval_time - start_eval_time
        print(f"Evaluation time for epoch {epoch}: {eval_time:.2f} seconds")
        
        # Update best validation metric and epochs_without_improvement
        if epoch > 0 and epoch % 5 == 0:

            if best_val_metric is None or abs(val_metrics[0] - best_val_metric) < 0.01:
                best_val_metric = val_metrics[0]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        # Check for early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break

        # Save the best model checkpoint
        if val_metrics[0] == best_val_metric:
            if not os.path.exists("./checkpoint"):
                os.makedirs("./checkpoint")
            transx.save_checkpoint('./checkpoint/'+MODEL_NAME+'.ckpt')

    # test the model
    transx.load_checkpoint('./checkpoint/'+MODEL_NAME+'.ckpt')
    tester = Tester(model = transx, data_loader = test_dataloader, use_gpu = True)
    tester.run_link_prediction(type_constrain = False)
    
    return val_metrics[0]
    # Log the end time and total time taken
#    end_time = time.time()
#    total_time = end_time - start_time
#    print(f"Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    # wandb.agent(sweep_id, function=train_main, count=NUM_TRIALS)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name=f"openKE_{MODEL_NAME}_HPO", direction="maximize", sampler=sampler
    )
    study.optimize(
        train_main_optuna, n_trials=NUM_TRIALS, catch=(ValueError,), callbacks=[wandbc]
    )
    print(study.best_params)
    print(study.trials)
    joblib.dump(study, f"openKE_{MODEL_NAME}_STUDY")
