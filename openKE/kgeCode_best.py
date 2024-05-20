import os, time
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
from types import SimpleNamespace
from config import *
import joblib
import pandas as pd
import yaml
import sys
sys.path.append("../")
from config import *
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
MODEL_NAME = "TransE"

# read the hyperparams file
with open(r"../best_params.yaml") as stream:
    try:
        openke_row = yaml.safe_load(stream)['openke']
    except yaml.YAMLError as exc:
        print(exc)

# Extract values from the filtered row
embedding_dimension = openke_row["embeddingDimension"]
batch_size = openke_row["batchSize"]
optimizer = openke_row["optimizer"]
margin = openke_row["margin"]
alpha = openke_row["alpha"]


def train_main():

    # define the model
    transx = TransE(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = embedding_dimension, 
        p_norm = 1, 
        norm_flag = False
        )

    print("batch size = ",train_dataloader.get_batch_size())
    # define the loss function
    model = NegativeSampling(
        model = transx, 
        loss = MarginLoss(margin = margin),
        # batch_size = batch_size
       batch_size = train_dataloader.get_batch_size()
    )

    # Define the early stopping parameters
    patience = 5  # number of epochs to wait before stopping
    best_val_metric = None  # initialize with None
    epochs_without_improvement = 0

    # start time
    start_time = time.time()

    for epoch in range(400):
        # Train for one epoch
        print(f"Epoch #: {epoch+1}")
        start_train_time = time.time()  # Record the start time for training
        trainer = Trainer(model=model, data_loader=train_dataloader, train_times=1, alpha=alpha, use_gpu=True, opt_method=optimizer)
        trainer.run()
        end_train_time = time.time()  # Record the end time for training
        train_time = end_train_time - start_train_time
        print(f"Training time for epoch {epoch+1}: {train_time:.2f} seconds")
        
        start_eval_time = time.time()  # Record the start time for evaluation
        val_metrics = get_validation_metrics(transx, test_dataloader, use_gpu=True)
        end_eval_time = time.time()  # Record the end time for evaluation
        eval_time = end_eval_time - start_eval_time
        print(f"Evaluation time for epoch {epoch+1}: {eval_time:.2f} seconds")
        
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

    # Log the end time and total time taken
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
    
    return val_metrics[0]


train_main()
    
