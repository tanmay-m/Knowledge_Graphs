import optuna
from pykg2vec.data.kgcontroller import KnowledgeGraph
from pykg2vec.common import Importer, KGEArgParser
from pykg2vec.utils.trainer import Trainer
import wandb
from optuna_integration.wandb import WeightsAndBiasesCallback
from types import SimpleNamespace
from config import *
import joblib
import torch
import yaml
import sys
sys.path.append("../")
from config import *
wandb.login(key=WANDB_API_KEY)

wandb_kwargs = {"project": WANDB_PROJECT, "group": "pykg2vec_" + MODEL_NAME}
wandbc = WeightsAndBiasesCallback(
    metric_name="mrr", wandb_kwargs=wandb_kwargs, as_multirun=True
)


@wandbc.track_in_wandb()
def train_main():
    base_params = SimpleNamespace(
        lmbda=0.1,
        batch_size=128,
        margin=0.8,
        optimizer="adam",
        sampling="uniform",
        neg_rate=1,
        epochs=400,
        learning_rate=0.01,
        hidden_size=50,
        ent_hidden_size=50,
        rel_hidden_size=50,
        hidden_size_1=10,
        l1_flag=True,
        alpha=0.1,
        filter_sizes=[1, 2, 3],
        num_filters=50,
        feature_map_dropout=0.2,
        input_dropout=0.3,
        hidden_dropout=0.3,
        hidden_dropout1=0.4,
        hidden_dropout2=0.5,
        label_smoothing=0.1,
        cmax=0.05,
        cmin=5.0,
        feature_permutation=1,
        reshape_height=20,
        reshape_width=10,
        kernel_size=9,
        in_channels=9,
        way="parallel",
        first_atrous=1,
        second_atrous=2,
        third_atrous=2,
        acre_bias=True,
        model_name="TransE",
        debug=False,
        exp=False,
        dataset_name="fb15k_237",
        dataset_path=None,
        load_from_data=None,
        save_model=True,
        test_num=1000,
        test_step=10,
        tmp="../intermediate",
        result="../results",
        figures="../figures",
        plot_embedding=False,
        plot_entity_only=False,
        device="cuda",
        num_process_gen=2,
        hp_abs_file=None,
        ss_abs_file=None,
        max_number_trials=400,
    )
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    with open(r"../best_params.yaml") as stream:
        try:
            best_params = yaml.safe_load(stream)['pykg2vec']
        except yaml.YAMLError as exc:
            print(exc)
    
    knowledge_graph = KnowledgeGraph(dataset="fb15k_237")
    knowledge_graph.prepare_data()
    base_params.batch_size = best_params.batch_size
    base_params.hidden_size = best_params.hidden_size
    base_params.margin = 0.5
    base_params.optimizer = best_params.optimizer
    base_params.learning_rate = best_params.learning_rate
    base_params.model_name = MODEL_NAME
    # Extracting the corresponding model config and definition from Importer().
    config_def, model_def = Importer().import_model_config(MODEL_NAME.lower())
    config = config_def(base_params)
    model = model_def(**config.__dict__)

    trainer = Trainer(model, config)
    trainer.build_model()
    trainer.train_model()
    trainer.evaluator.metric_calculator.display_summary()
    training_results_table = wandb.Table(
        columns=["Epoch", "Train_Loss"], data=trainer.training_results
    )
    wandb.log({"Training": training_results_table})
    val_scores = trainer.evaluator.metric_calculator.get_curr_scores()
    wandb.log(val_scores)
    fmrr = trainer.evaluator.metric_calculator.fmrr[0]
    return fmrr


if __name__ == "__main__":
    # wandb.agent(sweep_id, function=train_main, count=NUM_TRIALS)
    train_main()
