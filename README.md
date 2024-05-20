## KNOWLEDGE GRAPH LIBRARIES BENCHMARK

* This repository contains implementations for different KGE libaries and its benchmarking. It offers support for hyperparameter tuning using Optuna and Weights and Biases for MLOPs

Add the MODEL_NAME and WANDB configuration in the `config.py` file

To run the benchmarks, run the following command

1. Run torchkge with only best parameters defined in the `best_params.yaml` file
```python
python3 main.py -lib torchkge -hp false
```

2. Run pykg2vec with only best parameters defined in the `best_params.yaml` file
```python
python3 main.py -lib pykg2vec -hp false
```

3. Run torchkge with hyperparameter tuning
```python
python3 main.py -lib torchkge -hp true
```

