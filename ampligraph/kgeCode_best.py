import numpy as np
from ampligraph.datasets import load_fb15k_237
from ampligraph.latent_features import ScoringBasedEmbeddingModel
from ampligraph.evaluation import mrr_score, hits_at_n_score
from ampligraph.latent_features.loss_functions import get as get_loss
from ampligraph.latent_features.regularizers import get as get_regularizer
import tensorflow as tf
import sys
sys.path.append("../")
from config import *
# Load Wordnet18 dataset
X = load_fb15k_237()

# Initialize a ComplEx neural embedding model with specified parameters
model = ScoringBasedEmbeddingModel(k=150,
                                   eta=10,
                                   scoring_type='TransE')

# Optimizer, loss and regularizer definition with specified parameters
optim = tf.keras.optimizers.Adagrad(learning_rate=0.004)
loss = get_loss('pairwise', {'margin': 0.5})
regularizer = get_regularizer('LP', {'p': 2, 'lambda': 1e-5})

# Compilation of the model
model.compile(optimizer=optim, loss=loss, entity_relation_regularizer=regularizer)

# For evaluation, define the filter set by concatenating all the positives
filter_set = {'test': np.concatenate((X['train'], X['valid'], X['test']))}

# Early Stopping callback
checkpoint = tf.keras.callbacks.EarlyStopping(
    monitor='val_{}'.format('hits10'),
    min_delta=0,
    patience=5,
    verbose=1,
    mode='max',
    restore_best_weights=True
)

# Fit the model on training and validation set with specified parameters
model.fit(X['train'],
          batch_size=1024,               # Batch size
          epochs=400,                     # Number of training epochs
          validation_freq=20,            # Epochs between successive validation
          validation_burn_in=100,        # Epoch to start validation
          validation_data=X['valid'],    # Validation data
          validation_filter=filter_set,  # Filter positives from validation corruptions
          callbacks=[checkpoint],        # Early stopping callback
          verbose=True                   # Enable stdout messages
          )

# Run the evaluation procedure on the test set (with filtering)
ranks = model.evaluate(X['test'],
                       use_filter=filter_set,
                       corrupt_side='s,o')

# Compute and print metrics
mrr = mrr_score(ranks)
hits_10 = hits_at_n_score(ranks, n=10)
print("MRR: %f, Hits@10: %f" % (mrr, hits_10))
