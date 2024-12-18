import torch
import torch.nn as nn
import pandas as pd
from trainers.cross_validation import k_fold_cross_validation_static, k_fold_cross_validation_dynamic
from models.static_model_multibranch import StaticModelMultibranch
from models.modulable_lstm_model import LSTMDynamicModel
from losses.losses import CrossEntropy, CrossEntropyWithLasso, CrossEntropyWithTemporalSmoothness
import json

# Initialize random seed
seed = 30026
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Define data of the problem
num_classes = 15
num_timesteps = 5
# Define hyperparameters
num_epochs = 100
patience = 10
batch_size = 32
k_folds = 4
test_size = 0.2 # for train/test splitting
# Define device to be used for computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
file_path = 'final_dataset.csv'
data = pd.read_csv(file_path)
'''
parameter_grid = {
    "dropout": [0, 0.25, 0.5],
    "learning_rate": [0.001, 0.01],
    "weight_decay": [0, 0.0001, 0.01],
    "scheduler": ['CosineAnnealingLR', 'StepLR', 'ExponentialLR']
}

best_params, results = k_fold_cross_validation_static(
    StaticModelMultibranch,
    data,
    CrossEntropy,
    parameter_grid,
    num_epochs,
    patience,
    batch_size,
    device,
    k_folds=k_folds,
    lambda_penalty=0.0,
    seed=seed,
    cross_validation_on_loss=True,
    verbose=True
)

results_file = 'cross_val_results'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=4)

best_file = 'best_results'
with open(best_file, 'w') as f:
    json.dump(best_params, f, indent=4)
'''

# FOR DYNAMIC MODEL
# Load best parameters

# Now run and save model with best parameters
file_path = 'best_results'
with open(file_path, 'r') as file:
    best_params_static = json.load(file)
    
# Add criterion to best parameters
best_params_static["criterion"] = CrossEntropy

parameter_grid = {
    "dropout": [0.2],
    "learning_rate": [0.001, 0.01],
    "weight_decay": [0.01],
    "scheduler": ['StepLR'],
    "criterion": [CrossEntropyWithTemporalSmoothness],
    "lambda_penalty": [0, 1e-5, 1e-2],
    "static_learnable": [True, False],
    "hidden_size": [32, 64]
}


best_params, results = k_fold_cross_validation_dynamic(
    StaticModelMultibranch,
    LSTMDynamicModel,
    data,
    parameter_grid,
    best_params_static,
    num_epochs,
    patience,
    batch_size,
    device,
    k_folds=3,
    seed=seed,
    cross_validation_on_loss=True,
    verbose=True
)

results_file = 'cross_val_results_dynamic'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=4)

best_file = 'best_results_dynamic'
with open(best_file, 'w') as f:
    json.dump(best_params, f, indent=4)