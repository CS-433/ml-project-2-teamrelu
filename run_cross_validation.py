import torch
import torch.nn as nn
import pandas as pd
from trainers.cross_validation import k_fold_cross_validation
from models.static_model_multibranch import StaticModelMultibranch
from losses.losses import CrossEntropy

# Initialize random seed
seed = 32
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Define data of the problem
num_classes = 15
num_timesteps = 5
# Define hyperparameters
weight_decay = 1e-4
num_epochs = 2
patience = 5
batch_size = 32
static_learnable = False
learning_rate = 1e-2
eta_min = 1e-5
dropout = 0.2
lambda_penalty = 1e-5
test_size = 0.2 # for train/test splitting
# Define device to be used for computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
file_path = 'final_dataset.csv'
data = pd.read_csv(file_path)

parameter_grid = 

best_params, results = k_fold_cross_validation(
    StaticModelMultibranch,
    data,
    CrossEntropy,
    parameter_grid,
    num_epochs,
    patience,
    batch_size,
    device,
    k_folds=5,
    lambda_penalty=0.0,
    seed=32,
    cross_validation_on_loss=True,
    verbose=True
)