import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from sklearn.model_selection import train_test_split
from trainers.training_utils import validate, run_training
from models.static_model_MLP import StaticModelMLP
from models.static_model_multibranch import StaticModelMultibranch
from models.simple_model import SimpleDynamicModel
from models.temporal_block import TemporalBlock
from models.tcn_model import TCNDynamicModel
from models.modulable_lstm_model import ModulableLSTMDynamicModel
from losses.losses import CrossEntropy, CrossEntropyWithTemporalSmoothness, CrossEntropyWithLasso
from utils.training_graphs import plot_training_results
from configs.dataloaders import create_data_loaders

# Initialize random seed
seed = 30026
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Define data of the problem
num_classes = 15
num_timesteps = 5
# Define hyperparameters
weight_decay = 1e-2
num_epochs = 100
patience = 20
batch_size = 32
static_learnable = False
learning_rate = 1e-3
eta_min = 1e-5
dropout = 0.25
lambda_penalty = 1e-5
test_size = 0.2 # for train/test splitting
# Define device to be used for computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

no_concentration = False
no_interaction = True
no_static = True

# Load data
file_path = 'datasets/final_dataset_dyn_with_te.csv'
data = pd.read_csv(file_path)


# Split dataset in train and test sets keeping balance for the static label localization
data_train, data_test = train_test_split(data, test_size=test_size, random_state=seed, stratify = data['static_localization'])
# Create dataloaders
dataloader_train, dataloader_test = create_data_loaders(data_train, data_test, batch_size)

# First run the static model
static_model = StaticModelMultibranch(num_classes=num_classes, embedding_dim=640, extremities_dim=20, char_vocab_size=20, char_embed_dim=16, intermediate_dim=32, dropout=dropout)
static_model.init_weights()
optimizer = torch.optim.AdamW(static_model.parameters(), learning_rate, weight_decay=weight_decay)
static_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, learning_rate, gamma=0.1)
[train_loss, train_accuracy, test_loss, test_accuracy] = run_training(model = static_model,
                                                                        criterion = CrossEntropy,
                                                                        optimizer = optimizer,
                                                                        scheduler = static_scheduler,
                                                                        lambda_penalty = lambda_penalty,
                                                                        dataloader_train = dataloader_train,
                                                                        dataloader_test = dataloader_test,
                                                                        num_epochs = num_epochs,
                                                                        patience = patience,
                                                                        device = device,
                                                                        verbose = True)
# Define TCN block and initialize its weights
temporal_block = TemporalBlock(input_dim=17, intermediate_dim=15, output_dim=num_classes, kernel_size=3, stride=1, dropout=0.2)
temporal_block.initialize_weights()
# Create dynamic model and initialize its weights
#dynamic_model = TCNDynamicModel(static_model, temporal_block, static_learnable=False, num_timesteps=num_timesteps, num_classes=num_classes)
#dynamic_model.initialize_weights()
dynamic_model = ModulableLSTMDynamicModel(static_model, static_learnable=static_learnable, num_timesteps=num_timesteps, num_classes=num_classes, num_features = 34, hidden_size = 64, num_layers=2, dropout=dropout, no_concentration=no_concentration, no_interaction=no_interaction, no_static=no_static)
# Define optimizer
if static_learnable == True:
    optimizer = torch.optim.AdamW( [{'params': dynamic_model.StaticModel.parameters(), 'lr': eta_min},
                                {'params': dynamic_model.TemporalBlock.parameters(), 'lr': learning_rate},
                                {'params': dynamic_model.CombinationLayer.parameters(), 'lr': learning_rate}], weight_decay=weight_decay)
else:
    optimizer = torch.optim.AdamW(dynamic_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# Define scheduler
dynamic_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, learning_rate, gamma=0.1)

[train_loss, train_accuracy, test_loss, test_accuracy] = run_training(model = dynamic_model,
                                                                        criterion = CrossEntropy,
                                                                        optimizer = optimizer,
                                                                        scheduler = dynamic_scheduler,
                                                                        lambda_penalty = lambda_penalty,
                                                                        dataloader_train = dataloader_train,
                                                                        dataloader_test = dataloader_test,
                                                                        num_epochs = num_epochs,
                                                                        patience = patience,
                                                                        device = device,
                                                                        verbose = True)
plot_training_results(train_loss, train_accuracy, test_loss, test_accuracy)


'''
model = SimpleDynamicModel(embeddings_dim=640, hidden_dim=32, num_classes=15, num_timesteps=5)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
[train_loss, train_accuracy, test_loss, test_accuracy] = run_training(model = model,
                                                                        criterion = CrossEntropy,
                                                                        optimizer = optimizer,
                                                                        scheduler = False,
                                                                        lambda_penalty = lambda_penalty,
                                                                        dataloader_train = dataloader_train,
                                                                        dataloader_test = dataloader_test,
                                                                        num_epochs = num_epochs,
                                                                        device = device,
                                                                        verbose = True)
                                                                        '''