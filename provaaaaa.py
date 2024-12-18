import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from trainers.training_utils import validate, run_training
from models.static_model_MLP import StaticModelMLP
from models.static_model_multibranch import StaticModelMultibranch
from models.simple_model import SimpleDynamicModel
from models.temporal_block import TemporalBlock
from models.tcn_model import TCNDynamicModel
from models.modulable_lstm_model import ModulableLSTMDynamicModel
from losses.losses import CrossEntropy, CrossEntropyWithTemporalSmoothness, CrossEntropyWithLasso, loss_functions_map
from utils.training_graphs import plot_training_results
from configs.dataloaders import create_data_loaders

# Initialize random seed
seed = 30026
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Define data of the problem
num_classes = 15
num_timesteps = 5

# Decide whever to be verbose with display outputs and set device
verbose = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
with open('hyperparameters.json', 'r') as file:
    params = json.load(file)

static_params = params["static_model"]
dynamic_params = params["dynamic_model"]

# Load file
data = pd.read_csv(params["file_path"])

# Split dataset in train and test sets keeping balance for the static label localization
data_train, data_test = train_test_split(data, test_size=params["test_size"], random_state=seed, stratify = data['static_localization'])
# Create dataloaders
dataloader_train, dataloader_test = create_data_loaders(data_train, data_test,params["batch_size"])

# Set embedding dimensions based on json parameter
if static_params["long_embeddings"]:
    embedding_dim = 640
else:
    embedding_dim = 320

# Inizialize model: 0 for StaticModelMLP, 1 for StaticModelMultibranch, 2 for XGBoost
if static_params["model_type"] == 0:
    static_model = StaticModelMLP(input_size=embedding_dim, num_classes=num_classes)
if static_params["model_type"] == 1:
    static_model = StaticModelMultibranch(num_classes=num_classes, embedding_dim=embedding_dim, extremities_dim=20, char_vocab_size=20, char_embed_dim=16, intermediate_dim=static_params["intermediate_dim"], dropout=static_params["dropout"])

# Weight inizialization for robustness
static_model.initialize_weights()

print(static_params["learning_rate"])
print(static_params["weight_decay"])
print(static_params["scheduler"]["step_size"])
print(static_params["scheduler"]["gamma"])
print(static_params["scheduler"]["type"])

# Set optimizer
static_optimizer = torch.optim.AdamW(static_model.parameters(), static_params["learning_rate"], weight_decay=static_params["weight_decay"])

# Set schduler if required
if static_params["scheduler"]["type"] == "StepLR":
    static_scheduler = torch.optim.lr_scheduler.StepLR(static_optimizer, step_size=static_params["scheduler"]["step_size"], gamma=static_params["scheduler"]["gamma"])
elif static_params["scheduler"]["type"] == "CosineAnnealingLR":
    static_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(static_optimizer, T_max=static_params["scheduler"]["T_max"], eta_min=static_params["scheduler"]["eta_min"])
else:
    static_scheduler = False

# Run training
[train_loss, train_accuracy, test_loss, test_accuracy] = run_training(model = static_model,
                                                                        criterion = loss_functions_map[static_params["criterion"]],
                                                                        optimizer = static_optimizer,
                                                                        scheduler = static_scheduler,
                                                                        lambda_penalty = 0, # Not used for training static model
                                                                        dataloader_train = dataloader_train,
                                                                        dataloader_test = dataloader_test,
                                                                        num_epochs = static_params["num_epochs"],
                                                                        patience = static_params["patience"],
                                                                        device = device,
                                                                        verbose = verbose)
'''
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
dynamic_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

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