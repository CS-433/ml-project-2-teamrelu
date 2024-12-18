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


def static_training(num_classes, static_params, dataloader_train, dataloader_test, verbose, device, seed):

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
    #static_model.initialize_weights()

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

    return train_loss, train_accuracy, test_loss, test_accuracy, static_model.state_dict()