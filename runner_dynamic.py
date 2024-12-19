import torch
import torch.nn as nn
from trainers.training_utils import run_training
from models.dynamic_models.simple_model import SimpleDynamicModel
from models.dynamic_models.lstm_model import LSTMDynamicModel
from models.dynamic_models.temporal_block import TemporalBlock
from models.dynamic_models.modulable_lstm_model import ModulableLSTMDynamicModel
from models.dynamic_models.tcn_model import TCNDynamicModel
from models.dynamic_models.naive_model import NaiveModel
from losses.losses import CrossEntropy, CrossEntropyWithTemporalSmoothness, CrossEntropyWithLasso, loss_functions_map
from trainers.training_graphs import plot_training_results
from configs.dataloaders import create_data_loaders


def dynamic_training(num_features, num_classes, num_timesteps, embedding_dim, dynamic_params, static_model, dataloader_train, dataloader_test, verbose, device):

    static_model = static_model.to(device)
    # Inizialize model: 0 for LSTMDynamicModel, 1 for TCNDynamicModel, 2 SimpleDynamicModel, 3 ModulableLSTMDynamicModel
    if dynamic_params["model_type"] == 0:
        if verbose:
            print('Starting training of LSTMDynamicModel')
        dynamic_model = LSTMDynamicModel(static_model, static_learnable=dynamic_params["static_learnable"], num_timesteps=num_timesteps, num_classes=num_classes, num_features = num_features, hidden_size = dynamic_params["intermediate_dim"], dropout=dynamic_params["dropout"])
    if dynamic_params["model_type"] == 1:
        if verbose:
            print('Starting training of TCNDynamicModel')
        temporal_block = TemporalBlock(input_dim=19, intermediate_dim=dynamic_params["intermediate_dim"], output_dim=num_classes, dropout=dynamic_params["dropout"])
        temporal_block.initialize_weights()
        dynamic_model = TCNDynamicModel(static_model, temporal_block, static_learnable=dynamic_params["static_learnable"], num_timesteps=num_timesteps, num_classes=num_classes)
    if dynamic_params["model_type"] == 2:
        if verbose:
            print('Starting training of SimpleDynamicModel')
        dynamic_model = SimpleDynamicModel(embeddings_dim=embedding_dim, hidden_dim=dynamic_params["intermediate_dim"], num_classes=num_classes, num_timesteps=num_timesteps)
    if dynamic_params["model_type"] == 3:
        if verbose:
            print('Starting training of ModulableLSTMDynamicModel')
        dynamic_model = ModulableLSTMDynamicModel(static_model, static_learnable=dynamic_params["static_learnable"], num_timesteps=num_timesteps, num_classes=num_classes, num_features = num_features, hidden_size = dynamic_params["intermediate_dim"], dropout=dynamic_params["dropout"], no_concentration=dynamic_params["no_concentration"], no_interaction=dynamic_params["no_interaction"], no_static=dynamic_params["no_static"])

    # Weight inizialization for robustness
    #static_model.initialize_weights()
    # Set optimizer
    dynamic_optimizer = torch.optim.AdamW(dynamic_model.parameters(), dynamic_params["learning_rate"], weight_decay=dynamic_params["weight_decay"])

    # Set schduler if required
    if dynamic_params["scheduler"]["type"] == "StepLR":
        dynamic_scheduler = torch.optim.lr_scheduler.StepLR(dynamic_optimizer, step_size=dynamic_params["scheduler"]["step_size"], gamma=dynamic_params["scheduler"]["gamma"])
    elif dynamic_params["scheduler"]["type"] == "CosineAnnealingLR":
        dynamic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dynamic_optimizer, T_max=dynamic_params["scheduler"]["T_max"], eta_min=dynamic_params["scheduler"]["eta_min"])
    else:
        dynamic_scheduler = False

    # Run training
    [train_loss, train_accuracy, test_loss, test_accuracy] = run_training(model = dynamic_model,
                                                                            criterion = loss_functions_map[dynamic_params["criterion"]],
                                                                            optimizer = dynamic_optimizer,
                                                                            scheduler = dynamic_scheduler,
                                                                            lambda_penalty = dynamic_params["lambda_penalty"], # Not used for training static model
                                                                            dataloader_train = dataloader_train,
                                                                            dataloader_test = dataloader_test,
                                                                            num_epochs = dynamic_params["num_epochs"],
                                                                            patience = dynamic_params["patience"],
                                                                            device = device,
                                                                            verbose = verbose)
    if verbose:
        plot_training_results(train_loss, train_accuracy, test_loss, test_accuracy)

    if verbose:
        dynamic_accuracy = eval_dynamic_accuracy(dynamic_model, dataloader_test, device)
        print(f"Accuracy only on dynamic proteins: {dynamic_accuracy}")


    return train_loss, train_accuracy, test_loss, test_accuracy, dynamic_model.state_dict() 


def eval_dynamic_accuracy(dynamic_model, dataloader_test, device):

    dynamic_model.eval()
    total_correct_accuracy = 0  
    total_count = 0  

    for batch in dataloader_test:

        dynamic_data, global_input, ind_start_seq, ind_end_seq, static_label, dynamic_label = batch
        dynamic_data = dynamic_data.to(device)
        global_input = global_input.to(device)
        ind_start_seq = ind_start_seq.to(device)
        ind_end_seq = ind_end_seq.to(device)
        static_label = static_label.to(device)
        dynamic_label = dynamic_label.to(device)
        
        output_val = dynamic_model(dynamic_data, global_input, ind_start_seq, ind_end_seq)
        
        _, predicted = torch.max(output_val, 2)
        
        mask = ~torch.all(dynamic_label[:, 1:] == dynamic_label[:, :-1], dim=1) 
        filtered_predicted = predicted[mask]
        filtered_labels = dynamic_label[mask]
        
        if filtered_predicted.size(0) > 0:
            correct = (filtered_predicted == filtered_labels).sum().item() #counts how many correct predictions we have
            total_correct_accuracy += correct
            total_count += filtered_predicted.numel()

    # Calcola l'accuracy finale
    accuracy = total_correct_accuracy / total_count if total_count > 0 else 0
    
    return accuracy
