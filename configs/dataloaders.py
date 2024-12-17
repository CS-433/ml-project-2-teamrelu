import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from configs.dataset_class import ProteinDataset, create_vocabulary
from utils.data_cleaning_helpers import create_dynamic_tensor

def create_data_loaders(data_train, data_test, batch_size):
    """ Create train and test dataloaders
    Args:
        data: DataFrame with all data
        batch_size: size of each batch
        seed: random generator seed
    Returns:
        dataloader_train: dataloader for training set
        dataloader_test: dataloader for test set
    """
    # Handle dynamic data
    dynamic_data_train = create_dynamic_tensor(data_train.iloc[:, 653:728], data_train.iloc[:, 648:653], data_train.iloc[:,643:648])
    dynamic_data_test = create_dynamic_tensor(data_test.iloc[:, 653:728], data_test.iloc[:, 648:653], data_test.iloc[:,643:648])
    # Split dataframe columns based on feature groups
    global_embeddings_train = data_train.iloc[:, 1:641]
    start_sequences_train = data_train.iloc[:, 641]
    end_sequences_train = data_train.iloc[:, 642]
    static_labels_train = data_train.iloc[:, 728]
    dynamic_labels_train = data_train.iloc[:, 730:735] #729 to keep G1-Prestart
    global_embeddings_test = data_test.iloc[:, 1:641]
    start_sequences_test = data_test.iloc[:, 641]
    end_sequences_test = data_test.iloc[:, 642]
    static_labels_test = data_test.iloc[:, 728]
    dynamic_labels_test = data_test.iloc[:, 730:735] #729 to keep G1-Prestart
    # Create vocabulary for dataset creation
    vocabulary = create_vocabulary()
    # Create datasets
    dataset_train = ProteinDataset(dynamic_data_train, global_embeddings_train, start_sequences_train, end_sequences_train, static_labels_train, dynamic_labels_train, vocabulary)
    dataset_test = ProteinDataset(dynamic_data_test, global_embeddings_test, start_sequences_test, end_sequences_test, static_labels_test, dynamic_labels_test, vocabulary)
    # Create dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_test