import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    """ Dataset class for handling proteins dynamic and static data

    Attributes:
        global_embeddings (torch.Tensor): Tensor containing global embeddings for each protein
        ind_start_sequences (torch.Tensor): Tensor containing indices for the first 20 amino acids of each protein sequence
        ind_end_sequences (torch.Tensor): Tensor containing indices for the last 20 amino acids of each protein sequence
        static_labels (torch.Tensor): Tensor containing static labels for each protein
        dynamic_labels (torch.Tensor): Tensor containing dynamic labels for each protein
    
    Methods:
        __len__ : Returns the number of data points in the dataset
        __getitem__ : Returns the data point at a specific index (idx), including dynamic data, embeddings, start and end sequences, and labels
    """
    def __init__(self, dynamic_data, global_embeddings, start_sequences, end_sequences, static_labels, dynamic_labels, vocabulary):
        """ Class constructor

        Args:
            dynamic_data (torch tensor): 3D torch tensor containing the dynamic data for each protein.
            global_embeddings (pd.DataFrame or ndarray): Embeddings representing global features of each protein.
            start_sequences (list of str or pd.Series of str): List containing the first 20 amino acids of each protein sequence.
            end_sequences (list of str or pd.Series of str): List containing the last 20 amino acids of each protein sequence.
            static_labels (pd.Series or ndarray): Static labels associated with each protein.
            dynamic_labels (pd.DataFrame or ndarray): Dynamic labels associated with each protein.
            vocabulary (dict): A dictionary mapping amino acid characters to indices for encoding the protein sequences.
        """
        self.dynamic_data = dynamic_data
        self.global_embeddings = torch.tensor(global_embeddings.values, dtype=torch.float32) if isinstance(global_embeddings, pd.DataFrame) else torch.tensor(global_embeddings, dtype=torch.float32)
        self.ind_start_sequences = torch.tensor([[vocabulary[char] for char in seq] for seq in start_sequences], dtype=torch.long)
        self.ind_end_sequences = torch.tensor([[vocabulary[char] for char in seq] for seq in end_sequences], dtype=torch.long)
        self.static_labels = torch.tensor(static_labels.values, dtype=torch.long) if isinstance(static_labels, pd.Series) else torch.tensor(static_labels, dtype=torch.float32)
        self.dynamic_labels = torch.tensor(dynamic_labels.values, dtype=torch.long) if isinstance(dynamic_labels, pd.DataFrame) else torch.tensor(dynamic_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.static_labels)

    def __getitem__(self, idx):
        return self.dynamic_data[idx], self.global_embeddings[idx], self.ind_start_sequences[idx], self.ind_end_sequences[idx], self.static_labels[idx], self.dynamic_labels[idx]

#Function to create a vocabulary to be passed to the ProteinDataset class
def create_vocabulary():
    """
    Creates a vocabulary mapping amino acid characters to unique indices.
    
    The vocabulary includes 20 standard amino acids, each mapped to an index 
    based on their position in the list.

    Returns:
        dict: A dictionary where keys are amino acid characters (str) and 
              values are their corresponding indices (int).
    """
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    vocab = {amino_acid: idx for idx, amino_acid in enumerate(amino_acids)}

    return vocab