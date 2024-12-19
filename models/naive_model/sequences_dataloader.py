import torch
from torch.utils.data import Dataset

class NaiveProteinDataset(Dataset):
    def __init__(self, sequences, dynamic_data, labels, max_length=512):
        """
        Args:
            sequences (pd:Dataframe): List of protein sequences
            sequences (Tensor): 3D tensor with dynamic data
            labels (pd.Dataframe): Dataser of shape (N, 5) containing 5D labels for each sequence
            max_length (int): Maximum sequence length for padding/truncation. Default is 512
        """
        super(NaiveProteinDataset, self).__init__()
        sequences = sequences.tolist()
        labels = labels.to_numpy()

        self.sequences = sequences_naive_processing(sequences, max_length=max_length)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
        self.sequences = sequences_naive_processing(sequences, max_length=max_length)
        self.dynamic_data = dynamic_data
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns the processed sequence and the corresponding label.

        Args:
            idx (int): Index of the item to fetch.

        Returns:
            torch.Tensor: Processed sequence tensor.
            torch.Tensor: Corresponding label tensor.
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        dynamic_datum = self.dynamic_data[idx]

        return sequence, dynamic_datum, label
    
def sequences_naive_processing(sequences, max_length=512):
    """
    Processes a list of sequences into a fixed-length tensor representation.

    Each sequence is truncated or padded to a specified maximum length, 
    and characters are mapped to integer indices using a vocabulary.

    Args:
        sequences (list of str): A list of input sequences (strings) to process.
        max_length (int, optional): The maximum length for sequences. 
                                    Sequences longer than this will be truncated, 
                                    and shorter sequences will be padded. Default is 512.

    Returns:
        torch.Tensor: A tensor of shape (num_sequences, max_length) containing the 
                      processed sequences, where each character is replaced by its 
                      corresponding integer index from the vocabulary.
    """

    vocabulary = create_vocabulary() 
    processed_sequences = [] 

    for seq in sequences:
        if len(seq) > max_length:
            seq = seq[:max_length]
        
        padded_seq = seq + "A" * (max_length - len(seq))

        processed_sequences.append(padded_seq)

    processed_sequences = torch.tensor([[vocabulary[char] for char in seq] for seq in processed_sequences], dtype=torch.long)

    return processed_sequences

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

def create_dynamic_tensor(interaction_data, te_levels, tl_levels):
    """
    Create a the dynamic tensor to pass to the models
    
    Args:
        interaction_data : pandas Dataframe containing the interactions 
        te_levels : pandas DataFrame containing the TE concentration levels 
        tl_levels : pandas DataFrame containing the TL concentration levels 
        
    Returns:
        result_tensor : pandas DataFrame containing the dynamic dataset
    """
    #Dataframe to tensor
    dynamic_tensor = torch.tensor(interaction_data.values, dtype=torch.float32)
    # Add temporal length
    dynamic_tensor = dynamic_tensor.unsqueeze(1)
    # Split data of each timestep
    split_tensors = torch.split(dynamic_tensor, 15, dim=2)
    # Concatenate along the second dimension (dim=1)
    result_tensor = torch.cat(split_tensors, dim=1)

    # Concatenate the concentration tensors
    te_levels_tensor = torch.tensor(te_levels.values, dtype=torch.float32)
    delta_TE = te_levels_tensor[:, 1:] - te_levels_tensor[:, :-1]
    # Padding to keep original dimension
    delta_TE = torch.cat([torch.zeros(delta_TE.size(0), 1), delta_TE], dim=1)
    te_levels_tensor = te_levels_tensor.unsqueeze(-1)
    tl_levels_tensor = torch.tensor(tl_levels.values, dtype=torch.float32)
    delta_TL = tl_levels_tensor[:, 1:] - tl_levels_tensor[:, :-1]
    # Padding to keep original dimension
    delta_TL = torch.cat([torch.zeros(delta_TL.size(0), 1), delta_TL], dim=1)
    tl_levels_tensor = tl_levels_tensor.unsqueeze(-1)
    result_tensor = torch.cat((result_tensor, te_levels_tensor, tl_levels_tensor), dim=2)
    delta_TE = delta_TE.unsqueeze(-1)
    delta_TL = delta_TL.unsqueeze(-1)
    # Concatenate the variation tensors
    result_tensor = torch.cat((result_tensor, delta_TE, delta_TL), dim=2)

    return result_tensor