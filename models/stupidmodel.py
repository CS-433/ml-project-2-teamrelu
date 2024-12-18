import torch
import torch.nn as nn
from .configs.dataset_class import create_vocabulary

# 2. Padding delle sequenze per renderle della stessa lunghezza
def sequences_naive_processing(sequences, max_length=512):
    vocabulary = create_vocabulary() 
    processed_sequences = [] 

    for seq in sequences:
        if len(seq) > max_length:
            seq = seq[:max_length]
        
        padded_seq = seq + "A" * (max_length - len(seq))

        processed_sequences.append(padded_seq)

    processed_sequences = torch.tensor([[vocabulary[char] for char in seq] for seq in processed_sequences], dtype=torch.long)

    return processed_sequences

# 5. Creare un semplice modello con uno strato denso
class NaiveModel(nn.Module):
    def __init__(self, input_dim=529, num_classes=15):
        super(NaiveModel, self).__init__()
        self.dense1 = nn.Linear(input_dim, 64)  # Primo strato denso
        self.relu = nn.ReLU()  # Funzione di attivazione ReLU
        self.dense2 = nn.Linear(64, num_classes)  # Strato di output

    def forward(self, x_static, x_dynamic):
        x_static = x_static.unsqueeze(1).repeat(1, self.num_timesteps, 1)
        combined_inputs = torch.cat([x_static, x_dynamic], dim=2)
        x = self.relu(self.dense1(x))
        x = self.dense2(x)
        return x