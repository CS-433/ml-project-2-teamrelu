import torch
import torch.nn as nn

class SimpleDynamicModel(nn.Module):
    """ DynamicalModel construced with static and TCN models
    Args:
        static_model: model created for the static localization problem
        tcn_model: model created for dynamical data feature extraction
        static_learnable: boolean for deciding if static model weights have to be frozen
        timesteps: number of timesteps in out problem
        num_classes: number of classes for classification

    Methods:
		    initialize_weights: He weights initialization for all the layers
        forward: usual forward pass of neural networks
    """
    def __init__(self, embeddings_dim=640, hidden_dim=256, num_classes=15, num_timesteps=5):
        super(SimpleDynamicModel, self).__init__()

        self.num_timesteps = num_timesteps
        # Last linear submodule for classification
        self.CombinationLayer = nn.Sequential(
            #nn.Linear(embeddings_dim, hidden_dim),
            #nn.LayerNorm(hidden_dim, eps=1e-05),
            #nn.ReLU(),
            #nn.Linear(hidden_dim, num_classes), #  Second layer, reduces the dimension from 256 to 64
            #nn.LayerNorm(num_classes, eps=1e-05),
            #nn.ReLU(),
            nn.Linear(embeddings_dim, num_classes), # Output layer, reduces the dimension from 64 to the number of classes
        )

    def forward(self, dynamic_data, global_embeddings, ind_start_seq, ind_end_seq):
        """ 
        Args:
            dynamic_data: dynamic features
            global_embeddings: embeddings made with ESM
            ind_start_seq: first 20 amino acids of each sequence
            ind_end_seq: last 20 amino acids of each sequence
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, timesteps, num_classes), 
                          containing the predicted class scores for each sequence position
        """

        # Cast static output to match dynamic data dimensions
        global_embeddings = global_embeddings.unsqueeze(1).expand(-1, self.num_timesteps, -1)  # [batch_size, timesteps, num_classes]

        # Apply linear layer
        output = self.CombinationLayer(global_embeddings)  # [batch_size, timesteps, num_classes]

        return output