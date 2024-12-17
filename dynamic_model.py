import torch
import torch.nn as nn
from temporal_block import TemporalBlock

class DynamicModel(nn.Module):
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
    def __init__(self, static_model, tcn_model, static_learnable=False, num_timesteps=5, num_classes=15):
        super(DynamicModel, self).__init__()

        self.num_timesteps = num_timesteps
        self.TemporalBlock = tcn_model
        self.StaticModel = static_model
        self.StaticLearnable = static_learnable

        # Freeze static submodule weights if required
        if static_learnable == False:
          for param in static_model.parameters():
              param.requires_grad = False

        # Last linear submodule for classification
        self.CombinationLayer = nn.Sequential(
            nn.Linear(num_classes*2, num_classes)
        )

    def initialize_weights(self):
      for layer in self.CombinationLayer.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
          nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
          if layer.bias is not None:
              nn.init.constant_(layer.bias, 0)

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
        # Static submodule
        if self.StaticLearnable == False:
          with torch.no_grad():
            static_output = self.StaticModel(global_embeddings, ind_start_seq, ind_end_seq)
        else:
          static_output = self.StaticModel(global_embeddings, ind_start_seq, ind_end_seq)

        # Dynamic submodule
        dynamic_output = self.TemporalBlock(dynamic_data) # [batch_size, timesteps, num_classes]

        # Cast static output to match dynamic data dimensions
        static_output = static_output.unsqueeze(1).expand(-1, self.num_timesteps, -1)  # [batch_size, timesteps, num_classes]

        # Concatenate outputs along the feature dimension
        combined = torch.cat((dynamic_output, static_output), dim=-1)  # [batch_size, timesteps, num_classes * 2]

        # Apply linear layer
        output = self.CombinationLayer(combined)  # [batch_size, timesteps, num_classes]

        return output