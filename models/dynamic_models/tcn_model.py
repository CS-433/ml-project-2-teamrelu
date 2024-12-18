import torch
import torch.nn as nn
from models.dynamic_models.temporal_block import TemporalBlock

class TCNDynamicModel(nn.Module):
    """ 
      DynamicalModel construced with static and TCN models
      
      Attributes:
        input_dim (int): Dimension of the input embeddings
        inner (nn.Sequential): Multilayer Perceptron
        num_timesteps (int): Number of timesteps in the dynamic data
        TemporalBlock (nn.Module): TCN model for dynamic data feature extraction
        StaticModel (nn.Module): Model created for the static localization problem
        StaticLearnable (bool): Whether to allow fine-tuning of the static model
    
      Methods:
		    forward : forward pass of the method
        initialize_weights: applies He initialization to all the linear layers
    
    """
    def __init__(self, static_model, tcn_model, static_learnable=False, num_timesteps=5, num_classes=15):
        super(TCNDynamicModel, self).__init__()
        """
        Class constructor

        Args:
          static_model (nn.Module): Model created for the static localization problem
          tcn_model (nn.Module): Model created for dynamical data feature extraction
          static_learnable (bool): Whether to allow fine-tuning of the static model
          num_timesteps (int): Number of timesteps in the dynamic data
          num_classes (int): Number of classes for classification
        """

        self.num_timesteps = num_timesteps
        self.TemporalBlock = tcn_model
        self.StaticModel = static_model
        self.StaticLearnable = static_learnable

        # Freeze static submodule weights if required
        if static_learnable == False:
          for param in self.StaticModel.parameters():
              param.requires_grad = False

        # Last linear submodule for classification
        self.CombinationLayer = nn.Sequential(
            nn.Linear(num_classes*2, num_classes)
        )

    def initialize_weights(self):
      """
      Initializes weights of the CombinationLayer using He initialization.
      Applies He initialization to weights of Conv2d and Linear layers 
      and sets biases to zero.
      """
      for layer in self.CombinationLayer.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
          nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
          if layer.bias is not None:
              nn.init.constant_(layer.bias, 0)

    def forward(self, dynamic_data, global_embeddings, ind_start_seq, ind_end_seq):
        """ 
        Forward pass

        Args:
            dynamic_data: dynamic features represented as a tensor
            global_embeddings: embeddings generated using ESM for each protein
            ind_start_seq: first 20 amino acids of each sequence
            ind_end_seq: last 20 amino acids of each sequence

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, timesteps, num_classes) 
                          containing predicted class scores for each sequence position
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