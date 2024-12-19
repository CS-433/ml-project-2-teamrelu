import torch
import torch.nn as nn


class NaiveModel(nn.Module):
    """
    A naive neural network model that processes static and dynamic input features.
        
    Attributes:
            num_timesteps (int): The number of timesteps in the dynamic input.
            dense1 (nn.Linear): A fully connected layer that reduces the input dimensionality to 64.
            relu (nn.ReLU): The ReLU activation function.
            dense2 (nn.Linear): A fully connected layer that maps the features to the number of classes.
    """
    def __init__(self, input_dim=531, num_classes=15, num_timesteps=5):
        super(NaiveModel, self).__init__()

        """
        Initializes the NaiveModel with given parameters.
        
        Args:
            input_dim (int): The dimensionality of the input features.
            num_classes (int): The number of output classes.
            num_timesteps (int): The number of timesteps in the dynamic input.
        """

        self.num_timesteps=num_timesteps
        self.dense1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(64, num_classes)

    def forward(self, x_static, x_dynamic):
        """
        Forward pass of the NaiveModel.

        Args:
            x_static (torch.Tensor): A tensor containing static features of shape (batch_size, input_dim).
            x_dynamic (torch.Tensor): A tensor containing dynamic features of shape 
                                       (batch_size, num_timesteps, input_dim).

        Returns:
            torch.Tensor: The output logits of shape (batch_size, num_timesteps, num_classes).
        """
        x_static = x_static.unsqueeze(1).repeat(1, self.num_timesteps, 1)
        x = torch.cat([x_static, x_dynamic], dim=2)
        x = self.relu(self.dense1(x))
        x = self.dense2(x)
        return x
 
