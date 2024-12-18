import torch
import torch.nn as nn

class StaticModelMLP(nn.Module):
    """ 
    Model based on a MLP that takes as input the proteins embeddings and returns predictions on the classes

    Attributes:
        input_dim (int): Dimension of the input embeddings
        inner (nn.Sequential): Multilayer Perceptron
    
    Methods:
		forward : forward pass of the method
        initialize_weights: applies He initialization to all the linear layers
    """
	
    def __init__(self, input_size=640, intermediate_dim = 64, num_classes=15, dropout=0.2):
        """ 
        Class constructor
        Args:
            input_size (int): dimension of the embeddings given as input
            num_classes (int): number of classes for the prediction
        """
        super(StaticModelMLP, self).__init__()

        self.input_dim = input_size

        self.inner = nn.Sequential(
            nn.Linear(input_size, intermediate_dim), # First layer, reduces the dimension from input_size to 256
            nn.LayerNorm(intermediate_dim, eps=1e-05),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, num_classes)
        )



    def forward(self, x):
        """ 
        Forward pass of the model

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes), 
                          containing the predicted class scores for each sequence position
        """
        return self.inner(x)

    def initialize_weights(self):
        """ 
        Applies initialization for weights, with the following algorithms:
            - He (Kaiming) uniform initialization for Linear layers
            - Zero initialization for biases
        """
        for layer in self.inner:
          if isinstance(layer, nn.Linear):
              nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
              if layer.bias is not None:
                  nn.init.zeros_(layer.bias)