import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    """ Temporal Convolution Network Class for dynamic feature extraction which takes as input data in the form [batch_size, timesteps, features]
    Args:
        input_dim: number of features
        intermediate_dim: number of channels in the hidden layer
        output_dim: number of output features (in our case equal to number of classes for classification)
        kernel_size: size of convolutional layers
        stride: how far the kernel moves across the input
        dilation_1: spacing between kernel elements in first layer
        dilation_2: spacing between kernel elements in second layer
        dropout: dropout probability (regularization technique)

    Methods:
		initialize_weights: He weights initialization for all the layers
        forward: usual forward pass of neural networks
    """

    def __init__(self, input_dim=17, intermediate_dim=15, output_dim=15, kernel_size=3, stride=1, dilation_1 = 1, dilation_2 = 2, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # 2 layers of TCN
        self.conv1 = nn.Conv1d(input_dim, intermediate_dim, kernel_size, stride=stride, padding=(kernel_size-1)*dilation_1, dilation=dilation_1)
        self.layer_norm1 = nn.LayerNorm(intermediate_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(intermediate_dim, output_dim, kernel_size, stride=stride, padding=(kernel_size-1)*dilation_2, dilation=dilation_2)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

    def initialize_weights(self):
        for layer in self.modules():
          if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """ Forward pass of the model
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, timesteps, input_size)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, timesteps, output_dim)
"""

        x = x.transpose(1, 2)  # Change to (batch_size, features, timesteps) for Conv1D
        x = self.conv1(x)
        x = x[:, :, :-self.conv1.padding[0]]  # Discard last part to maintain sequence length
        x = self.layer_norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = x[:, :, :-self.conv2.padding[0]]  # Discard last part to maintain sequence length
        x = self.layer_norm2(x.transpose(1, 2)).transpose(1, 2) # Layer Normalization (which takes data in form (batch_size, timesteps, features))
        x = self.relu2(x)
        x = self.dropout2(x)
        x = x.transpose(1, 2)  # Back to (batch_size, timesteps, output_dim)
        return x