import torch
import torch.nn as nn

class StaticModelTransformer(nn.Module):
    """ Model based on transformers that takes as input the proteins embeddings and returns predictions on the classes

    Attributes:
        input_dim (int): Dimension of the input embeddings
        encoder (nn.TransformerEncoder): Stacked Transformer encoder layers
        fc (nn.Linear): Final classification layer
    
    Methods:
		forward : forward pass of the method
        initialize_weights: applies initialization to all layers ()
    """
    def __init__(self, input_size=640, num_heads=4, num_layers=2, num_classes=15, ff_dim=512):
        """ Class constructor
        Args:
            input_size (int): Dimension of the input embeddings
            num_heads (int): Number of attention heads in the multi-head attention mechanism
            num_layers (int): Number of layers in the Transformer encoder
            num_classes (int): Number of output classes for prediction
            ff_dim (int): Dimension of the feedforward network within the Transformer layers
        """

        super(StaticModelTransformer, self).__init__()
        
        self.input_dim = input_size

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, 
            nhead=num_heads,  
            dim_feedforward=ff_dim,
            activation='relu', 
            batch_first=True    # Ensures (batch, feature) input format
        )

        # Stack multiple layers
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers  # Number of layers
        )

        # Output layer for classification
        self.fc = nn.Linear(input_size, num_classes) 

    def forward(self, x):
        """ Forward pass of the model

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes), 
                          containing the predicted class scores for each sequence position
        """

        encoded = self.encoder(x)
        output = self.fc(encoded)
        return output
    
    def initialize_weights(self):
        """ Applies initialization for weights, with the following algorithms:
            - Xavier uniform initialization for Linear layers
            - Zero initialization for biases
            - Standard initialization for LayerNorm
            - Default torch initialization for Transformer layers
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for weights
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # Standard initialization for LayerNorm
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.TransformerEncoderLayer):
                # Reset parameters of the Transformer layer to PyTorch default
                module.reset_parameters()