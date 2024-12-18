import torch
import torch.nn as nn

class LSTMDynamicModel(nn.Module):
    """ 
    Model based on a MLP that takes as input the proteins embeddings and returns predictions on the classes

    Attributes:
        input_dim (int): Dimension of the input embeddings
        inner (nn.Sequential): Multilayer Perceptron
    
    Methods:
		forward : forward pass of the method
        initialize_weights: applies He initialization to all the linear layers
    """

    def __init__(self, static_model, static_learnable=False, num_timesteps=5, num_classes=15, num_features = 34, hidden_size = 64, num_layers=2, dropout=0.2):
        super(LSTMDynamicModel, self).__init__()
        """
        Class constructor
        Args:
            static_model (nn.Module): Pretrained static model providing global embeddings
            static_learnable (bool): Whether to allow fine-tuning of the static model
            num_timesteps (int): Number of timesteps in the dynamic data
            num_classes (int): Number of target classes for the output
            num_features (int): Number of features in the input to the LSTM
            hidden_size (int): Size of the hidden state in the LSTM
            num_layers (int): Number of layers in the LSTM
            dropout (float): Dropout probability for LSTM and attention layers
        """

        # Define attributes
        self.StaticModel = static_model
        self.softmax = nn.Softmax(1)
        self.num_timesteps = num_timesteps

        # If required, freeze static model weights
        if static_learnable == False:
            for param in self.StaticModel.parameters():
                param.requires_grad = False
   
        # LSTM layer
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=2*hidden_size, num_heads=4, batch_first=True, dropout=dropout)

        # Output Layer
        self.output_layer = nn.Linear(2*hidden_size, num_classes) # 2*hidden size since bidirectional architecture

    def forward(self, dynamic_data, global_embeddings, start_sequences, end_sequences):
        

        # Pass all required inputs to the static model
        with torch.no_grad():
          static_features = self.StaticModel(global_embeddings, start_sequences, end_sequences)
          static_features = self.softmax(static_features)

        # Static representation
        static_representation = static_features.unsqueeze(1).repeat(1, self.num_timesteps, 1) # Expand static dataset to match dynamic dimensions [batch_size, num_timesteps, num_classes]

        # Combine static and dynamic inputs
        combined_inputs = torch.cat([static_representation, dynamic_data], dim=2)

        # Process through the LSTM
        lstm_out, _ = self.lstm(combined_inputs)
        attn_output, _ = self.multihead_attention(lstm_out, lstm_out, lstm_out)

        # Compute predictions for each phase
        predictions = self.output_layer(attn_output)

        return predictions