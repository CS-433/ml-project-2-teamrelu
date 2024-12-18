import torch
import torch.nn as nn

# Interactions, Protein concentrations, Concentration variation

class ModulableLSTMDynamicModel(nn.Module):
    """ Model based on a MLP that takes as input the proteins embeddings and returns predictions on the classes

    Attributes:
        input_dim (int): Dimension of the input embeddings
        inner (nn.Sequential): Multilayer Perceptron
    
    Methods:
		forward : forward pass of the method
        initialize_weights: applies He initialization to all the linear layers
    """

    def __init__(self, static_model, static_learnable=False, num_timesteps=5, num_classes=15, num_features = 34, hidden_size = 64, num_layers=2, dropout=0.2, no_concentration=False, no_interaction=False, no_static=False):
        super(LSTMDynamicModel, self).__init__()

        # Define attributes
        self.StaticModel = static_model
        self.softmax = nn.Softmax(1)
        self.num_timesteps = num_timesteps
        self.num_features = num_features
        self.no_concentration = no_concentration
        self.no_interaction = no_interaction
        self.no_static = no_static
        self.static_learnable = static_learnable

        if self.no_concentration:
            self.num_features -= 4
        if self.no_interaction:
            self.num_features -= 15
        if self.no_static:
            self.num_features -= 15

        # If required, freeze static model weights
        if static_learnable == False:
            for param in self.StaticModel.parameters():
                param.requires_grad = False
   
        # LSTM layer
        self.lstm = nn.LSTM(self.num_features, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=2*hidden_size, num_heads=4, batch_first=True, dropout=dropout)

        # Output Layer
        self.output_layer = nn.Linear(2*hidden_size, num_classes) # 2*hidden size since bidirectional architecture

    def forward(self, dynamic_data, global_embeddings, start_sequences, end_sequences):
        # Pass all required inputs to the static model
        if not self.no_static:
            if not self.static_learnable:
                with torch.no_grad():
                    static_features = self.StaticModel(global_embeddings, start_sequences, end_sequences)
                    static_features = self.softmax(static_features)
            else:
                static_features = self.StaticModel(global_embeddings, start_sequences, end_sequences)
                static_features = self.softmax(static_features)

            # Static representation
            static_representation = static_features.unsqueeze(1).repeat(1, self.num_timesteps, 1) # Expand static dataset to match dynamic dimensions [batch_size, num_timesteps, num_classes]

        if not self.no_concentration:
            # Use concentration features to calculate concentration variations as additional features
            concentration_TE = dynamic_data[:, :, 15]
            concentration_TL = dynamic_data[:, :, 16]
            delta_TE = concentration_TE[:, 1:] - concentration_TE[:, :-1]
            delta_TL = concentration_TL[:, 1:] - concentration_TL[:, :-1]

            # Padding to keep original dimension
            delta_TE = torch.cat([torch.zeros(delta_TE.size(0), 1), delta_TE], dim=1)
            delta_TL = torch.cat([torch.zeros(delta_TL.size(0), 1), delta_TL], dim=1)
            delta_TE = delta_TE.unsqueeze(-1)
            delta_TL = delta_TL.unsqueeze(-1)

            # Add a feature dimension to dynamic inputs
            dynamic_inputs = torch.cat([
                dynamic_data,
                delta_TE,
                delta_TL
            ], dim=2)

        if self.no_concentration:
            dynamic_inputs = dynamic_data[:,:,:15]

        if self.no_interaction:
            dynamic_inputs = dynamic_inputs[:,:,15:]

        if self.no_static:
            combined_inputs = dynamic_inputs
        else:
        # Combine static and dynamic inputs
            combined_inputs = torch.cat([static_representation, dynamic_inputs], dim=2)

        # Process through the LSTM
        lstm_out, _ = self.lstm(combined_inputs)
        attn_output, _ = self.multihead_attention(lstm_out, lstm_out, lstm_out)

        # Compute predictions for each phase
        predictions = self.output_layer(attn_output)

        print(combined_inputs.shape)

        return predictions