import torch
import torch.nn as nn

class StaticModelMultibranch(nn.Module):
    """ Model based on MLP and Transformers that takes as input the proteins embeddings and the ending parts 
        of their sequences (mapped to int sequences with a vocabulary) and returns predictions on the classes. 

    Attributes:
        global_dense (nn.Sequential): Layers for processing the global protein embedding
        char_embedding (nn.Embedding): Embedding layer for the character-level protein sequence extremities
        positional_embedding (nn.Parameter): Learnable positional embeddings for sequence positions
        transformer (nn.TransformerEncoder): Transformer encoder that processes the sequence embeddings
        seq_dense (nn.Linear): Dense layer applied to the output of the Transformer
        fc (nn.Sequential): Final classification layers that combines the previous ones

    Methods:
        process_sequence: Processes extremities sequences embeddings using the character-level embedding and transformer
        forward: forward pass of the model
        init_weights: Applies weight initialization to all layers of the model
    """

    def __init__(self, num_classes=15, embedding_dim=640, extremities_dim=20, char_vocab_size=20, char_embed_dim=16, intermediate_dim=32, dropout=0.3):
        """ Class constructor
        Args:
            num_classes (int): Number of output classes for the classification.
            embedding_dim (int): Dimension of the global protein embedding
            extremities_dim (int): Length of the sequence extremities
            char_vocab_size (int): Number of unique characters in the sequence
            char_embed_dim (int): Dimension of the character-level extremities embeddings
            intermediate_dim (int): Dimension of the output of the initial branches
            dropout (float): Dropout rate
        """
        super(StaticModelMultibranch, self).__init__()

        # Layer that takes global protein embedding
        self.global_dense = nn.Sequential(
            nn.Linear(embedding_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim, eps=1e-05),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Character-level embedding layer and positional embeddings for sequence extremities
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(extremities_dim, char_embed_dim))

        # Transformer that takes embeddings of sequence extremities
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=char_embed_dim,
                nhead=2,
                dim_feedforward = 16,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=1,
        )

        # Dense layer for sequence extremities after transformer processing
        self.seq_dense = nn.Linear(char_embed_dim, intermediate_dim)

        # Final classifier
        self.fc = nn.Sequential(
            nn.Linear(intermediate_dim * 3, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, num_classes)
        )

    def process_sequence(self, x):
        """ Processes sequence extremities using character-level embeddings and the transformer encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length) representing the sequence.

        Returns:
            torch.Tensor: Processed sequence features after transformer and dense layer.
        """
        x = self.char_embedding(x)
        x = x + self.positional_embedding.unsqueeze(0) #Adds a 
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.seq_dense(x)
        return x

    def forward(self, global_input, ind_start_seq, ind_end_seq):
        """ Forward pass of the model. Processes global embedding and sequence embeddings, 
        then combines features for final classification.

        Args:
            global_input (torch.Tensor): Global protein embedding tensor of shape (batch_size, embedding_dim)
            ind_start_seq (torch.Tensor): Tensor representing the start sequence positions (batch_size, extremities_dim)
            ind_end_seq (torch.Tensor): Tensor representing the end sequence positions (batch_size, extremities_dim)

        Returns:
            torch.Tensor: Predicted class scores for each input in the batch.
        """
         
        # Process global embedding
        global_features = self.global_dense(global_input)

        # Process start and end sequences
        start_features = self.process_sequence(ind_start_seq)
        end_features = self.process_sequence(ind_end_seq)

        # Concatenate all branches
        combined = torch.cat([global_features, start_features, end_features], dim=1)

        # Final classification
        output = self.fc(combined)
        return output

    def init_weights(self):
        """ Applies initialization for weights, with the following algorithms:
        - He initialization for Linear layers.
        - Xavier initialization for Transformer attention layers.
        - Uniform initialization for Embedding layers.
        """
        # Apply He initialization to all Linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')  # He initialization
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.TransformerEncoderLayer):
                # Apply Xavier initialization to the attention layers of the Transformer
                nn.init.xavier_normal_(module.self_attn.in_proj_weight)  
                nn.init.xavier_normal_(module.self_attn.out_proj.weight)  
                if module.self_attn.in_proj_bias is not None:
                    nn.init.zeros_(module.self_attn.in_proj_bias) 

                # Apply He initialization for the feedforward layers (within the TransformerEncoderLayer)
                nn.init.kaiming_normal_(module.linear1.weight, mode='fan_out', nonlinearity='relu') 
                nn.init.kaiming_normal_(module.linear2.weight, mode='fan_out', nonlinearity='relu')
                if module.linear1.bias is not None:
                    nn.init.zeros_(module.linear1.bias)
                if module.linear2.bias is not None:
                    nn.init.zeros_(module.linear2.bias)

            # Apply uniform initialization to Embedding layers
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.1, 0.1)