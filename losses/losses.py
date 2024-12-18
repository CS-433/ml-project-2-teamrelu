import torch
import torch.nn as nn

def CrossEntropy(outputs, targets):
    """
    Computes standard CrossEntropy loss
        Args:
            outputs (torch.Tensor): Predicted logits
            targets (torch.Tensor): Ground-truth labels
        Returns:
            torch.Tensor: CrossEntropy loss value
    """

    return nn.CrossEntropyLoss()(outputs, targets)


def CrossEntropyWithTemporalSmoothness(outputs, targets, params):
    """
    Computes CrossEntropy loss with a penalty for temporal inconsistency
        Args:
            outputs (torch.Tensor): Predicted logits
            targets (torch.Tensor): Ground-truth labels
            params (list): [lambda_penalty (float), num_timesteps (int)] for smoothness penalty
        Returns:
            torch.Tensor: Combined loss value
    """
    ce_loss = nn.CrossEntropyLoss()(outputs, targets)
    # Compute temporal smoothness penalty
    lambda_penalty = params[0]
    num_timesteps = params[1]
    outputs = outputs.view(outputs.size(0), num_timesteps, -1)
    
    temporal_diff = outputs[:, 1:, :] - outputs[:, :-1, :]  # Difference between consecutive timesteps DA SISTEMARE
    smoothness_penalty = torch.mean(temporal_diff.pow(2))  # L2 penalty

    # Combine loss terms
    return ce_loss + lambda_penalty * smoothness_penalty

def CrossEntropyWithLasso(outputs, targets, lambda_penalty, model):
    """
    Computes CrossEntropy loss with L1 regularization on model parameters
        Args:
            outputs (torch.Tensor): Predicted logits
            targets (torch.Tensor): Ground-truth labels
            lambda_penalty (float): Regularization strength
            model (torch.nn.Module): Model containing parameters to regularize
        Returns:
            torch.Tensor: Combined loss value
    """
    ce_loss = nn.CrossEntropyLoss()(outputs, targets)
    # Compute lasso term
    lasso_reg = sum(torch.abs(param).sum() for param in model.parameters())
    # Combine loss terms
    return ce_loss + lambda_penalty * lasso_reg

# Dictionary to map loss names (strings) to functions
loss_functions_map = {
    "CrossEntropy": CrossEntropy,
    "CrossEntropyWithTemporalSmoothness": CrossEntropyWithTemporalSmoothness,
    "CrossEntropyWithLasso": CrossEntropyWithLasso
}