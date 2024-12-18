import torch
import torch.nn as nn

def CrossEntropy(outputs, targets):
    """Loss function 1: Standard CrossEntropy loss"""
    return nn.CrossEntropyLoss()(outputs, targets)

def CrossEntropyWithTemporalSmoothness(outputs, targets, params):
    """Loss function 2: CrossEntropy with penalty for predictions which change a lot along time"""
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
    """Loss function 3: Categorical Cross-Entropy with Lasso (L1) Regularization"""
    ce_loss = nn.CrossEntropyLoss()(outputs, targets)
    # Compute lasso term
    lasso_reg = sum(torch.abs(param).sum() for param in model.parameters())

    # Combine loss terms
    return ce_loss + lambda_penalty * lasso_reg