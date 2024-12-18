import torch
import pandas as pd
from trainers.training_utils import validate, run_training
from sklearn.model_selection import KFold
import itertools
from configs.dataloaders import create_data_loaders


def k_fold_cross_validation(
    model_class,
    dataset,
    criterion,
    parameter_grid,
    num_epochs,
    patience,
    batch_size,
    device,
    k_folds=5,
    lambda_penalty=0.0,
    seed=32,
    cross_validation_on_loss=True,
    verbose=True
):
    """
    Performs k-fold cross-validation.

    Args:
        model_class: PyTorch model class (not an instance)
        dataset: PyTorch Dataset object
        criterion: Loss function
        parameter_grid: Dictionary with hyperparameters and their possible values
        num_epochs: Number of epochs for each fold
        patience: number of consecutive epochs allowed for test loss to not improve before stopping the training
        batch_size: number of samples in each batch
        device: Device to train on ('cuda' or 'cpu')
        k_folds: Number of folds for cross-validation
        lambda_penalty: Regularization coefficient
        seed: random number generator seed
        cross_validation_on_loss: boolean to decide on which metric to do cross validation
        verbose: Print training progress

    Returns:
        best_params: Dictionary of the best hyperparameters
        results: List of performance results for all hyperparameter combinations
    """

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    all_params = list(itertools.product(*parameter_grid.values()))
    param_names = list(parameter_grid.keys())

    results = []

    # Iterate through all combinations of hyperparameters
    for param_set in all_params:
        params = dict(zip(param_names, param_set))
        avg_val_loss = 0.0
        avg_val_accuracy = 0.0

        if verbose:
            print(f"Testing hyperparameters: {params}")

        # K-Fold Cross-Validation
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            if verbose:
                print(f"Fold {fold+1}/{k_folds}")

            # Split dataset into training and validation
            data_train = dataset.loc[train_idx]
            data_test = dataset.loc[val_idx]

            # Create DataLoaders
            train_loader, test_loader = create_data_loaders(data_train, data_test, batch_size)

            # Initialize the model
            model = model_class(dropout=params['dropout']).to(device)
            model.init_weights()

            # Initialize optimizer and scheduler
            optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
            # Initialize scheduler based on the passed parameter
            scheduler_type = params['scheduler']
            if scheduler_type == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
            elif scheduler_type == 'StepLR':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            elif scheduler_type == 'ExponentialLR':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")


            # Train and validate the model using run_training
            _, _, val_loss, val_acc = run_training(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                lambda_penalty=lambda_penalty,
                dataloader_train=train_loader,
                dataloader_test=test_loader,
                num_epochs=num_epochs,
                patience=patience,
                device=device,
                verbose=False
            )

            # Use the last epoch's validation metrics
            avg_val_loss += val_loss[-1]
            avg_val_accuracy += val_acc[-1]

        # Average metrics across folds
        avg_val_loss /= k_folds
        avg_val_accuracy /= k_folds

        if verbose:
            print(f"Hyperparameters: {params}, Average Validation Loss: {avg_val_loss:.4f}, "
                  f"Average Validation Accuracy: {avg_val_accuracy:.2f}%\n")

        # Save results
        results.append({
            'params': params,
            'avg_val_loss': avg_val_loss,
            'avg_val_accuracy': avg_val_accuracy
        })

    # Find the best hyperparameters based on validation loss
    if cross_validation_on_loss == True:
        best_result = min(results, key=lambda x: x['avg_val_loss'])
    else:
        best_result = min(results, key=lambda x: x['avg_val_accuracy'])
    best_params = best_result['params']

    print(f"Best Hyperparameters: {best_params}, Validation Loss: {best_result['avg_val_loss']:.4f}, "
          f"Validation Accuracy: {best_result['avg_val_accuracy']:.2f}%")

    return best_params, results