import torch
import pandas as pd
from trainers.training_utils import validate, run_training
from sklearn.model_selection import KFold
import itertools
from configs.dataloaders import create_data_loaders
from models.dynamic_models.temporal_block import TemporalBlock
from losses.losses import CrossEntropy


def k_fold_cross_validation_static(
    model_class,
    dataset,
    criterion,
    parameter_grid,
    num_epochs,
    patience,
    batch_size,
    device,
    k_folds=5,
    seed=32,
    cross_validation_on_loss=True,
    verbose=True
):
    """
    Performs k-fold cross-validation.

    Args:
        model_class (Type[nn.Module]): PyTorch model class for static problem (not an instance)
        dataset (torch.utils.data.Dataset): PyTorch Dataset object
        criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Loss function
        parameter_grid (Dict[str, List[Any]]): Dictionary with hyperparameters and their possible values
        num_epochs (int): Number of epochs for each fold
        patience (int): Number of consecutive epochs allowed for test loss to not improve before stopping the training
        batch_size (int): Number of samples in each batch
        device (str): Device to train on ('cuda' or 'cpu')
        k_folds (int): Number of folds for cross-validation
        seed (int): Random number generator seed
        cross_validation_on_loss (bool): Boolean to decide on which metric to do cross-validation
        verbose (bool): Print training progress

    Returns:
        Tuple[Dict[str, Any], List[float]]:
            - best_params (Dict[str, Any]): Dictionary of the best hyperparameters
            - results (List[float]): List of performance results for all hyperparameter combinations
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
            model.initialize_weights()
            
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
                lambda_penalty=0,
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


def k_fold_cross_validation_dynamic(
    static_model_class,
    dynamic_model_class,
    dataset,
    criterion,
    parameter_grid,
    static_params,
    num_epochs,
    patience,
    batch_size,
    device,
    k_folds=5,
    seed=32,
    cross_validation_on_loss=True,
    verbose=True
):
    """
    Performs k-fold cross-validation.

    Args:
        static_model_class (Type[nn.Module]): PyTorch model class for static problem (not an instance)
        dynamic_model_class (Type[nn.Module]): PyTorch model class for dynamic problem (not an instance)
        dataset (torch.utils.data.Dataset): PyTorch Dataset object
        criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Loss function
        parameter_grid (Dict[str, List[Any]]): Dictionary with hyperparameters and their possible values
        static_params (Dict[str, Any]): Dictionary with best parameters of static model
        num_epochs (int): Number of epochs for each fold
        patience (int): Number of consecutive epochs allowed for test loss to not improve before stopping the training
        batch_size (int): Number of samples in each batch
        device (str): Device to train on ('cuda' or 'cpu')
        k_folds (int): Number of folds for cross-validation
        seed (int): Random number generator seed
        cross_validation_on_loss (bool): Boolean to decide on which metric to do cross-validation
        verbose (bool): Print training progress

    Returns:
        Tuple[Dict[str, Any], List[float]]:
            - best_params (Dict[str, Any]): Dictionary of the best hyperparameters
            - results (List[float]): List of performance results for all hyperparameter combinations
    """

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    all_params = list(itertools.product(*parameter_grid.values()))
    param_names = list(parameter_grid.keys())

    results = []
    i = 0
    # Iterate through all combinations of hyperparameters
    for param_set in all_params:
        i += 1
        params = dict(zip(param_names, param_set))
        avg_val_loss = 0.0
        avg_val_accuracy = 0.0

        if verbose:
            print(f"Testing hyperparameters: {params},\n set: {i}")

        # K-Fold Cross-Validation
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            if verbose:
                print(f"Fold {fold+1}/{k_folds}")

            # Split dataset into training and validation
            data_train = dataset.loc[train_idx]
            data_test = dataset.loc[val_idx]

            # Create DataLoaders
            train_loader, test_loader = create_data_loaders(data_train, data_test, batch_size)

            # Train static model with its best parameters
            static_model = static_model_class(dropout=static_params['dropout']).to(device)
            static_model.initialize_weights()
            static_optimizer = torch.optim.AdamW(static_model.parameters(), lr=static_params['learning_rate'], weight_decay=static_params['weight_decay'])
            static_scheduler_type = static_params['scheduler']
            if static_scheduler_type == 'CosineAnnealingLR':
                static_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(static_optimizer, T_max=100, eta_min=1e-5)
            elif static_scheduler_type == 'StepLR':
                static_scheduler = torch.optim.lr_scheduler.StepLR(static_optimizer, step_size=30, gamma=0.1)
            elif static_scheduler_type == 'ExponentialLR':
                static_scheduler = torch.optim.lr_scheduler.ExponentialLR(static_optimizer, gamma=0.95)
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")
            
            _, _, _, _ = run_training(
                model=static_model,
                criterion=CrossEntropy,
                optimizer=static_optimizer,
                scheduler=static_scheduler,
                lambda_penalty=0,
                dataloader_train=train_loader,
                dataloader_test=test_loader,
                num_epochs=num_epochs,
                patience=patience,
                device=device,
                verbose=False
            )
            
            # Create dynamic model
            temporal_block = TemporalBlock(dropout=params['dropout'])
            temporal_block.initialize_weights()
            # Create dynamic model and initialize its weights
            #dynamic_model = TCNDynamicModel(static_model, temporal_block, static_learnable=False, num_timesteps=num_timesteps, num_classes=num_classes)
            #dynamic_model.initialize_weights()
            dynamic_model = dynamic_model_class(static_model, static_learnable=params['static_learnable'], hidden_size=params['hidden_size'], num_layers=2, dropout=params['dropout'])

            # Initialize optimizer and scheduler
            optimizer = torch.optim.AdamW(dynamic_model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
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
                model=dynamic_model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                lambda_penalty= params['lambda_penalty'],
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