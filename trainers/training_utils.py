import torch
import torch.nn as nn
from losses.losses import CrossEntropy, CrossEntropyWithTemporalSmoothness, CrossEntropyWithLasso
from inspect import signature


def validate(model, dataloader_test, device):
    """ Validation loop
        Args:
            model: PyTorch model to be evaluated
            dataloader_test: DataLoader object that provides an iterable over the test dataset
            device: the device (CPU or GPU) on which the model and data are loaded
            criterion: loss function used to compute the validation loss
            lambda_penalty: loss regularization coefficient (depending on loss used)
        Returns:
            test_loss: epoch loss on test set
            test_accuracy: epoch accuracy on test set
    """
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Initialize values for computing accuracy and loss for each epoch
        correct_predictions_test=0
        total_predictions_test=0
        epoch_loss_test=0

        for batch in dataloader_test:
            # Load batch data and labels
            dynamic_data, global_input, ind_start_seq, ind_end_seq, static_labels, dynamic_labels = batch
            # Move data to device
            dynamic_data = dynamic_data.to(device)
            global_input = global_input.to(device)
            ind_start_seq = ind_start_seq.to(device)
            ind_end_seq = ind_end_seq.to(device)
            static_labels = static_labels.to(device)
            dynamic_labels = dynamic_labels.to(device)

            # Forward pass (different for static or dynamic training, distinguish with parameters taken as input)
            if len(signature(model.forward).parameters) == 4:
                outputs = model(dynamic_data, global_input, ind_start_seq, ind_end_seq)
                labels = dynamic_labels
            elif len(signature(model.forward).parameters) == 3:
                outputs = model(global_input, ind_start_seq, ind_end_seq)
                labels = static_labels

            # Change output dimensions for computing predicions
            outputs = outputs.view(-1, outputs.size(-1))  # [batch_size * seq_length, num_classes]
            labels = labels.view(-1)  # [batch_size * seq_length]
            # Compute batch loss and add to the total loss (no regularization terms added for validation loss)
            loss_test = CrossEntropy(outputs, labels)

            epoch_loss_test += loss_test.item()
            # Count exact values and add to the total exact values
            _, predicted_test = torch.max(outputs, 1)
            correct_predictions_test += (predicted_test == labels).sum().item()
            total_predictions_test += labels.size(0)

        test_accuracy = correct_predictions_test / total_predictions_test * 100
        test_loss = epoch_loss_test / len(dataloader_test)

    # Return test loss and accuracy
    return test_loss, test_accuracy

def run_training(model, criterion, optimizer, scheduler, lambda_penalty, dataloader_train, dataloader_test, num_epochs, patience, device, verbose):
    """Training loop
        Args:
            model: PyTorch model to be evaluated
            criterion: loss function used to compute the validation loss
            optimizer: optimizer used to update the model's parameters
            scheduler: learning rate scheduler to adjust the learning rate during training
            lambda_penalty: loss regularization coefficient (depending on loss used)
            dataloader_train: DataLoader object that provides an iterable over the training dataset
            dataloader_test: DataLoader object that provides an iterable over the test dataset
            num_epochs: total number of epochs (full passes over the training dataset)
            patience: number of consecutive epochs allowed for test loss to not improve before stopping the training
            device: the device (CPU or GPU) on which the model and data are loaded
        Return:
            train_loss_history: list with train loss for each epoch
            train_accuracy_history: list with train accuracy for each epoch
            test_loss_history: list with test loss for each epoch
            test_accuracy_history: list with test accuracy for each epoch
    """
    # Ensure model is on the right device and set for training
    model = model.to(device)
    # Initialize vectors for loss and accuracy history
    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    # Inizialize variables for early stopping
    best_test_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Set model to train
        model.train()
        # Initialize values for computing accuracy and loss for each epoch
        epoch_loss = 0.0
        correct_predictions=0
        total_predictions=0

        for batch in dataloader_train:
            # Load batch data and labels
            dynamic_data, global_input, ind_start_seq, ind_end_seq, static_labels, dynamic_labels = batch
            # Move data to device
            dynamic_data = dynamic_data.to(device)
            global_input = global_input.to(device)
            ind_start_seq = ind_start_seq.to(device)
            ind_end_seq = ind_end_seq.to(device)
            static_labels = static_labels.to(device)
            dynamic_labels = dynamic_labels.to(device)

            # Forward pass (different for static or dynamic training, distinguish with parameters taken as input)
            if len(signature(model.forward).parameters) == 4:
                outputs = model(dynamic_data, global_input, ind_start_seq, ind_end_seq)
                labels = dynamic_labels
                num_timesteps = labels.size(1)
            elif len(signature(model.forward).parameters) == 3:
                outputs = model(global_input, ind_start_seq, ind_end_seq)
                labels = static_labels
            else:
                # If the number of parameters is not 3, or 4, raise an error
                raise ValueError(f"Unsupported number of parameters in the metod function. Expected 3, or 4, but got {len(signature(model).parameters)}.")

            # Reshape outputs and labels for accuracy computation in the case of dynamic data dimensions (i.e. with timesteps dimension)
            outputs = outputs.view(-1, outputs.size(-1))  # [batch_size * seq_length, num_classes]
            labels = labels.view(-1)  # [batch_size * seq_length]

            # Count exact values and add to the total exact values
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            # Compute loss
            # Compute batch loss and add to the total loss (handle different losses based on number of parameters required)
            if len(signature(criterion).parameters) == 4:
                loss = criterion(outputs, labels, lambda_penalty, model)
            elif len(signature(criterion).parameters) == 3:
                params = [lambda_penalty, num_timesteps]
                loss = criterion(outputs, labels, params)
            elif len(signature(criterion).parameters) == 2:
                loss = criterion(outputs, labels)
            else:
                # If the number of parameters is not 2, 3, or 4, raise an error
                raise ValueError(f"Unsupported number of parameters in the criterion function. Expected 2, 3, or 4, but got {len(signature(criterion).parameters)}.")

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Add batch loss to the total epoch loss
            epoch_loss += loss.item()

        # Scheduler step if required
        if scheduler:
            scheduler.step()
                # Save current learning rate to be displayed
            current_lr = scheduler.get_last_lr()[-1]
        else:
            current_lr = optimizer.param_groups[0]['lr']
        # Compute train accuracy
        train_accuracy = correct_predictions / total_predictions * 100

        # Check how the model do on the test set after each epoch
        test_loss, test_accuracy = validate(model, dataloader_test, device)

        train_loss_history.append(epoch_loss/len(dataloader_train))
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        # Check for early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict()  # Save the best model weights
            early_stop_counter = 0  # Reset counter
        else:
            early_stop_counter += 1  # Increment counter if no improvement


        # Display all the useful results of the epoch if verbose option set to true
        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}],\tTraining Loss: {epoch_loss/len(dataloader_train):.4f}, Training accuracy: {train_accuracy:.2f},\tTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f},\tLearning rate: {current_lr:.5f}")
        
        # Break training if patience is exhausted
        if early_stop_counter >= patience:
            if verbose:
                print(f"Early stopping triggered after {epoch+1} epochs. Best test loss: {best_test_loss:.4f}")
            break

    # Load best model weights before returning
    model.load_state_dict(best_model_state)

    return train_loss_history, train_accuracy_history, test_loss_history, test_accuracy_history