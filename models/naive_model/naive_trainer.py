import torch
import torch.nn as nn
from losses.losses import CrossEntropy

def run_training(model, optimizer, dataloader_train, dataloader_test, num_epochs, device, verbose):
    """
    Training loop

    Args:
        model (nn.Module): PyTorch model to be trained
        optimizer (torch.optim.Optimizer): Optimizer used to update the model's parameters
        dataloader_train (torch.utils.data.DataLoader): DataLoader object that provides an iterable over the training dataset
        dataloader_test (torch.utils.data.DataLoader): DataLoader object that provides an iterable over the test dataset
        num_epochs (int): Total number of epochs (full passes over the training dataset)
        device (str): The device (CPU or GPU) on which the model and data are loaded
        verbose (bool): Print progresses if set to true

    Returns:
        train_loss_history (List[float]): List with train loss for each epoch
        train_accuracy_history (List[float]): List with train accuracy for each epoch
        test_loss_history (List[float]): List with test loss for each epoch
        test_accuracy_history (List[float]): List with test accuracy for each epoch
    """
    # Ensure model is on the right device and set for training
    model = model.to(device)
    # Initialize vectors for loss and accuracy history
    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    for epoch in range(num_epochs):
        # Set model to train
        model.train()
        # Initialize values for computing accuracy and loss for each epoch
        epoch_loss = 0.0
        correct_predictions=0
        total_predictions=0

        for batch in dataloader_train:
            # Load batch data and labels
            sequences, dynamic_data, labels = batch
            # Move data to device
            sequences = sequences.to(device)
            labels = labels.to(device)
            dynamic_data = dynamic_data.to(device)
            outputs = model(sequences, dynamic_data)
            num_timesteps = labels.size(1)
            
            # Reshape outputs and labels for accuracy computation in the case of dynamic data dimensions (i.e. with timesteps dimension)
            outputs = outputs.view(-1, outputs.size(-1))  # [batch_size * seq_length, num_classes]
            labels = labels.view(-1)  # [batch_size * seq_length]

            # Count exact values and add to the total exact values
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            # Compute loss
            criterion = CrossEntropy
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Add batch loss to the total epoch loss
            epoch_loss += loss.item()

        train_accuracy = correct_predictions / total_predictions * 100

        # Check how the model do on the test set after each epoch
        test_loss, test_accuracy = validate(model, dataloader_test, device)

        train_loss_history.append(epoch_loss/len(dataloader_train))
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        # Display all the useful results of the epoch if verbose option set to true
        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}],\tTraining Loss: {epoch_loss/len(dataloader_train):.4f}, Training accuracy: {train_accuracy:.2f},\tTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}")

    return train_loss_history, train_accuracy_history, test_loss_history, test_accuracy_history

def validate(model, dataloader_test, device):
    """ Validation loop
        Args:
            model (nn.Module): PyTorch model to be evaluated
            dataloader_test (torch.utils.data.DataLoader): DataLoader object that provides an iterable over the test dataset
            device (str): the device (CPU or GPU) on which the model and data are loaded

        Returns:
            test_loss (float): epoch loss on test set
            test_accuracy (float): epoch accuracy on test set
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
            sequences, dynamic_data, labels = batch
            # Move data to device
            sequences = sequences.to(device)
            dynamic_data = dynamic_data.to(device)
            labels = labels.to(device)

            # Forward pass (different for static or dynamic training, distinguish with parameters taken as input)
            outputs = model(sequences, dynamic_data)

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
