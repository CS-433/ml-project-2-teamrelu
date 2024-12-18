import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(train_loss, train_accuracy, test_loss, test_accuracy):
    """ Plot performance of model training
    Args:
        train_loss: list with train loss for each epoch
        train_accuracy: list with train accuracy for each epoch
        test_loss: list with test loss for each epoch
        test_accuracy: list with test accuracy for each epoch
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(train_loss), label='Train Loss', color='blue')
    plt.plot(np.array(test_loss), label='Test Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.plot(np.array(train_accuracy), label='Train Accuracy', color='blue')
    plt.plot(np.array(test_accuracy), label='Test Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.show()