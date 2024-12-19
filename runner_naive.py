from naive_model.naive_trainer import run_training
from naive_model.naive_model import NaiveModel
from torch.optim import AdamW
import os
import pandas as pd
from naive_model.sequences_dataloader import NaiveProteinDataset, create_dynamic_tensor
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from utils.training_graphs import plot_training_results

model = NaiveModel(input_dim=531, num_classes=15, num_timesteps=5)
optimizer = AdamW(model.parameters(), lr=0.001)
current_dir = os.path.dirname(os.path.abspath(__file__)) 
sequences_path = os.path.join(current_dir, 'datasets/yORF_sequences.csv')
dynamic_data_path = os.path.join(current_dir, 'datasets/final_dataset_dyn_with_te.csv')
dynamic_data = pd.read_csv(dynamic_data_path)
sequences = pd.read_csv(sequences_path)
data = pd.merge(sequences, dynamic_data, on='yORF')
print(data.columns)
# dataset = NaiveProteinDataset(sequences, dynamic_data, labels, max_length=512)
# Split dataframe columns based on feature groups
data_train, data_test = train_test_split(data, test_size=0.2, random_state=30026)
sequences_train = data_train.iloc[:, 1]
dynamic_data_train = create_dynamic_tensor(data_train.iloc[:, 654:729], data_train.iloc[:, 649:654], data_train.iloc[:,644:649])
labels_train = data_train.iloc[:, -5:]
sequences_test = data_test.iloc[:, 1]
dynamic_data_test = create_dynamic_tensor(data_test.iloc[:, 654:729], data_test.iloc[:, 649:654], data_test.iloc[:,644:649])
labels_test = data_test.iloc[:, -5:]

dataset_train = NaiveProteinDataset(sequences_train, dynamic_data_train, labels_train)
dataset_test = NaiveProteinDataset(sequences_test, dynamic_data_test, labels_test)

dataloader_train = DataLoader(dataset_train, batch_size=5, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=5, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loss, train_accuracy, test_loss, test_accuracy = run_training(model, optimizer, dataloader_train, dataloader_test, num_epochs=50, device=device, verbose=True)
plot_training_results(train_loss, train_accuracy, test_loss, test_accuracy)
