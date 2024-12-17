import itertools
import numpy as np


parameter_grid = {
    'learning_rate': [1e-3, 1e-2],
    'weight_decay': [0.0, 1e-4],
    'scheduler': ['StepLR', 'ReduceLROnPlateau'],
    'dropout': [0.3, 0.5],
    'batch_size': [32, 64]
}

all_params = list(itertools.product(*parameter_grid.values()))
param_names = list(parameter_grid.keys())

#print(all_params)
#print(param_names)
#for param_set in all_params:
    #params = dict(zip(param_names, param_set))
    #print(params)

people = [
    {"name": "Alice", "age": 30, "sex": "Female"},
    {"name": "Bob", "age": 25, "sex": "Male"},
    {"name": "Charlie", "age": 35, "sex": "Male"},
    {"name": "Diana", "age": 20, "sex": "Female"},
    {"name": "Eve", "age": 20, "sex": "Female"}
]

print(min(people, key=lambda x: x['age']))