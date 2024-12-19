[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)

# Dynamic Protein Localization
This repository contains code and data for the second ML project on dynamic protein localization by Leonardo Bocchieri, Emilia Farina, RIccardo Rota. It covers all the pipeline followed for solving our problem, from datacleaning to postprocessing and interpretation of the results.

<details open><summary><b>Table of contents</b></summary>

- [Problem Descprition](#problem-description)
- [Usage](#usage)
  - [Quick Start](#quickstart)
  - [Data Cleaning](#data-cleaning)
  - [Run Models](#run-models)
- [Models](#models)
  - [ESM Models](#esm-models)
  - [Static Models](#static-models)
  - [Dynamic Models](#dynamic-models)
- [Citations](#citations)
- [License](#license)
</details>

## Problem Description <a name="problem-description"></a>

This repository contains a selections of models for static and dynamic protein localization on datasets of budding yeast proteins. The task performed by the models is predicting the position of the protein inside the cell among 15 possible classes, taking as input the protein sequences and information about its concentration and how it interacts with other proteins. 
The static predictions are made on a single label, i.e. the most frequent position of the given protein in the cell, while dynamic predictions are performed on 5 time-steps, correspondent to 5 different phases of the eukaryotic cell cycle. 

For a much deeper insite, refer to the report of the project.

## Usage <a name="usage"></a>

### Quick start <a name="quickstart"></a>

For setting up all the needed packages, you need to run on your bash file:

pip install -r requirements.txt

Adding if necessary the full path of the requirements.txt on your computer.

### Data Cleaning <a name="data-cleaning"></a>

You can create dataset running data_cleaning.ipynb. In this notebook, we provide our datacleaning and data inspection pipeline, together with plots and remarks.

All the created datasets are then saved to the datasets folder. Feel free to skip this preprocessing and directly go to the trainings of the models.

### Run Models <a name="run-models"></a>

You can go through all our workflow in run_all.ipynb. In this Jupyter Notebook, we create and run all the models we have implemented for the static and the dynamic problem, and also do cross-validation of the main ones.

The architectures used allow for a wide range of different hyperparameters: for easily tune them we provide some JSON files with which you can play. In particular, you can decide which architecture to use setting the _model_type_ variable in the JSON files. For more informations, please refer to the tables shown below.

## Models <a name="models"></a>

### ESM Models <a name="esm-models"></a>

The main difficulty of our problem is the small number of samples in our datasets, which forces us to use some pre-trained SOTA architectures. We decided to use ESM [(Evolutionary Scale Modeling)](https://esmatlas.com).
ESM provides large models pre-trained on huge protein datasets coming from different species. The models take as input proteic sequences of variable length, and output a correspondent fixed-dimensional embedding. In all our implementation we use the model esm2_t30_150M_UR50D, that outputs embeddings of dimension 640.

### Static Models <a name="static-models"></a>

For our static localization problem, we tried 3 different architectures:

| Name | model_type           | Sequences of the Extremities | Description  |
|-----------|---------------|---------|--------------|
| MLP    | 0       | Not Used  | Simple Fully Connected Neural Network |
| XGBoost   | 1         | Not Used | Model created loading the xgboost library: just hyperparameter tuning |
| StaticModelBranch    | 2 | Used  | Combines transformers to make sequence embeddings for extremities and linear layers to  |

### Dynamic Models <a name="dynamic-models"></a>

For our dynamic localization problem, we tried 4 different architectures.

| Name | model_type           |  | Description  |
|-----------|---------------|---------|--------------|
| LSTM Dynamic Model    | Dynamic and static Data are combined together and given to bidirectional LSTM layers for feature extraction. Linear layers for classification |
| TCN Model   | 1         | Temporal Convolutional Layers for dynamic data feature extraction, then combined with static model output with linear layers |
| Simple Model    | 2 | Just linear layers, no temporal architecture involved  |
| Modulable LSTM Dynamic Model    | 3 | Same architecture as LSTM Dynamic Model, but allows to choose which data to consider for training |