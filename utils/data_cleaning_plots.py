import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
from utils.data_cleaning_helpers import *

# Pie Chart (general and for each phases) 
def plot_stationary_proteins(data):
    """ Plots the frequencies of the total stationary and dynamic proteins in the dataset 
    
    Args:
		data : pandas DataFrame containing the dynamic dataset
  
    """
    # It counts the proteins that do not change location (stationary)
    stationary_count = sum(data.iloc[:, 1:].nunique(axis=1) == 1)
    # It counts the dynamic proteins
    moving_count = len(data) - stationary_count
    
    labels = ["Stationary Proteins", "Dynamic Proteins"]
    sizes = [stationary_count, moving_count]
    colors = ["skyblue", "salmon"]

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    plt.title("Frequency of stationary and dynamic proteins")
    plt.axis("equal") 
    


def plot_stationary_proteins_per_phase(data):
    """ Plots the frequencies of the stationary and dynamic proteins in the dataset for each phase as subplots.
    
    Args:
        data : pandas DataFrame containing the dynamic dataset
  
    """
    phases = ["G1 Pre-START", "G1 Post-START", "S/G2", "Metaphase", "Anaphase", "Telophase"]
    
    n_plots = len(phases) - 1  
    n_cols = 3  
    n_rows = -(-n_plots // n_cols)  
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten()  
    
    for i in range(n_plots):
        start_phase = phases[i]
        end_phase = phases[i + 1]
        
        stationary_count = 0
        moving_count = 0
        
        for _, row in data.iterrows():
            start = row[start_phase]
            end = row[end_phase]
            if start == end:
                stationary_count += 1
            else:
                moving_count += 1
        
        labels = ["Stationary Proteins", "Dynamic Proteins"]
        sizes = [stationary_count, moving_count]
        colors = ["skyblue", "salmon"]
        
        # Draw in the current subplot
        ax = axes[i]
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
        ax.set_title(f"{start_phase} → {end_phase}")
        ax.axis("equal")  
    
    # Remove empty axes 
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout() 

        

# Transition matrices (general and for each phase)
def plot_transition_matrix(data):
    """ Plots the total transition matrices
    
    Args:
		data : pandas DataFrame containing the dynamic dataset
  
    """
    # phases to take into account
    phases = ["G1 Pre-START", "G1 Post-START", "S/G2", "Metaphase", "Anaphase", "Telophase"]
    
    unique_locations = sorted(set(data[phases].values.flatten()))
    transition_matrix = pd.DataFrame(0, index=unique_locations, columns=unique_locations)
    
    for _, row in data.iterrows():
        for i in range(len(phases) - 1):
            start = row[phases[i]]
            end = row[phases[i + 1]]
            transition_matrix.loc[start, end] += 1
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(transition_matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=True)
    plt.title("Transition Matrix")
    plt.xlabel("Destination")
    plt.ylabel("Origin")
    
    
    
def plot_individual_transition_matrices(data):
    """ Plots the transition matrices for each phase as subplots in multiple rows.
    
    Args:
        data : pandas DataFrame containing the dynamic dataset
  
    """
    phases = ["G1 Pre-START", "G1 Post-START", "S/G2", "Metaphase", "Anaphase", "Telophase", 'G1 Pre-START']
    
    unique_locations = sorted(set(data[phases].values.flatten()))
    n_plots = len(phases) - 1  # number of transition to plot

    # Determine the number od rows and of columns for the grid
    n_cols = 3  
    n_rows = math.ceil(n_plots / n_cols)  
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten()  
    
    transition_matrices = []
    
    for i in range(n_plots):
        # Create transition matrix 
        transition_matrix = pd.DataFrame(0, index=unique_locations, columns=unique_locations)
        
        for _, row in data.iterrows():
            start = row[phases[i]]
            end = row[phases[i + 1]]
            transition_matrix.loc[start, end] += 1
        
        transition_matrices.append(transition_matrix)
        
        # Draw matrix in the current subplot
        ax = axes[i]
        sns.heatmap(transition_matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=True, ax=ax)
        ax.set_title(f"Transition Matrix: {phases[i]} -> {phases[i + 1]}")
        ax.set_xlabel("Destination")
        ax.set_ylabel("Origin")
    
    # Remove empty axes
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()


    
# Distribution of the long sequences proteins compared to the total distribution among the classes
def long_sequences_distribution(data):
    """ Plots the comparison between the distribution among the classes of the long sequences and of all the sequences
    
    Args:
		data : pandas DataFrame containing all the sequences (also the long ones)
  
    """
    data_to_process = data.copy()
    
    # Store the columns yORF and Sequence in two vectors
    vec_yorfs = data_to_process[('yORF', 'ORF')]
    vec_sequences = data_to_process[('Sequence', 'Seq')]
    
    # Process the dataset to analyze the distribution
    final_data = data_to_process.drop(columns = {('Sequence', 'Seq'), ('yORF', 'ORF')})
    final_mean = final_data.groupby(axis=1, level=0).mean()
    final_mean['yORF'] = vec_yorfs
    final_mean['Sequence'] = vec_sequences
    
    # Consider only the numeric columns
    df_numeric = final_mean.select_dtypes(include=[np.number])
    
    # Find the maximum values for each row among the columns  
    max_mask = df_numeric.eq(df_numeric.max(axis=1), axis=0)
    df_binary = max_mask.astype(int)
    result = pd.concat([final_mean[['yORF']], df_binary, final_mean[['Sequence']]], axis=1)
    
    # Find the long sequences 
    long_sequences =  detect_long_sequences(result)
    
    indexes = result[result['yORF'].isin(long_sequences)].index.tolist()
    long_df = result.loc[indexes]
    localizations = {}
    for index, row in long_df.iterrows():
        localizations[row['yORF']] = row[row == 1].index.tolist()
    
    # Distibution of the long sequences
    cell_part_counts = long_df.iloc[:, 1:-1].sum()
    localizations = []
    counts = []
    total = cell_part_counts[:].sum()
    for localization, count in list(cell_part_counts.items()):
        localizations.append(localization)
        counts.append(count/total)
        
    # Distibution of all the sequences
    complete_cell_part_counts = result.iloc[:, 1:-1].sum()
    complete_counts = []
    complete_total = complete_cell_part_counts[:].sum()
    for complete_localization, complete_count in list(complete_cell_part_counts.items()):
        complete_counts.append(complete_count/complete_total) 
        
    # Plot the comparison
    y_positions = np.arange(len(localizations))
    plt.figure(figsize=(12, 8))
    bar_width = 0.4
    plt.barh(y_positions - bar_width/2, counts, height=bar_width, label='Long sequences Dataset', edgecolor="black")
    plt.barh(y_positions + bar_width/2, complete_counts, height=bar_width, label='Total Dataset', edgecolor="black")
    plt.yticks(y_positions, localizations)
    plt.xlabel("Relative Frequency", fontsize=12)
    plt.ylabel("Cellular Components", fontsize=12)
    plt.title("Comparison of Protein Distribution in Cellular Components", fontsize=14)
    plt.legend()
    plt.tight_layout()


