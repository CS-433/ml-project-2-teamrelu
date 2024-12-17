import pandas as pd
import numpy as np

# DA PROVARE

def create_correlation_scores(path_interaction, path_locations, path_concentrations, output_path, top_n=10, n_classes=15, null_row='zeros'):
    """
    Function to generate correlation scores for protein interactions across different cell cycle phases and save them in a CSV

    Args:
    - path_interaction (str): Path to the CSV file containing interaction data
    - path_locations (str): Path to the CSV file containing protein locations and phase-specific localization data
    - path_concentrations (str or bool): Path to the CSV file with protein concentration data or `False` if concentrations are not provided
    - output_path (str): Path to save the resulting correlation scores as a CSV file
    - top_n (int, optional): Number of top interacting proteins to consider for correlations. Default is 10
    - n_classes (int, optional): Number of distinct protein classes. Default is 15
    - null_row (str, optional): Strategy for default vectors:
        -'zeros' (default) = set at 0 the values of proteins with no correlations
        -'freq' = cast the values at the relative frequences of the classes in the dataset                 

    Returns:
    - pd.DataFrame: A DataFrame containing the correlation scores for all proteins across phases, saved to `output_path`.
    """
    
    #importing files and filling NaN's in concentrations with the mean values
    locations = pd.read_csv(path_locations)
    yorfs = locations['yORF']  
    interactions = pd.read_csv(path_interaction)
    interactions.rename(columns={'Unnamed: 0': 'yORF'}, inplace=True) #???????

    if path_concentrations is not False:
        concentrations = pd.read_csv(path_concentrations)
    else:
        concentrations = pd.DataFrame({'ones': [1] * len(yorfs)}) #if we don't want to take in account concentrations we just set to 1
    concentrations = pd.merge(yorfs, concentrations, on='yORF', how='left')
    concentrations.fillna(concentrations.mean(numeric_only=True), inplace=True) 

    # analyze correlation for each phase
    df_1 = merge_data_per_phase('G1 Post-START', concentrations, locations, interactions)
    df_2 = merge_data_per_phase('S/G2', concentrations, locations, interactions)
    df_3 = merge_data_per_phase('Metaphase', concentrations, locations, interactions)
    df_4 = merge_data_per_phase('Anaphase', concentrations, locations, interactions)
    df_5 = merge_data_per_phase('Telophase', concentrations, locations, interactions)
    cor_pos_1 = correlation_data_per_phase(df_1, null_vector_option=null_row, num_classes=n_classes, top_n=top_n)
    cor_pos_2 = correlation_data_per_phase(df_2, null_vector_option=null_row, num_classes=n_classes, top_n=top_n)
    cor_pos_3 = correlation_data_per_phase(df_3, null_vector_option=null_row, num_classes=n_classes, top_n=top_n)
    cor_pos_4 = correlation_data_per_phase(df_4, null_vector_option=null_row, num_classes=n_classes, top_n=top_n)
    cor_pos_5 = correlation_data_per_phase(df_5, null_vector_option=null_row, num_classes=n_classes, top_n=top_n)

    # concatenate all phases and save the dataset
    df_all = pd.concat([cor_pos_1.set_index('yORF'),
                    cor_pos_2.set_index('yORF'),
                    cor_pos_3.set_index('yORF'),
                    cor_pos_4.set_index('yORF'),
                    cor_pos_5.set_index('yORF')], axis=1, keys=['Phase_1', 'Phase_2', 'Phase_3', 'Phase_4', 'Phase_5'])
    df_all.columns = ['{}_{}'.format(phase, col) for phase, col in df_all.columns]
    df_all = df_all.reset_index()
    df_all.to_csv(output_path, header=True, index=False)

    return df_all

def get_correlated_positions(correlation_matrix, locations, concentrations, null_row, n=10, n_classes=15):
    """  
    Function to compute the correlation weights for proteins based on a correlation matrix. It identifies the top `n` most correlated 
    proteins for each row and assigns weights based on their interaction scores, including ties at the threshold value

    Parameters:
    - correlation_matrix (np.ndarray): An NxN square matrix representing correlation values between proteins.
    - locations (np.ndarray): An array of length N indicating the class of each protein
    - concentrations (np.ndarray or False): An array of length N with protein concentrations, or False if not provided
    - null_row (np.ndarray): A default row vector (length = n_classes) to assign to rows with no interactions
    - n (int): The number of top interacting proteins to consider.
    - n_classes (int): The number of distinct protein classes

    Returns:
    - output (np.ndarray): An Nxn_classes array where each row represents the probability distribution 
      on the classes classes based on correlation and interaction scores for a given protein
    """

    if concentrations is not False:
        if correlation_matrix.shape[1] != concentrations.shape[0]:
            raise ValueError("Number of matrix columns different from concentration vector length")
        correlation_matrix = (correlation_matrix.T * concentrations).T #multiplies column-wise for the concentration value
    
    if correlation_matrix.shape[1] != locations.shape[0]:
        raise ValueError("Number of matrix rows different from locations vector length")

    sorted_indices = np.argsort(correlation_matrix, axis=1)
    thresholds = correlation_matrix[np.arange(correlation_matrix.shape[0]), sorted_indices[:, -n]]; #threshold is the interaction score of the 10th most interacting protein
    bests = [np.where(row > threshold)[0] for row, threshold in zip(correlation_matrix, thresholds)] #proteins with value higher than threshold
    n_bests = np.array([len(a) for a in bests])
    
    # Tied proteins at threshold value
    evens = [np.where(row == threshold)[0] if threshold != 0 else np.array([]) for row, threshold in zip(correlation_matrix, thresholds)] #proteins with value equal to threshold
    n_evens = np.array([len(a) for a in evens])

    # Assign weights to each protein
    n_bests_without_0 = np.where(n_bests == 0, 0.001, n_bests) #line to avoid division by 0 warning in the next
    weight_bests = np.where(n_bests == 0, 0, np.where(n_evens > 0, 1/n, 1/n_bests_without_0)) #if at least 10 proteins interacts it will be 0.1
    n_evens_without_0 = np.where(n_evens == 0, 0.001, n_evens) #line to avoid division by 0 warning in the next
    weight_evens = np.where(n_evens > 0, (n-n_bests)/(n*n_evens_without_0), 0) #weight of the proteins with value equal to threshold

    # Classes of the interacting proteins
    best_classes = [locations[indices] if len(indices) > 0 else [] for indices in bests] #classes of proteins in bests
    even_classes = [locations[indices] if len(indices) > 0 else [] for indices in evens] #classes of proteins in evens

    # Creating output
    output = np.zeros([correlation_matrix.shape[0], n_classes])
    for i in range(correlation_matrix.shape[0]): #add weight*class for each protein in bests and evens
        np.add.at(output[i], best_classes[i], weight_bests[i])
        np.add.at(output[i], even_classes[i], weight_evens[i])

    output[np.all(output == 0, axis=1)] = null_row #if some proteins don't interact we assign a default value to them, usually zeros
    return output

def merge_data_per_phase(phase, concentrations, locations, interactions):
    """
    Helper function: Combines protein data for a specific cell cycle phase.

    Args:
    - phase (str): The phase of the cell cycle (e.g., 'G1 Post-START', 'S/G2', etc.)
    - concentrations (pd.DataFrame): DataFrame containing protein concentrations, including a column for the specific phase
    - locations (pd.DataFrame): DataFrame containing protein localization data, including a column for the specific phase
    - interactions (pd.DataFrame): DataFrame containing protein interaction data, with 'yORF' as the key column

    Returns:
    - pd.DataFrame: A DataFrame containing the merged data for the given phase, including `yORF`, `concentration`, `localization`, and interaction scores
    """
    #0: Yorf, 1: concentration, 2: location 3->end:interactions
    concentrations_phase = concentrations[['yORF', phase]].rename(columns={phase: 'concentration'})
    locations_phase= locations[['yORF', phase]].rename(columns={phase: 'localization'})
    df_phase = pd.merge(concentrations_phase, locations_phase, on='yORF', how='right')
    df_phase = pd.merge(df_phase, interactions, on='yORF')
    return df_phase

def correlation_data_per_phase(df, null_vector_option, num_classes, top_n):
    """
    Helper function: calculates class correlation data for a specific cell cycle phase

    Args:
    - df (pd.DataFrame): A DataFrame containing protein data, including columns for `yORF`, `concentration`, `localization`, and interactions
    - null_vector_option (str): Strategy for default vectors ('zeros' or 'freq') for proteins with no interactions
    - num_classes (int): Number of distinct protein classes
    - top_n (int): Number of top interacting proteins to consider

    Returns:
    - pd.DataFrame: A DataFrame with correlation probabilities for each protein, including a `yORF` column and class probabilities
    """
    correlation_matrix = df.iloc[:, 3:].to_numpy()
    locations = df.iloc[:, 2].to_numpy()
    frequences = np.bincount(locations, minlength=num_classes)
    if null_vector_option == 'zeros':
        null_vector = np.zeros(num_classes)
    elif null_vector_option == 'freq':
        relative_freq = frequences / frequences.sum()
        null_vector = relative_freq
    concentrations = df.iloc[:, 1].to_numpy()
    cor_pos = get_correlated_positions(correlation_matrix=correlation_matrix, locations=locations, concentrations=concentrations, n=top_n, n_classes=15, null_row=null_vector)
    cor_pos =pd.DataFrame(cor_pos)
    cor_pos['yORF'] = df['yORF']
    return cor_pos