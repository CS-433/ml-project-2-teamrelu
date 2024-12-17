import numpy as np
import pandas as pd
import torch


def parse_fasta(file_path):
    """ Parses a FASTA file and extracts the identifiers and sequences
    
    Args:
		file_path : path of the fasta file (str)

    Returns:
		results : A list of tuples, each containing an identifier and a sequence

    """
    results = []

    with open(file_path, 'r') as file:
        identifier = None
        sequence = ""
        
        for line in file:
            line = line.strip()
            
            if line.startswith(">"):
				# Save the previous sequence data if present
                if identifier and sequence:
                    results.append((identifier, sequence))
                
                # Reset for the new sequence 
                identifier = line.split()[0][1:]  # First word after ">"
                sequence = ""
            
            else:
				# Accumulate the sequence of letters
                sequence += line
        
        # Save the last sequence
        if identifier and sequence:
            results.append((identifier, sequence))
    
    return results



def load_data(file_path, sheet_name, headers):
    """ Load the data from an Excel file containing many sheets
        
    Args:
        file_path : path of the Excel file (str)
        sheet_name : name of the sheet to load (str)
        headers : row numbers to use as the column names (int or list of int)

    Returns:
        data : A pandas DataFrame containing the loaded data
    """
    data = pd.read_excel(file_path, sheet_name=sheet_name, header = headers)
    
    return data



def detect_long_sequences_mh(data):
    """ Detects the protein sequences containing more than 1024 amminoacids 
    
    Args:
		data : a pandas DataFrame with a column calles ('Sequence', 'Seq'), i.e multiheader

    Returns:
		long_sequences : A list containing the sequences that are too long to perform the embedding

    """
    long_sequences = []
    for ii in range(len(data)):
        if len(data[('Sequence', 'Seq')].iloc[ii])>1024:
            long_sequences.append(data[('yORF', 'ORF')].iloc[ii])
            
    return long_sequences



def detect_long_sequences(data):
    """ Detects the protein sequences containing more than 1024 amminoacids 
    
    Args:
		data : a pandas DataFrame with a column calles 'Sequence'

    Returns:
		long_sequences : A list containing the sequences that are too long to perform the embedding

    """
    long_sequences = []
    for ii in range(len(data)):
        if len(data['Sequence'].iloc[ii])>1024:
            long_sequences.append(data['yORF'].iloc[ii])
            
    return long_sequences



def datasets_creation(file_path, sheet_names, proteins_path):
    """ Creation of the dataset common between the static and dynamic data 
    
    Args:
        file_path : path of the Excel file (str)
        sheet_names : list of sheet names to load (list of str)
        proteins_path : path of the fasta file (str)

    Returns:
        final_df_with_long : A pandas DataFrame containing also the long sequences needed for the data analyses
        dynamic_dataset : A pandas DataFrame containing the merged and processed dynamic data
        static_dataset : A pandas DataFrame containing the merged and processed static data
    """
    headers = [0,1]
    
    # Load the three replicas of the data
    first_data = load_data(file_path, sheet_names[0], headers)
    second_data = load_data(file_path, sheet_names[1], headers)
    third_data = load_data(file_path, sheet_names[2], headers)
    
    # Rename the first two columns without a name 
    first_data = first_data.rename(columns={'Unnamed: 0_level_0': 'yORF'})
    first_data = first_data.rename(columns={'Unnamed: 1_level_0': 'Name'})
    second_data = second_data.rename(columns={'Unnamed: 0_level_0': 'yORF'})
    second_data = second_data.rename(columns={'Unnamed: 1_level_0': 'Name'})
    third_data = third_data.rename(columns={'Unnamed: 0_level_0': 'yORF'})
    third_data = third_data.rename(columns={'Unnamed: 1_level_0': 'Name'})
    
    # Delete the useless columns
    first_data = first_data.drop(columns=["Name"])
    second_data = second_data.drop(columns=["Name"])
    third_data = third_data.drop(columns=["Name"])
    
    # Merge the three DataFrames based on the 'yORF' column
    merged_df = pd.concat([first_data, second_data, third_data], axis=0)
    final_df = merged_df.groupby([('yORF', 'ORF')]).mean().reset_index()
    
    final_df = rename_classes(final_df)
   
    # add the protein sequences to de dataset 
    proteins = pd.DataFrame(parse_fasta(proteins_path), columns=["Identifier", "Sequence"])
    final_df_with_sequences = add_sequences(final_df, proteins)
    # We need to remove * at the end of each sequence (since our embedding model do not have * in dictionary)
    final_df_with_sequences[('Sequence', 'Seq')] = final_df_with_sequences[('Sequence', 'Seq')].str[:-1]
    
    # Group some classes considering the near localizations
    final_df_with_sequences = group_classes(final_df_with_sequences)
    
    # Move the sequence column at the end of the table
    column_to_move = ('Sequence', 'Seq')
    final_df_with_sequences = final_df_with_sequences[[col for col in final_df_with_sequences.columns if col != column_to_move] + [column_to_move]]

    long_sequences = detect_long_sequences_mh(final_df_with_sequences)
    # Drop the proteins with a sequence too long for the embedding 
    final_df_with_long = final_df_with_sequences.copy()
    final_df_with_sequences = final_df_with_sequences[~final_df_with_sequences[('yORF','ORF')].isin(long_sequences)].reset_index(drop=True)
    
    # G1 Pre Start
    columns_to_select = [('yORF', 'ORF')]
    columns_with_g1_pre_start = [col for col in final_df_with_sequences.columns if col[1] == 'G1 Pre-START']
    columns_to_select.extend(columns_with_g1_pre_start)
    proteins_Pre_START = final_df_with_sequences[columns_to_select]
    proteins_Pre_START.columns = proteins_Pre_START.columns.droplevel(1)
    G1_Pre_Start = proteins_Pre_START.copy()
    G1_Pre_Start['G1 Pre-START'] = G1_Pre_Start.iloc[:, 1:].values.argmax(axis=1)
    G1_Pre_Start = G1_Pre_Start.drop(G1_Pre_Start.columns[1:-1], axis = 1)
    
    # G1 Post Start
    columns_to_select = [('yORF', 'ORF')]
    columns_with_g1_post_start = [col for col in final_df_with_sequences.columns if col[1] == 'G1 Post-START']
    columns_to_select.extend(columns_with_g1_post_start)
    proteins_Post_START = final_df_with_sequences[columns_to_select]
    proteins_Post_START.columns = proteins_Post_START.columns.droplevel(1)
    G1_Post_Start = proteins_Post_START.copy()
    G1_Post_Start['G1 Post-START'] = G1_Post_Start.iloc[:, 1:].values.argmax(axis=1)
    G1_Post_Start = G1_Post_Start.drop(G1_Post_Start.columns[1:-1], axis = 1)
    
    # S/G2
    columns_to_select = [('yORF', 'ORF')]
    columns_with_s_g2 = [col for col in final_df_with_sequences.columns if col[1] == 'S/G2']
    columns_to_select.extend(columns_with_s_g2)
    proteins_S_G2 = final_df_with_sequences[columns_to_select]
    proteins_S_G2.columns = proteins_S_G2.columns.droplevel(1)
    s_G2 = proteins_S_G2.copy()
    s_G2['S/G2'] = s_G2.iloc[:, 1:].values.argmax(axis=1)
    s_G2 = s_G2.drop(s_G2.columns[1:-1], axis = 1)
    
    # Metaphase
    columns_to_select = [('yORF', 'ORF')]
    columns_with_metaphase = [col for col in final_df_with_sequences.columns if col[1] == 'Metaphase']
    columns_to_select.extend(columns_with_metaphase)
    proteins_metaphase = final_df_with_sequences[columns_to_select]
    proteins_metaphase.columns = proteins_metaphase.columns.droplevel(1)
    Metaphase = proteins_metaphase.copy()
    Metaphase['Metaphase'] = Metaphase.iloc[:, 1:].values.argmax(axis=1)
    Metaphase = Metaphase.drop(Metaphase.columns[1:-1], axis = 1)
    
    # Anaphase
    columns_to_select = [('yORF', 'ORF')]
    columns_with_anaphase = [col for col in final_df_with_sequences.columns if col[1] == 'Anaphase']
    columns_to_select.extend(columns_with_anaphase)
    proteins_anaphase = final_df_with_sequences[columns_to_select]
    proteins_anaphase.columns = proteins_anaphase.columns.droplevel(1)
    Anaphase = proteins_anaphase.copy()
    Anaphase['Anaphase'] = Anaphase.iloc[:, 1:].values.argmax(axis=1)
    Anaphase = Anaphase.drop(Anaphase.columns[1:-1], axis = 1)
    
    # Telophase
    columns_to_select = [('yORF', 'ORF')]
    columns_with_telophase = [col for col in final_df_with_sequences.columns if col[1] == 'Telophase']
    columns_to_select.extend(columns_with_telophase)
    proteins_telophase = final_df_with_sequences[columns_to_select]
    proteins_telophase.columns = proteins_telophase.columns.droplevel(1)
    Telophase = proteins_telophase.copy()
    Telophase['Telophase'] = Telophase.iloc[:, 1:].values.argmax(axis=1)
    Telophase = Telophase.drop(Telophase.columns[1:-1], axis = 1)
    
    # CREATION OF THE DYNAMIC DATASET
    dynamic_dataset = pd.concat([G1_Pre_Start, G1_Post_Start, s_G2, Metaphase, Anaphase, Telophase], axis=1)
    # Remove the duplicates
    dynamic_dataset = dynamic_dataset.loc[:, ~dynamic_dataset.columns.duplicated()]
    
    # export in a csv file
    create_csv(dynamic_dataset, 'dynamic_localizations.csv', index = False)
    
    # CREATION OF THE STATIC DATASET
    data = pd.concat([proteins_Pre_START, proteins_Post_START, proteins_metaphase, proteins_telophase], axis=0)
    result_df = data.groupby(['yORF']).mean().reset_index()
    
    result_df['localization'] = result_df.iloc[:, 1:].values.argmax(axis=1)
    result_df = result_df.drop(result_df.columns[1:-1], axis=1)
    
    # Export in a csv file 'yORF_localization'
    create_csv(result_df, 'yORF_localizations.csv', index = False)
    
    # add sequences 
    static_dataset = result_df.merge(proteins, left_on=['yORF'], right_on=['Identifier'], how='left')
    
    # Drop the redundant columns
    static_dataset.drop(columns = {'Identifier', 'localization'})
    
    # Export in a csv file 'yORF_sequences'
    create_csv(static_dataset, 'yORF_sequences.csv', index = False)

    return final_df_with_long, dynamic_dataset, static_dataset



def add_sequences(data, proteins):
    """
    Add the protein sequences to the dataset
    
    Args:
        data : A pandas DataFrame without the protein sequences
        proteins : A pandas DataFrame containing the protein sequences

    Returns:
        data_with_sequences : The result of merging the two DataFrames
    """
    second_level = ["ID", "Seq"] 
    proteins_ = proteins.copy()
    multi_index = pd.MultiIndex.from_tuples(zip(proteins_.columns, second_level))
    proteins_.columns = multi_index
    
    # Merge dynamic data with proteins to add the protein sequences
    data_with_sequences = data.merge(proteins_, left_on=[('yORF', 'ORF')], right_on=[('Identifier', 'ID')], how='left')

    # Drop the redundant 'Identifier' column
    data_with_sequences.drop(columns=[('Identifier','ID')], inplace=True)

    # Rename the 'Sequence' column to 'Protein Sequence' and drop the nan
    data_with_sequences.rename(columns={('Sequence','Seq'): ('Protein Sequence','Seq')}, inplace=True)
    data_with_sequences = data_with_sequences.dropna(subset=[('Sequence', 'Seq')]).reset_index(drop=True)
    
    return data_with_sequences



def rename_classes(data):
    """
    Renames the classes in the dataset to more readable names
    
    Args:
        data : A pandas DataFrame containing the dataset

    Returns:
        data : A pandas DataFrame with renamed classes
    """
    data = data.rename(columns={'Bud': 'bud'})
    data = data.rename(columns={'Bud Neck': 'bud neck'})
    data = data.rename(columns={'Cell Periphery': 'cell periphery'})
    data = data.rename(columns={'Endoplasmic Reticulum': 'ER'})
    data = data.rename(columns={'Endosome': 'endosome'})
    data = data.rename(columns={'Lipid Particles': 'lipid particle'})
    data = data.rename(columns={'Mitochondria': 'mitochondrion'})
    data = data.rename(columns={'Nucleolus': 'nucleolus'})
    data = data.rename(columns={'Nucleus': 'nucleus'})
    data = data.rename(columns={'Peroxisomes': 'peroxisome'})
    data = data.rename(columns={'Vacuole': 'vacuole'})  
    
    return data



def group_classes(data):
    """
    Groups similar (for what concern the localization) classes together in the dataset
    
    Args:
        data : A pandas DataFrame containing the dataset

    Returns:
        data : A pandas DataFrame with grouped classes
    """
    data = data.rename(columns={'Actin': 'cytoskeleton'})
    data = data.rename(columns={'Golgi': 'golgi'})
    data[('Bud', 'G1 Pre-START')] = data[[('bud', 'G1 Pre-START'), ('Bud Periphery', 'G1 Pre-START'), ('Bud Site', 'G1 Pre-START'), ('bud neck', 'G1 Pre-START')]].sum(axis=1)
    data[('Bud', 'G1 Post-START')] = data[[('bud', 'G1 Post-START'), ('Bud Periphery', 'G1 Post-START'), ('Bud Site', 'G1 Post-START'), ('bud neck', 'G1 Post-START')]].sum(axis=1)
    data[('Bud', 'S/G2')] = data[[('bud', 'S/G2'), ('Bud Periphery', 'S/G2'), ('Bud Site', 'S/G2'), ('bud neck', 'S/G2')]].sum(axis=1)
    data[('Bud', 'Metaphase')] = data[[('bud', 'Metaphase'), ('Bud Periphery', 'Metaphase'), ('Bud Site', 'Metaphase'), ('bud neck', 'Metaphase')]].sum(axis=1)
    data[('Bud', 'Anaphase')] = data[[('bud', 'Anaphase'), ('Bud Periphery', 'Anaphase'), ('Bud Site', 'Anaphase'), ('bud neck', 'Anaphase')]].sum(axis=1)
    data[('Bud', 'Telophase')] = data[[('bud', 'Telophase'), ('Bud Periphery', 'Telophase'), ('Bud Site', 'Telophase'), ('bud neck', 'Telophase')]].sum(axis=1)
    data = data.drop(columns=['bud', 'Bud Periphery', 'Bud Site', 'bud neck'])
    data[('cytoplasm', 'G1 Pre-START')] = data[[('Cytoplasm', 'G1 Pre-START'), ('Cytoplasmic Foci', 'G1 Pre-START')]].sum(axis=1)
    data[('cytoplasm', 'G1 Post-START')] = data[[('Cytoplasm', 'G1 Post-START'), ('Cytoplasmic Foci', 'G1 Post-START')]].sum(axis=1)
    data[('cytoplasm', 'S/G2')] = data[[('Cytoplasm', 'S/G2'), ('Cytoplasmic Foci', 'S/G2')]].sum(axis=1)
    data[('cytoplasm', 'Metaphase')] = data[[('Cytoplasm', 'Metaphase'), ('Cytoplasmic Foci', 'Metaphase')]].sum(axis=1)
    data[('cytoplasm', 'Anaphase')] = data[[('Cytoplasm', 'Anaphase'), ('Cytoplasmic Foci', 'Anaphase')]].sum(axis=1)
    data[('cytoplasm', 'Telophase')] = data[[('Cytoplasm', 'Telophase'), ('Cytoplasmic Foci', 'Telophase')]].sum(axis=1)
    data = data.drop(columns = ['Cytoplasm', 'Cytoplasmic Foci'])
    data[('Nucleus', 'G1 Pre-START')] = data[[('nucleus', 'G1 Pre-START'), ('Nuclear Periphery', 'G1 Pre-START'), ('nucleolus', 'G1 Pre-START')]].sum(axis=1)
    data[('Nucleus', 'G1 Post-START')] = data[[('nucleus', 'G1 Post-START'), ('Nuclear Periphery', 'G1 Post-START'), ('nucleolus', 'G1 Post-START')]].sum(axis=1)
    data[('Nucleus', 'S/G2')] = data[[('nucleus', 'S/G2'), ('Nuclear Periphery', 'S/G2'), ('nucleolus', 'S/G2')]].sum(axis=1)
    data[('Nucleus', 'Metaphase')] = data[[('nucleus', 'Metaphase'), ('Nuclear Periphery', 'Metaphase'), ('nucleolus', 'Metaphase')]].sum(axis=1)
    data[('Nucleus', 'Anaphase')] = data[[('nucleus', 'Anaphase'), ('Nuclear Periphery', 'Anaphase'), ('nucleolus', 'Anaphase')]].sum(axis=1)
    data[('Nucleus', 'Telophase')] = data[[('nucleus', 'Telophase'), ('Nuclear Periphery', 'Telophase'), ('nucleolus', 'Telophase')]].sum(axis=1)
    data = data.drop(columns = ['nucleus', 'Nuclear Periphery', 'nucleolus'])
    data[('Vacuole', 'G1 Pre-START')] = data[[('vacuole', 'G1 Pre-START'), ('Vacuole Periphery', 'G1 Pre-START')]].sum(axis=1)
    data[('Vacuole', 'G1 Post-START')] = data[[('vacuole', 'G1 Post-START'), ('Vacuole Periphery', 'G1 Post-START')]].sum(axis=1)
    data[('Vacuole', 'S/G2')] = data[[('vacuole', 'S/G2'), ('Vacuole Periphery', 'S/G2')]].sum(axis=1)
    data[('Vacuole', 'Metaphase')] = data[[('vacuole', 'Metaphase'), ('Vacuole Periphery', 'Metaphase')]].sum(axis=1)
    data[('Vacuole', 'Anaphase')] = data[[('vacuole', 'Anaphase'), ('Vacuole Periphery', 'Anaphase')]].sum(axis=1)
    data[('Vacuole', 'Telophase')] = data[[('vacuole', 'Telophase'), ('Vacuole Periphery', 'Telophase')]].sum(axis=1)
    data = data.drop(columns = ['vacuole', 'Vacuole Periphery'])
    
    return data
    
  
    
def create_csv(data, csv_file, index=False):
    """
    Exports the data in a csv file 
    
    Args:
        data : A pandas DataFrame containing the dataset
        csv_file : the path of the new file (str)
    """
    data.to_csv(csv_file, index=index)



def interaction_matrix(file_path, data, output_file_path, delimiter=';'):
    """
    Create a matrix containing the interactions between the different proteins and exports it in a csv file
    
    Args:
        file_path : path to the file containing the interactions between the proteins 
        delimiter : a char that separates the data in the file
        data : A pandas DataFrame containing the static dataset
        
    Returns:
        final_interaction_mat : pandas DataFrame representing a symmetric matrix of the interactions between the proteins  
    """
    # Extract the data from the file 
    interaction_matrices = pd.read_csv(file_path, delimiter=delimiter)
    columns_to_select = ['source', 'target', 'score_FDR+cor']
    interaction_matrices = interaction_matrices[columns_to_select]
    interaction_matrices['target'] = (
        interaction_matrices['target']
        .astype(str) 
        .str.replace(r"[\[\]']", "", regex=True)  
        .str.split(";") 
    )

    expanded_matrices = interaction_matrices.explode('target')
    expanded_matrices = expanded_matrices.dropna(subset=['source', 'target', 'score_FDR+cor'])

    expanded_matrices = expanded_matrices.reset_index(drop=True)
    dynamic_yorf_set = set(data['yORF']) 

    final_interactions = expanded_matrices[
        (expanded_matrices['source'].isin(dynamic_yorf_set)) & 
        (expanded_matrices['target'].isin(dynamic_yorf_set))    
    ]

    final_interactions = final_interactions.reset_index(drop=True)
    
    proteins = sorted(set(final_interactions['source']).union(set(final_interactions['target'])))

    interaction_matrix = pd.DataFrame(0, index=proteins, columns=proteins)

    # Populate the matrix
    for _, row in final_interactions.iterrows():
        source = row['source']
        target = row['target']
        score = row['score_FDR+cor']
        interaction_matrix.loc[source, target] = score 
    
    # Using the symmetry
    interaction_matrix = interaction_matrix.combine_first(interaction_matrix.T)
    proteins_list = data['yORF'].unique()

    existing_proteins = interaction_matrix.index
    # Adding the missing proteins with a value for the interactions equal to 0.0 
    missing_proteins = list(set(proteins_list) - set(existing_proteins))
    new_data = pd.DataFrame(0.0, index=missing_proteins, columns=interaction_matrix.columns)

    final_interaction_mat = interaction_matrix.copy()
    final_interaction_mat = pd.concat([final_interaction_mat, new_data])

    new_columns = pd.DataFrame(0.0, index=final_interaction_mat.index, columns=missing_proteins)
    final_interaction_mat = pd.concat([final_interaction_mat, new_columns], axis=1)

    final_interaction_mat = final_interaction_mat.sort_index(axis=0).sort_index(axis=1)

    create_csv(final_interaction_mat, output_file_path)

    return final_interaction_mat       



def create_dynamic_tensor(interaction_data, te_levels, tl_levels, use_concentration = True):
    """
    Create a matrix containing the interactions between the different proteins
    
    Args:
        file_path : path to the file containing the interactions between the proteins 
        delimiter : a char that separates the data in the file
        data : A pandas DataFrame containing the static dataset
    """
    #Dataframe to tensor
    dynamic_tensor = torch.tensor(interaction_data.values, dtype=torch.float32)
    # Add temporal length
    dynamic_tensor = dynamic_tensor.unsqueeze(1)
    # Split data of each timestep
    split_tensors = torch.split(dynamic_tensor, 15, dim=2)
    # Concatenate along the second dimension (dim=1)
    result_tensor = torch.cat(split_tensors, dim=1)

    if use_concentration == True:
        te_levels_tensor = torch.tensor(te_levels.values, dtype=torch.float32)
        te_levels_tensor = te_levels_tensor.unsqueeze(-1)
        tl_levels_tensor = torch.tensor(tl_levels.values, dtype=torch.float32)
        tl_levels_tensor = tl_levels_tensor.unsqueeze(-1)
    result_tensor = torch.cat((result_tensor, te_levels_tensor, tl_levels_tensor), dim=2)

    return result_tensor
    

def protein_levels_cleaning(file_path, sheet_names):
    """
    Create a matrix containing the interactions between the different proteins
    
    Args:
        file_path : path to the file containing the interactions between the proteins 
        delimiter : a char that separates the data in the file
        data : A pandas DataFrame containing the static dataset
    """
     
    headers = [0,1]
    tl_data = load_data(file_path = file_path, sheet_name = sheet_names[0], )