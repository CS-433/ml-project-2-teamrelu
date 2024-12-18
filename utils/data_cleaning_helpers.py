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



def datasets_creation(file_path, sheet_names, proteins_path, output_sequences_path, output_static_path, output_dynamic_path):
    """ Creation of the dataset common between the static and dynamic data 
    
    Args:
        file_path : path of the Excel file (str)
        sheet_names : list of sheet names to load (list of str)
        proteins_path : path of the fasta file (str)
        output_sequences_path : path (str) in which store the dataset containing the yORF and sequences for all proteins
        output_static_path : path (str) in which store the dataset containing the yORF and static localizations for all proteins
        output_static_path : path (str) in which store the dataset containing the yORF and dynamic localizations for all proteins

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
    create_csv(dynamic_dataset, output_dynamic_path, index = False)
    
    # CREATION OF THE STATIC DATASET
    data = pd.concat([proteins_Pre_START, proteins_Post_START, proteins_metaphase, proteins_telophase], axis=0)
    result_df = data.groupby(['yORF']).mean().reset_index()
    
    result_df['localization'] = result_df.iloc[:, 1:].values.argmax(axis=1)
    result_df = result_df.drop(result_df.columns[1:-1], axis=1)
    
    # Export in a csv file 'yORF_localization'
    create_csv(result_df, output_static_path, index = False)
    
    # add sequences 
    static_dataset = result_df.merge(proteins, left_on=['yORF'], right_on=['Identifier'], how='left')
    
    static_dataset['Sequence'] = static_dataset['Sequence'].str[:-1]
    
    # Drop the redundant columns
    static_dataset = static_dataset.drop(columns = {'Identifier', 'localization'})
    
    # Export in a csv file 'yORF_sequences'
    create_csv(static_dataset, output_sequences_path, index = False)

    return final_df_with_long, dynamic_dataset, static_dataset



def create_extrem_sequences(file_path, output_file_path, num = 20):
    """
    Create a dataset which contains the yORFs and the first and last 'num' amino acids of the sequences
    
    Args:
        file_path : path (str) to a file containing the yORFs and the sequences of amino acids
        output_file_path : path (str) in which save the pandas DataFrame generated
        num : number of amino acids 
        
    Returns:
        data : pandas DataFrame containing the yORFs and the first and last 'num' amino acids of the sequences
    """
    df = pd.read_csv(file_path)
    data = ((seq[:num], seq[-num:]) for seq in df.iloc[:, -1])
    data_extremities=pd.DataFrame(data, columns = ["beginning", "end"])
    data_extremities.insert(0, "yORF", df.iloc[:,0])
    create_csv(data_extremities, output_file_path)
    return data_extremities
    


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

    final_interaction_mat.to_csv(output_file_path, index = True, index_label='yORF')

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
    

def protein_levels_cleaning(file_path, sheet_names, output_file_path_te, output_file_path_tl):
    """
    Export two files csv containing the protein concentrations
    
    Args:
        file_path : path to the file containing the concentrations of proteins 
        sheet_names : a list containing the different sheets to read
        output_file_path_te : a file path (str) in which store the TE concentrations of proteins
        output_file_path_te : a file path (str) in which store the TL concentrations of proteins
    """
    headers = 0
    data_TE = load_data(file_path = file_path, sheet_name = sheet_names[1], headers = headers)
    data_TL = load_data(file_path = file_path, sheet_name = sheet_names[0], headers = headers)
    
    # Drop useless rows and columns
    data_TE = data_TE.drop(data_TE.columns[1], axis=1)
    data_TL = data_TL.drop(data_TL.columns[1], axis=1)
    data_TE = data_TE.drop(index=0).reset_index(drop=True)
    data_TL = data_TL.drop(index=0).reset_index(drop=True)
    
    # Cast values in dataset saved as strings
    data_TE.iloc[:, 1:] = data_TE.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    data_TL.iloc[:, 1:] = data_TL.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    
    # Make mean among replicates
    data_TE['G1 Post-START'] = data_TE.iloc[:, [1, 6]].mean(axis=1)
    data_TE['S/G2'] = data_TE.iloc[:, [2, 7]].mean(axis=1)
    data_TE['Metaphase'] = data_TE.iloc[:, [3, 8]].mean(axis=1)
    data_TE['Anaphase'] = data_TE.iloc[:, [4, 9]].mean(axis=1)
    data_TE['Telophase'] = data_TE.iloc[:, [5, 10]].mean(axis=1)
    data_TL['G1 Post-START'] = data_TL.iloc[:, [1, 6]].mean(axis=1)
    data_TL['S/G2'] = data_TL.iloc[:, [2, 7]].mean(axis=1)
    data_TL['Metaphase'] = data_TL.iloc[:, [3, 8]].mean(axis=1)
    data_TL['Anaphase'] = data_TL.iloc[:, [4, 9]].mean(axis=1)
    data_TL['Telophase'] = data_TL.iloc[:, [5, 10]].mean(axis=1)
    
    #Drop single replicates data
    data_TE = data_TE.drop(data_TE.columns[1:11], axis=1)
    data_TL = data_TL.drop(data_TL.columns[1:11], axis=1)
    
    # Rename the first column to 'yORF'
    data_TE.rename(columns={data_TE.columns[0]: 'yORF'}, inplace=True)
    data_TL.rename(columns={data_TL.columns[0]: 'yORF'}, inplace=True)
    
    # Normalize the data
    for col in data_TE.columns[1:]:
        data_TE[col] = (data_TE[col] - data_TE[col].mean()) / data_TE[col].std()

    for col in data_TL.columns[1:]:
        data_TL[col] = (data_TL[col] - data_TL[col].mean()) / data_TL[col].std()
        
    # Export the data in csv files
    create_csv(data_TE, output_file_path_te)
    create_csv(data_TL, output_file_path_tl)



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

    if path_concentrations is not False:
        concentrations = pd.read_csv(path_concentrations)
    else:
        concentrations = pd.DataFrame({
            'yORF': yorfs,
            'G1 Post-START': [1] * len(yorfs),
            'S/G2': [1] * len(yorfs),
            'Metaphase': [1] * len(yorfs),
            'Anaphase': [1] * len(yorfs),
            'Telophase': [1] * len(yorfs)
        }) #if we don't want to take in account concentrations we just set to 1

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



def final_dynamic_dataset(dynamic_localizations_path, embeddings_path, extrem_sequences_path, TE_levels_path, TL_levels_path, static_localization_path, local_interactions_path, output_file_path):
    """
    Assemble the complete dynamic dataset ussing all the dynamic data

    Args:
    - dynamic_localizations_path : path (str) to a file containing the localizations of the proteins in each phases
    - embeddings_path : path (str) to a file containing the embeddings for all the protein sequences
    - extrem_sequences_path : path (str) to a file containing the first and last 20 amminoacids for each sequence
    - TE_levels_path : path (str) to a file containing the TE concentration for each protein
    - TL_levels_path : path (str) to a file containing the TL concentration for each protein
    - static_localization_path : path (str) to a file containing the static localization
    - local_interactions_path : path (str) to a file containing the interaction between the different proteins
    - output_file_path : path (str) to a file in which save the final dynamic dataset 

    Returns:
    - data : pandas DataFrame containing the final dynamic dataset
    """
    embeddings = pd.read_csv(embeddings_path)
    # Rename the embeddings columns
    embeddings.columns = ['yORF'] + [f'emb{i}' for i in range(1, 641)]
    
    ext_sequence = pd.read_csv(extrem_sequences_path)
    concentration_TE = pd.read_csv(TE_levels_path)
    concentration_TL = pd.read_csv(TL_levels_path)
    
    # Rename the columns in concentration_TE and concentration_TL
    concentration_TE = concentration_TE.rename(columns={
    'S/G2': 'S/G2_TE',
    'Metaphase': 'Metaphase_TE',
    'Telophase': 'Telophase_TE',
    'G1 Post-START': 'G1_Post_START_TE',
    'G1 Pre-START': 'G1_Pre_START_TE',
    'Anaphase': 'Anaphase_TE'
    })
    
    concentration_TL = concentration_TL.rename(columns={
    'S/G2': 'S/G2_TL',
    'Metaphase': 'Metaphase_TL',
    'Telophase': 'Telophase_TL',
    'G1 Post-START': 'G1_Post_START_TL',
    'G1 Pre-START': 'G1_Pre_START_TL',
    'Anaphase': 'Anaphase_TL'
    })
    
    local_interaction = pd.read_csv(local_interactions_path)
    dynamic_localizations = pd.read_csv(dynamic_localizations_path)
    dynamic_localizations = dynamic_localizations.rename(columns={
    'S/G2': 'S/G2_localization',
    'Metaphase': 'Metaphase_localization',
    'Telophase': 'Telophase_localization',
    'G1 Post-START': 'G1_Post_START_localization',
    'G1 Pre-START': 'G1_Pre_START_localization',
    'Anaphase': 'Anaphase_localization'
    })
    
    static_localizations = pd.read_csv(static_localization_path)
    static_localizations = static_localizations.rename(columns={
    'localization': 'static_localization',
    })
    
    #setting Nan in TE, TL to mean
    yorfs = static_localizations['yORF']
    concentration_TE = pd.merge(yorfs, concentration_TE, on='yORF', how='left')
    concentration_TE.fillna(concentration_TE.mean(numeric_only=True), inplace=True)
    concentration_TL = pd.merge(yorfs, concentration_TL, on='yORF', how='left')
    concentration_TL.fillna(concentration_TL.mean(numeric_only=True), inplace=True)
    
    # Merge the data
    data = pd.merge(embeddings, ext_sequence, on='yORF')
    data = pd.merge(data, concentration_TL, on='yORF')
    data = pd.merge(data, concentration_TE, on='yORF')
    data = pd.merge(data, local_interaction, on='yORF')
    data = pd.merge(data, static_localizations, on='yORF')
    data = pd.merge(data, dynamic_localizations, on='yORF')
    
    create_csv(data, output_file_path)
    return data