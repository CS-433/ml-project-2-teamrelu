import esm
import torch
import pandas as pd
        
# You need "pip install fair-esm" and "pip install fairscale"

def create_protein_embeddings(input_file_path, output_file_path, chunk_size=5, small_embedder=False, verbose=False):
    """ 
    Generate and save protein sequence embeddings in chunks

    Parameters:
    input_file_path (str): Path to the input CSV file containing yORF and sequences
    output_file_path (str): Path to save the output CSV file
    chunk_size (int): Number of sequences to process in each chunk
    small_embedder (bool): Whether to use a smaller ESM-2 model for faster processing (320-dim). If false it uses a 640-dim embedding
    verbose (bool): Whether to print progress messages

    Returns: None
    """
    # Load ESM-2 model
    if not small_embedder:
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        final_idx = 30 # index of last layer for extracting embeddings
        emb_len = 640
    else:
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        final_idx = 6 # index of last layer for extracting embeddings
        emb_len = 320

    # Set model to evaluation mode
    model.eval()

    # Extract data
    data = pd.read_csv(input_file_path)
    n_data = data.iloc[:, 0].size
    
    # Create and save embeddings
    for i in range(0, n_data-chunk_size+1, chunk_size):
        chunk=data.iloc[i:i+chunk_size, :]
        chunk = [tuple(row) for row in chunk.values]
        mode = 'w' if i==0 else 'a'
        save_chunk_embeddings(chunk, model, alphabet, batch_converter, final_idx, output_file_path, mode, emb_len)
        if verbose:
            print(f"Completed {i+chunk_size} of {n_data}")

    # Processing last proteins if n_data is not multiple of chunk_size
    if n_data%chunk_size != 0:
        chunk = data.iloc[n_data - n_data%chunk_size:, :]
        chunk = [tuple(row) for row in chunk.values]
        mode = 'a' if n_data > chunk_size else 'w'
        save_chunk_embeddings(chunk, model, alphabet, batch_converter, final_idx, output_file_path, mode, emb_len)
        if verbose:
            print(f"Completed {n_data} of {n_data}")


def create_extremities_embeddings(input_file_path, output_file_path, extremities_len, chunk_size=100, small_embedder=False, verbose=False):
    """
    Generate and save embeddings for the first and last residues of sequences.

    Parameters:
        input_file_path (str): Path to the input CSV file containing sequences
        output_file_path (str): Path to save the output CSV file
        extremities_len (int): Length of the sequence extremities to extract
        chunk_size (int): Number of sequences to process in each chunk
        small_embedder (bool): Whether to use a smaller ESM-2 model for faster processing (320-dim). If false it uses a 640-dim embedding
        verbose (bool): Whether to print progress messages

    Returns: None
"""

    # Load ESM-2 model
    if not small_embedder:
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        final_idx = 30 # index of last layer for extracting embeddings
        emb_len = 640
    else:
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        final_idx = 6 # index of last layer for extracting embeddings
        emb_len = 320

    # Set model to evaluation mode
    model.eval()

    # Extract data
    df = pd.read_csv(input_file_path)
    n_data = df.iloc[:, 0].size
    data = ((seq[:extremities_len], seq[-extremities_len:]) for seq in df.iloc[:, -1])
    seq_extremities = pd.DataFrame(data, columns = ["beginning", "end"])
    seq_extremities.insert(0, "yORF", df.iloc[:, 0])
    for i in range(0, n_data-chunk_size+1, chunk_size):
        chunk = seq_extremities.iloc[i:i+chunk_size, :]
        #starting part
        beginning = chunk.iloc[:, [0]+[1]]
        beginning = [tuple(row) for row in beginning.values]
        #ending part
        end = chunk.iloc[:, [0]+[2]]
        end=[tuple(row) for row in end.values]
        #creating and savinging chunk embeddings
        mode = 'w' if i==0 else 'a'
        save_chunk_extremities(beginning, end, model, alphabet, batch_converter, final_idx, output_file_path, mode, emb_len)
        if verbose:
            print(f"Completed {i+chunk_size} of {n_data}")

    # Processing last proteins if n_data is not multiple of chunk_size
    if n_data%chunk_size != 0:
        beginning = chunk.iloc[:, [0]+[1]]
        beginning = [tuple(row) for row in beginning.values]
        end = chunk.iloc[:, [0]+[2]]
        end=[tuple(row) for row in end.values]
        mode = 'a' if n_data > chunk_size else 'w'
        save_chunk_extremities(beginning, end, model, alphabet, batch_converter, final_idx, output_file_path, mode, emb_len)
        if verbose:
            print(f"Completed {n_data} of {n_data}")


# Helper function to get embeddings of a chunk of data
def get_chunk_embeddings(data, model, alphabet, batch_converter, final_idx):
    """ Extract sequence embeddings from a chunk of data using the ESM-2 model

    Parameters:
        data (list): List of tuples containing (label, sequence)
        model (nn.Module): Pretrained ESM-2 model
        alphabet (Alphabet): Alphabet used by the model
        batch_converter (function): Function to convert data into model inputs
        final_idx (int): Index of the layer to extract embeddings from

    Returns:
        batch_labels (list): List of sequence labels
        sequence_representations (list): List of tensor embeddings for each sequence
    """

    batch_labels, _ , batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[final_idx], return_contacts=False)

    token_representations = results["representations"][final_idx]
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    sequence_representations=[]
    for i, tokens_len in enumerate(batch_lens):
      sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    return batch_labels, sequence_representations


# Helper function to save embeddings of a chunk of data
def save_chunk_embeddings(data, model, alphabet, batch_converter, final_idx, output_file_path, mode, emb_len):
    """
    Save sequence embeddings for a chunk of data to a CSV file

    Parameters:
        data (list): List of tuples containing (label, sequence)
        model (nn.Module): Pretrained ESM-2 model
        alphabet (Alphabet): Alphabet used by the model
        batch_converter (function): Function to convert data into model inputs
        final_idx (int): Index of the layer to extract embeddings from
        output_file_path (str): Path to save the output CSV file
        mode (str): Write ('w') or append ('a') mode for the CSV file
        emb_len (int): Length of the embedding vectors

    Returns: None
    """

    batch_labels, sequence_representation = get_chunk_embeddings(data, model, alphabet, batch_converter, final_idx)

    sequence_representation_list = [tensor.squeeze().tolist() for tensor in sequence_representation]

    #Create a DataFrame with columns "protein_id" and "embedding"
    embeddings = pd.DataFrame(sequence_representation_list)
    embeddings.insert(0, 'protein_id', batch_labels)

    if mode == 'w':
        headers = ['yORF'] + [f'emb{i}' for i in range(1, emb_len + 1)]
        embeddings.columns = headers
        embeddings.to_csv(output_file_path, index=False, header=True, mode='w')
    elif mode == 'a':
        embeddings.to_csv(output_file_path, index=False, header=False, mode='a')


def save_chunk_extremities(begin, end, model, alphabet, batch_converter, final_idx, output_file_path, mode, emb_len):
    """
    Save extremities sequence embeddings for a chunk of data to a CSV file

    Parameters:
        begin (list): List of tuples containing (label, sequence) for the beginning part of the sequence
        end (list): List of tuples containing (label, sequence) for the end part of the sequence
        model (nn.Module): Pretrained ESM-2 model
        alphabet (Alphabet): Alphabet used by the model
        batch_converter (function): Function to convert data into model inputs
        final_idx (int): Index of the layer to extract embeddings from
        output_file_path (str): Path to save the output CSV file
        mode (str): Write ('w') or append ('a') mode for the CSV file
        emb_len (int): Length of the embedding vectors

    Returns: None
    """
    
    batch_labels, begin_representation = get_chunk_embeddings(begin, model, alphabet, batch_converter, final_idx)
    begin_representation_list = [tensor.squeeze().tolist() for tensor in begin_representation]
    _, end_representation = get_chunk_embeddings(end, model, alphabet, batch_converter, final_idx)
    end_representation_list = [tensor.squeeze().tolist() for tensor in end_representation]

    df_b = pd.DataFrame(begin_representation_list, columns=[f'emb_b{i+1}' for i in range(emb_len)])
    df_e = pd.DataFrame(end_representation_list, columns=[f'emb_e{i+1}' for i in range(emb_len)])

    emb_extremities = pd.concat([df_b, df_e], axis=1)
    emb_extremities.insert(0, "yORF", batch_labels)

    if mode == 'w':
        headers = ['yORF'] + [f'emb_b{i+1}' for i in range(emb_len)] + [f'emb_e{i+1}' for i in range(emb_len)]
        emb_extremities.columns = headers
        emb_extremities.to_csv(output_file_path, index=False, header=True, mode='w')
    elif mode == 'a':
        emb_extremities.to_csv(output_file_path, index=False, header=False, mode='a')