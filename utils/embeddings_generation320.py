from create_embeddings import *
import os


# Paths
current_dir = os.path.dirname(os.path.abspath(__file__)) 
output_embeddings_path = os.path.join(current_dir, '../datasets/yORF_embeddings_320.csv')
output_ext_embeddings_path = os.path.join(current_dir, '../datasets/yORF_extrem_embeddings_320.csv')
sequences_path = os.path.join(current_dir, '../datasets/yORF_sequences.csv')

# Create the file containing yORF and embeddings
create_protein_embeddings(sequences_path, output_embeddings_path, chunk_size=5, small_embedder=True, verbose=True)

# Create the file containing yORF and embeddings
create_extremities_embeddings(sequences_path, output_ext_embeddings_path, 20, small_embedder=True)