from create_embeddings import *
import os


# Paths
current_dir = os.path.dirname(os.path.abspath(__file__)) 
output_embeddings_path = os.path.join(current_dir, '../datasets/yORF_embeddings_640.csv')
output_ext_embeddings_path = os.path.join(current_dir, '../datasets/yORF_extrem_embeddings_640.csv')
sequences_path = os.path.join(current_dir, '../datasets/yORF_sequences.csv')

# Create the file containing yORF and embeddings
print('generating global sequences embeddings: this may take a long time')
create_protein_embeddings(sequences_path, output_embeddings_path, chunk_size=5, small_embedder=False, verbose=True)


# Create the file containing yORF and embeddings
print('generating sequences extremities embeddings: this may take a long time')
create_extremities_embeddings(sequences_path, output_ext_embeddings_path, extremities_len=20, chunk_size=100, small_embedder=False, verbose=True)