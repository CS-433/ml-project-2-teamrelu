from utils.data_cleaning_helpers import *
from utils.data_cleaning_plots import *


file_path = 'S1_protein_location.xlsx'
proteins_path = 'orf_trans_all.fasta'
sheet_names = ["PL - All Data Replicate 1", "PL - All Data Replicate 2", "PL - All Data Replicate 3"]
sheet_names_concentration = ["TL - Data", "TE - Data"]

# Dynamic and static dataset creation
final_df_with_long, dynamic_dataset, static_dataset = datasets_creation(file_path, sheet_names, proteins_path)

# Generation of plots
# plot_stationary_proteins(dynamic_dataset)

# plot_stationary_proteins_per_phase(dynamic_dataset)  

# plot_transition_matrix(dynamic_dataset)  

# plot_individual_transition_matrices(dynamic_dataset) 

# long_sequences_distribution(final_df_with_long) 

# plt.show()

matrix = interaction_matrix('The_Yeast_Interactome_Edges.csv', ';', static_dataset)