from data_cleaning_helpers import *
from data_cleaning_plots import *
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Definition of paths
current_dir = os.path.dirname(os.path.abspath(__file__)) 
file_path = os.path.join(current_dir, '../datasets/S1_protein_location.xlsx')
proteins_path = os.path.join(current_dir,'../datasets/orf_trans_all.fasta')
sheet_names = ["PL - All Data Replicate 1", "PL - All Data Replicate 2", "PL - All Data Replicate 3"]
file_path_concentrations = os.path.join(current_dir, '../datasets/S3_protein_level.xlsx')
output_sequences_path = os.path.join(current_dir, '../datasets/yORF_sequences.csv')
output_static_path = os.path.join(current_dir, '../datasets/yORF_localizations.csv')
output_dynamic_path = os.path.join(current_dir, '../datasets/dynamic_localizations.csv')
sheet_names_concentration = ["TL - Data", "TE - Data"]
output_interaction_matrix = os.path.join(current_dir, '../datasets/Interaction_matrix.csv')
embeddings_path = os.path.join(current_dir, '../datasets/yORF_embeddings.csv')
ext_embeddings_path = os.path.join(current_dir, '../datasets/yORF_extrem_embeddings.csv')
output_concentrationTE = os.path.join(current_dir, '../datasets/TE_levels.csv')
output_concentrationTL = os.path.join(current_dir,'../datasets/TL_levels.csv')
interaction_matrix_path = os.path.join(current_dir,'../datasets/The_Yeast_Interactome_Edges.csv')
interaction_matrix_with_TE = os.path.join(current_dir,'../datasets/interaction_matrix_TE.csv')
interaction_matrix_with_TL = os.path.join(current_dir,'../datasets/interaction_matrix_TL.csv')
interaction_matrix_without_concentrations = os.path.join(current_dir,'../datasets/interaction_matrix_without_concentrations.csv')
final_dataset_dyn =  os.path.join(current_dir,'../datasets/final_dataset_dyn.csv')

# Path in which save the figures
output_folder = "plots_output"
os.makedirs(output_folder, exist_ok=True)

# Dynamic and static dataset creation
final_df_with_long, dynamic_dataset, static_dataset = datasets_creation(file_path, sheet_names, proteins_path, output_sequences_path, output_static_path, output_dynamic_path)

# Create the dataset with the concentrations of protein
protein_levels_cleaning(file_path_concentrations, sheet_names_concentration, output_concentrationTE, output_concentrationTL)

# Create the interaction matrix
interaction_mat = interaction_matrix(interaction_matrix_path, static_dataset, output_interaction_matrix, delimiter=';')

# Create the dataset containing the informations about the interactions : AT FIRST CONSIDERING TE concentration level
create_correlation_scores(output_interaction_matrix, output_dynamic_path, output_concentrationTE, interaction_matrix_with_TE, top_n=10, n_classes=15, null_row='zeros')

# Create the dataset containing the informations about the interactions : CONSIDERING TL concentration level
create_correlation_scores(output_interaction_matrix, output_dynamic_path, output_concentrationTL, interaction_matrix_with_TL, top_n=10, n_classes=15, null_row='zeros')

# Create the dataset containing the informations about the interactions : WITHOUT CONCENTRATIONS DATA
create_correlation_scores(output_interaction_matrix, output_dynamic_path, False, interaction_matrix_without_concentrations, top_n=10, n_classes=15, null_row='zeros')

# Create the final dynamic dataset
final_dynamic_dataset(output_dynamic_path, embeddings_path, ext_embeddings_path, output_concentrationTE, output_concentrationTL, output_static_path, output_interaction_matrix, final_dataset_dyn)


# Generation of plots
plot_stationary_proteins(dynamic_dataset)
plt.savefig(os.path.join(output_folder, "stationary_proteins.png"), dpi=300)
plt.close()

plot_stationary_proteins_per_phase(dynamic_dataset)  
plt.savefig(os.path.join(output_folder, "stationary_proteins_per_phase.png"), dpi=300)
plt.close()

plot_transition_matrix(dynamic_dataset)  
plt.savefig(os.path.join(output_folder, "transition_matrix.png"), dpi=300)
plt.close()

plot_individual_transition_matrices(dynamic_dataset) 
plt.savefig(os.path.join(output_folder, "transition_matrix_per_phase.png"), dpi=300)
plt.close()

long_sequences_distribution(final_df_with_long) 
plt.savefig(os.path.join(output_folder, "distribution_long_sequences.png"), dpi=300)
plt.close()

