from utils.graph_with_physicochemical_properties import create_graph_with_properties, visualize_graph
from PIL import Image

main_folder = "C:/Users/Asus/OneDrive/Desktop/Project_implementation/Visualize peptide/"
# Define amino acid groups
amino_acid_groups = {
    'G': 'G1', 'A': 'G1', 'V': 'G1', 'L': 'G1', 'M': 'G1', 'I': 'G1',  # Aliphatic Group
    'F': 'G2', 'Y': 'G2', 'W': 'G2',                                   # Aromatic Group
    'K': 'G3', 'R': 'G3', 'H': 'G3',                                   # Positive Charge Group
    'D': 'G4', 'E': 'G4',                                              # Negative Charge Group
    'S': 'G5', 'T': 'G5', 'C': 'G5', 'P': 'G5', 'N': 'G5', 'Q': 'G5'   # Uncharged Group
}

# Example peptide sequence (from filtered data)
peptide_sequence = "SYSMEHFRWGKPVGKKRRPVKKYLKKGA"

# Create the graph with physicochemical properties
G = create_graph_with_properties(peptide_sequence, amino_acid_groups)

# Visualize the graph
out_folder = main_folder + 'peptide_graph.png'

visualize_graph(G, title="Peptide Graph with Physicochemical Properties", save_path=out_folder)


