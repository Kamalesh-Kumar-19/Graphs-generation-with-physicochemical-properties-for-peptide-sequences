import networkx as nx
import matplotlib.pyplot as plt

amino_acid_properties = {
    'A': {'hydrophobicity': 0.61, 'alpha_CH_chemical_shifts': 4.35, 'positive_charge': 0, 'negative_charge': 0, 'polarity': -0.06},
    'C': {'hydrophobicity': 1.07, 'alpha_CH_chemical_shifts': 4.65, 'positive_charge': 0, 'negative_charge': 0, 'polarity': 1.36},
    'D': {'hydrophobicity': 0.46, 'alpha_CH_chemical_shifts': 4.76, 'positive_charge': 0, 'negative_charge': 1, 'polarity': -0.80},
    'E': {'hydrophobicity': 0.47, 'alpha_CH_chemical_shifts': 4.29, 'positive_charge': 0, 'negative_charge': 1, 'polarity': -0.77},
    'F': {'hydrophobicity': 2.02, 'alpha_CH_chemical_shifts': 4.66, 'positive_charge': 0, 'negative_charge': 1, 'polarity': 1.27},
    'G': {'hydrophobicity': 0.07, 'alpha_CH_chemical_shifts': 3.97, 'positive_charge': 0, 'negative_charge': 0, 'polarity': -0.41},
    'H': {'hydrophobicity': 0.61, 'alpha_CH_chemical_shifts': 4.63, 'positive_charge': 1, 'negative_charge': 0, 'polarity': 0.49},
    'I': {'hydrophobicity': 2.22, 'alpha_CH_chemical_shifts': 3.95, 'positive_charge': 0, 'negative_charge': 0, 'polarity': 1.31},
    'K': {'hydrophobicity': 1.15, 'alpha_CH_chemical_shifts': 4.36, 'positive_charge': 1, 'negative_charge': 0, 'polarity': -1.18},
    'L': {'hydrophobicity': 1.53, 'alpha_CH_chemical_shifts': 4.17, 'positive_charge': 0, 'negative_charge': 0, 'polarity': 1.21},
    'M': {'hydrophobicity': 1.18, 'alpha_CH_chemical_shifts': 4.52, 'positive_charge': 0, 'negative_charge': 0, 'polarity': 1.27},
    'N': {'hydrophobicity': 0.06, 'alpha_CH_chemical_shifts': 4.75, 'positive_charge': 0, 'negative_charge': 0, 'polarity': -0.48},
    'P': {'hydrophobicity': 0.00, 'alpha_CH_chemical_shifts': 4.44, 'positive_charge': 0, 'negative_charge': 0, 'polarity': 0},
    'Q': {'hydrophobicity': 0.00, 'alpha_CH_chemical_shifts': 4.37, 'positive_charge': 0, 'negative_charge': 0, 'polarity': -0.73},
    'R': {'hydrophobicity': 0.60, 'alpha_CH_chemical_shifts': 4.38, 'positive_charge': 1, 'negative_charge': 1, 'polarity': -0.84},
    'S': {'hydrophobicity': 0.05, 'alpha_CH_chemical_shifts': 4.50, 'positive_charge': 0, 'negative_charge': 0, 'polarity': -0.5},
    'T': {'hydrophobicity': 0.05, 'alpha_CH_chemical_shifts': 4.35, 'positive_charge': 0, 'negative_charge': 1, 'polarity': -0.27},
    'V': {'hydrophobicity': 1.32, 'alpha_CH_chemical_shifts': 3.95, 'positive_charge': 0, 'negative_charge': 0, 'polarity': 1.09},
    'W': {'hydrophobicity': 2.65, 'alpha_CH_chemical_shifts': 4.70, 'positive_charge': 0, 'negative_charge': 0, 'polarity': 0.88},
    'Y': {'hydrophobicity': 1.88, 'alpha_CH_chemical_shifts': 4.60, 'positive_charge': 0, 'negative_charge': 0, 'polarity': 0.33}
}

# Function to create the graph for a given peptide sequence
def create_graph_with_properties(peptide_sequence, amino_acid_groups):
    G = nx.Graph()
    previous_group = None  # Track the last group for connecting group nodes

    for i, aa in enumerate(peptide_sequence):
        aa_group = amino_acid_groups.get(aa, None)
        aa_properties = amino_acid_properties.get(aa, None)

        if aa_group and aa_properties:
            # Create a unique node name for each amino acid occurrence
            amino_acid_node = f"{aa}_{i}" # Ensures that the amino acid has both a group and defined properties

            # Add the amino acid node with its physicochemical properties
            G.add_node(amino_acid_node, label=aa, type='amino_acid', **aa_properties)

            # Add group node if it doesn't exist, initialize with empty properties
            if not G.has_node(aa_group):
                G.add_node(aa_group, label=aa_group, type='group', hydrophobicity=0, alpha_CH_chemical_shifts=0, positive_charge=0, negative_charge=0, polarity=0, count=0)
                # The group node is initialized with zero values for all physicochemical properties and a count of 0 (to track the number of amino acids in this group).

            # Add an edge between the amino acid and its group
            G.add_edge(amino_acid_node, aa_group)

            # Update the group node with the amino acid properties
            group_data = G.nodes[aa_group]
            group_data['hydrophobicity'] += aa_properties['hydrophobicity']
            group_data['alpha_CH_chemical_shifts'] += aa_properties['alpha_CH_chemical_shifts']
            group_data['positive_charge'] += aa_properties['positive_charge']
            group_data['negative_charge'] += aa_properties['negative_charge']
            group_data['polarity'] += aa_properties['polarity']
            group_data['count'] += 1  # Track the number of amino acids in the group

            # Connect groups if they are adjacent
            if previous_group and previous_group != aa_group:
                G.add_edge(previous_group, aa_group)

            # Update previous group
            previous_group = aa_group

    # Calculate average properties for group nodes
    for node in G.nodes(data=True):
        if node[1]['type'] == 'group' and node[1]['count'] > 0:
            node[1]['hydrophobicity'] /= node[1]['count']
            node[1]['alpha_CH_chemical_shifts'] /= node[1]['count']
            node[1]['positive_charge'] /= node[1]['count']
            node[1]['negative_charge'] /= node[1]['count']
            node[1]['polarity'] /= node[1]['count']

    return G

    # node[1] gives you access to the dictionary containing the properties of the node.
    # This allows you to modify or read specific properties of the node, such as hydrophobicity, alpha_CH_chemical_shifts.
    # node[1]['type'] == 'group' checks if the node is a group node (as opposed to an amino acid node).
    # node[1]['count'] > 0 ensures that the group node has amino acids associated with it (the count is greater than zero).

# Function to visualize the graph
def visualize_graph(G, title="Peptide Graph with Properties", save_path=None):
    pos = nx.spring_layout(G)
    node_labels = nx.get_node_attributes(G, 'label')
    
    # Differentiate node colors between amino acids and groups
    node_colors = ['lightgreen' if G.nodes[node].get('type') == 'group' else 'lightblue' for node in G.nodes()]
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_color=node_colors, node_size=500, font_size=10, font_color="black")
    plt.title(title)
    plt.savefig(save_path)
    # plt.show()
