import pandas as pd
from tqdm import tqdm

# Load the Excel file into a DataFrame
df = pd.read_excel("C:/Users/Asus/OneDrive/Desktop/Project_implementation/dataset/AB_train.xlsx")
print(f"Total instances: {len(df)}")

# Function to compute Hamming distance for equal-length sequences
def compute_hamming_distance(seq1, seq2):
    if len(seq1) != len(seq2):
        return 1.0  # Treat sequences of unequal lengths as completely dissimilar
    
    # Count how many positions differ
    return sum(el1 != el2 for el1, el2 in zip(seq1, seq2)) / len(seq1)

# Function to filter sequences based on similarity threshold using Hamming distance
def filter_sequences_debug(df, threshold=0.10):  # Threshold is a difference percentage
    sequences = df['sequence'].tolist()
    filtered_sequences = []
    filtered_indices = set()

    print("Starting sequence filtering...")

    # Progress bar for the outer loop (processing each sequence)
    for i, seq1 in tqdm(enumerate(sequences), total=len(sequences), desc="Filtering sequences"):
        if i in filtered_indices:
            continue
        # Append both the sequence and the associated label to filtered_sequences
        filtered_sequences.append((df.iloc[i]['sequence'], df.iloc[i]['label']))

        # Inner loop for comparing seq1 with other sequences
        for j in tqdm(range(i + 1, len(sequences)), desc=f"Comparing with sequence {i+1}", leave=False):
            if j in filtered_indices:
                continue

            seq2 = sequences[j]

            # Use Hamming distance for similarity calculation
            similarity = compute_hamming_distance(seq1, seq2)

            if similarity <= threshold:  # Use Hamming distance for filtering
                filtered_indices.add(j)

    print("Sequence filtering completed!")
    return filtered_sequences

# Test with a smaller subset first
filtered_sequences = filter_sequences_debug(df)  
print(f"Filtered sequences count: {len(filtered_sequences)}")

# Convert filtered sequences to a DataFrame, preserving both columns
filtered_df = pd.DataFrame(filtered_sequences, columns=['sequence', 'label'])

# Save the filtered sequences to an Excel file
output_file = "C:/Users/Asus/OneDrive/Desktop/Project_implementation/dataset/AB_filtered_sequences.xlsx"
filtered_df.to_excel(output_file, index=False)
print(f"Filtered sequences saved to {output_file}")
