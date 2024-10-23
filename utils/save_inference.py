import torch
import pandas as pd

def save_inference(model, loader, output_file='inference_results.csv'):
    """
    Perform inference using the trained model and save results to a CSV file.

    Parameters:
    model (torch.nn.Module): The trained GNN model.
    loader (DataLoader): The DataLoader for the inference dataset.
    output_file (str): The file path where to save the inference results (default: 'inference_results.csv').

    Returns:
    None
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_sequences = []

    with torch.no_grad():
        for data in loader:
            out = model(data)  # Perform inference (graph-level classification)
            pred = out.argmax(dim=1)  # Get the predicted class
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())  # True labels

            # Assuming data object contains the sequence, add the sequence to results
            all_sequences.extend(data.sequence)  # Add sequence if available in data

    # Create a DataFrame to save the results
    df = pd.DataFrame({
        'Sequence': all_sequences,  # Assuming sequence exists in the data
        'True Label': all_labels,
        'Predicted Label': all_preds
    })

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Inference results saved to {output_file}")

