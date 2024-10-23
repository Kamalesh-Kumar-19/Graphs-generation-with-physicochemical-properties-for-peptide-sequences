import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.graph_with_physicochemical_properties import create_graph_with_properties
from utils.convert_graph_to_pytorch_geometric import convert_to_pytorch_geometric

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(5, 16)  # Input is 5 features (hydrophobicity, molecular_weight, isoelectric_point, charge, stability)
        self.conv2 = GCNConv(16, 32)  # Increase hidden layer size to 32 for richer representations
        self.conv3 = GCNConv(32, 64)  # Third GCN layer for more complexity
        self.fc = torch.nn.Linear(64, 2)  # Fully connected layer for graph-level classification into 2 classes (0 or 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Global mean pooling to aggregate node features into a graph-level representation
        x = global_mean_pool(x, batch)

        # Pass through fully connected layer for classification
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Function to load data and convert to PyTorch Geometric format
def prepare_data(df, amino_acid_groups):
    data_list = []

    for _, row in df.iterrows():
        peptide_sequence = row['sequence']
        label = row['label']

        # Create graph with properties
        G = create_graph_with_properties(peptide_sequence, amino_acid_groups)

        # Convert to PyTorch Geometric Data format
        data = convert_to_pytorch_geometric(G)

        # Add the label
        data.y = torch.tensor([label], dtype=torch.long)

        # Add the sequence as a custom attribute in the Data object
        data.sequence = peptide_sequence  # Adding sequence for inference

        # Append to data list
        data_list.append(data)

    return data_list

# Training function with class weights for class imbalance handling
def train_model(model, train_loader, val_loader, optimizer, class_weights, epochs=100):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Training phase
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)  # Output is graph-level
            loss = F.nll_loss(out, data.y, weight=class_weights)  # Using class weights
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Track accuracy
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(correct / total)

        # Validation phase
        val_loss, val_accuracy = evaluate_model(model, val_loader, class_weights)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies

# Evaluation function for validation and test sets
def evaluate_model(model, loader, class_weights=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            out = model(data)  # Output is graph-level
            if class_weights is not None:
                loss = F.nll_loss(out, data.y, weight=class_weights)
            else:
                loss = F.nll_loss(out, data.y)
            total_loss += loss.item()

            # Track accuracy
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

    accuracy = correct / total
    return total_loss / len(loader), accuracy

# Function to calculate evaluation metrics
def evaluate_metrics(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            out = model(data)  # Output is graph-level
            pred = out.argmax(dim=1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, precision, recall, f1

# Function to plot confusion matrix
def plot_confusion_matrix(model, loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            out = model(data)
            pred = out.argmax(dim=1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

# Plot training and validation accuracy/loss
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = len(train_losses)
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(epochs), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Main function
if __name__ == '__main__':
    # Loadin Dataset
    df = pd.read_csv('dataset/Filtered_data/AB_train_90.csv') 
    
    # Define amino acid groups
    amino_acid_groups = {
        'G': 'G1', 'A': 'G1', 'V': 'G1', 'L': 'G1', 'M': 'G1', 'I': 'G1',  # Aliphatic Group
        'F': 'G2', 'Y': 'G2', 'W': 'G2',                                   # Aromatic Group
        'K': 'G3', 'R': 'G3', 'H': 'G3',                                   # Positive Charge Group
        'D': 'G4', 'E': 'G4',                                              # Negative Charge Group
        'S': 'G5', 'T': 'G5', 'C': 'G5', 'P': 'G5', 'N': 'G5', 'Q': 'G5'   # Uncharged Group
    }

    # Prepare data
    data_list = prepare_data(df, amino_acid_groups)

    # Split data into train (80%), validation (10%), and test (10%)
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=754)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=754)

    # Calculate class weights to handle imbalance
    labels = [data.y.item() for data in train_data]
    class_counts = {0: labels.count(0), 1: labels.count(1)}

    # Avoid division by zero if one class is missing
    if class_counts[0] == 0 or class_counts[1] == 0:
        print("One of the classes is missing in the training data. Using default class weights.")
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float)
    else:
        class_weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    # Initialize model and optimizer
    model = GNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Reduced learning rate for better convergence

    # Train the model and collect metrics
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, optimizer, class_weights, epochs=100
    )

    # Plot training and validation metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    # Evaluate on test set
    test_accuracy, test_precision, test_recall, test_f1 = evaluate_metrics(model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}")

    # Plot confusion matrix for test set
    plot_confusion_matrix(model, test_loader)

