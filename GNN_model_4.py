import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.graph_with_physicochemical_properties import create_graph_with_properties
from utils.convert_graph_to_pytorch_geometric import convert_to_pytorch_geometric
from scipy import stats
import seaborn as sns
from tqdm import tqdm 
import glob 
import os

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(5, 16)  # Input is 5 features 
        self.conv2 = GCNConv(16, 32)  # Increase hidden layer size to 32 for richer representations
        self.conv3 = GCNConv(32, 64)  # Add a third GCN layer for more complexity
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

        # Apply global mean pooling to aggregate node features into a graph-level representation
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

# Modified Training function to track losses and accuracies for each seed
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
            loss = F.nll_loss(out, data.y, weight=class_weights)  # Use class weights
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(correct / total)

        # Validation phase
        val_loss, val_accuracy = evaluate_model(model, val_loader, class_weights)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    return model, train_losses, val_losses, train_accuracies, val_accuracies

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
    auc = roc_auc_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)

    return accuracy, precision, recall, f1, auc, mcc

# Function to calculate confidence intervals
def calculate_confidence_intervals(metric_list, confidence=0.95):
    mean = np.mean(metric_list)
    sem = stats.sem(metric_list)
    interval = sem * stats.t.ppf((1 + confidence) / 2., len(metric_list) - 1)
    return mean, mean - interval, mean + interval

# Function to plot confidence intervals
def plot_confidence_intervals_distribution(metrics, metric_name, save_fig = None):
    plt.figure(figsize=(8, 6))
    mean = np.mean(metrics)
    ci_lower, ci_upper = stats.t.interval(0.95, len(metrics) - 1, loc=mean, scale=stats.sem(metrics))

    # Plot the kernel density estimate (KDE) for a smooth curve
    sns.kdeplot(metrics, color='blue', fill=True, alpha=0.3, label='Density', linewidth=2)
    
    # Plot a vertical line for the mean and the confidence interval bounds
    plt.axvline(mean, color='red', linestyle='--', label=f"Mean: {mean:.4f}")
    plt.axvline(ci_lower, color='green', linestyle='--', label=f"95% CI Lower: {ci_lower:.4f}")
    plt.axvline(ci_upper, color='green', linestyle='--', label=f"95% CI Upper: {ci_upper:.4f}")

    plt.title(f'{metric_name} Distribution with 95% Confidence Interval')
    plt.legend()
    plt.xlabel(metric_name)
    plt.ylabel('Density')
    plt.savefig(save_fig)
    # plt.show()

# Function to plot training and validation metrics for all seeds
def plot_metrics_all_seeds(train_accs, val_accs, train_losses, val_losses, epochs, seeds, save_fig = None):
    plt.figure(figsize=(15, 10))

    # Plot training and validation accuracy for all seeds
    plt.subplot(2, 2, 1)
    for i, seed in enumerate(seeds):
        plt.plot(range(epochs), train_accs[i], label=f'Train Acc (Seed {seed})')
    plt.title("Training Accuracy for All Seeds")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(2, 2, 2)
    for i, seed in enumerate(seeds):
        plt.plot(range(epochs), val_accs[i], label=f'Val Acc (Seed {seed})')
    plt.title("Validation Accuracy for All Seeds")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot training and validation loss for all seeds
    plt.subplot(2, 2, 3)
    for i, seed in enumerate(seeds):
        plt.plot(range(epochs), train_losses[i], label=f'Train Loss (Seed {seed})')
    plt.title("Training Loss for All Seeds")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 4)
    for i, seed in enumerate(seeds):
        plt.plot(range(epochs), val_losses[i], label=f'Val Loss (Seed {seed})')
    plt.title("Validation Loss for All Seeds")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_fig)
    # plt.show()

# Main function
if __name__ == '__main__':
    dataset = glob.glob('dataset/Filtered_data/A*.csv')
    for data_file in dataset:
        data = data_file[22:-4]
        df = pd.read_csv(f'dataset/Filtered_data/{data}.csv')  
        Inference_path = f"C:/Users/Asus/OneDrive/Desktop/Project_implementation/Inferences/Inference_for_{data}/"
        if not os.path.exists(Inference_path):
            os.makedirs(Inference_path)

        # Define amino acid groups
        amino_acid_groups = {
            'G': 'G1', 'A': 'G1', 'V': 'G1', 'L': 'G1', 'M': 'G1', 'I': 'G1',  # Aliphatic Group
            'F': 'G2', 'Y': 'G2', 'W': 'G2',  # Aromatic Group
            'K': 'G3', 'R': 'G3', 'H': 'G3',  # Positive Charge Group
            'D': 'G4', 'E': 'G4',             # Negative Charge Group
            'S': 'G5', 'T': 'G5', 'C': 'G5', 'P': 'G5', 'N': 'G5', 'Q': 'G5'  # Uncharged Group
        }

        # Prepare data
        data_list = prepare_data(df, amino_acid_groups)

        # Store metrics across 10 random seeds
        random_seeds = [754, 832, 987, 124, 432, 997, 187, 41, 999, 555]  # Example seeds

        # Lists to store metrics for each seed
        train_losses_all_seeds = []
        val_losses_all_seeds = []
        train_accuracies_all_seeds = []
        val_accuracies_all_seeds = []
        
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        aucs = []
        mccs = []

        for seed in tqdm(random_seeds, desc="Running model with different random seeds"):
            print(f"\n========Running model for random seed {seed}========")
            print("-----------------------------------------------------")
            torch.manual_seed(seed)
            train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=seed)
            val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=seed)

            # Calculate class weights to handle imbalance
            labels = [data.y.item() for data in train_data]
            class_counts = {0: labels.count(0), 1: labels.count(1)}

            if class_counts[0] == 0 or class_counts[1] == 0:
                class_weights = torch.tensor([1.0, 1.0], dtype=torch.float)
            else:
                class_weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float)

            # Create DataLoaders
            train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

            # Initialize model and optimizer
            model = GNN()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Train the model and collect losses and accuracies
            model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
                model, train_loader, val_loader, optimizer, class_weights
            )

            # Store the losses and accuracies for this seed
            train_losses_all_seeds.append(train_losses)
            val_losses_all_seeds.append(val_losses)
            train_accuracies_all_seeds.append(train_accuracies)
            val_accuracies_all_seeds.append(val_accuracies)

            # Evaluate metrics on test set
            accuracy, precision, recall, f1, auc, mcc = evaluate_metrics(model, test_loader)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            aucs.append(auc)
            mccs.append(mcc)

            print(f"Metrics for seed {seed}: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}")

        # Calculate and print confidence intervals
        metrics = {
            'Accuracy': accuracies,
            'Precision': precisions,
            'Recall': recalls,
            'F1 Score': f1_scores,
            'AUC': aucs,
            'MCC': mccs
        }

        for metric_name, metric_values in metrics.items():
            mean, ci_lower, ci_upper = calculate_confidence_intervals(metric_values, confidence=0.95)
            print(f"{metric_name}: Mean = {mean:.4f}, 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            # Plot confidence interval distribution for each metric
            plot_confidence_intervals_distribution(metric_values, metric_name, save_fig= Inference_path + f'confidence_Interval_for_{metric_name}')
        # Plot all training and validation losses and accuracies for different seeds
        epochs = 100  
        plot_metrics_all_seeds(train_accuracies_all_seeds, val_accuracies_all_seeds, train_losses_all_seeds, val_losses_all_seeds, epochs, random_seeds, save_fig= Inference_path + f'Train_val_image')
