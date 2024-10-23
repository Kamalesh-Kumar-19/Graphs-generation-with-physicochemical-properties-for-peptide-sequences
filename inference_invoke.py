import os
import torch
import datetime
import numpy as np
from utils.save_inference import save_inference
from utils.plot_metrics import plot_metrics
from GNN_model import model, test_loader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function to create a unique directory for saving results
def create_results_dir(base_dir="inference_results"):
    # Create a folder with a timestamp for unique naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    return results_dir

# Function to save confusion matrix as an image
def save_confusion_matrix(model, loader, results_dir):
    all_preds = []
    all_labels = []

    # Perform inference
    model.eval()
    with torch.no_grad():
        for data in loader:
            out = model(data)
            pred = out.argmax(dim=1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    # Save the confusion matrix plot as an image
    confusion_matrix_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    print(f"Confusion matrix saved at: {confusion_matrix_path}")

# Function to save any other images, like accuracy/loss plots (if needed)
def save_plot(plot_func, plot_name, results_dir):
    plot_path = os.path.join(results_dir, f"{plot_name}.png")
    plot_func()  # Call the plot function
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved at: {plot_path}")

# Main function to run inference and save results
if __name__ == '__main__':
    # Create a directory to store the results for this run
    results_dir = create_results_dir()

    # Save the inference results (CSV)
    save_inference(model, test_loader, output_file=os.path.join(results_dir, 'test_inference_results.csv'))

    # Save confusion matrix as an image
    save_confusion_matrix(model, test_loader, results_dir)
    
    # If you have accuracy/loss plots you want to save:
    save_plot(plot_metrics, "accuracy_loss", results_dir)
