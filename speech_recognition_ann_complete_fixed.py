
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Setup output directory for saving plots (cross-platform)
# ==============================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Plots will be saved to: {OUTPUT_DIR}")

# ==============================================================================
# Part 1: Audio Preprocessing & Feature Extraction (Mock Data)
# ==============================================================================

print("Simulating dataset (35 classes, 2574 flattened features per sample)...")
num_samples = 2000
mock_feats = np.random.randn(num_samples, 99, 26).astype(np.float32)  # 99x26 = 2574
mock_labels = np.random.randint(0, 35, num_samples)

# Flatten 2D features into a 1D vector: (num_samples, time, filters) -> (num_samples, time * filters)
feats_flattened = mock_feats.reshape(mock_feats.shape[0], -1)
num_features = feats_flattened.shape[1]
print(f"Shape of flattened feature matrix: {feats_flattened.shape}")

# ==============================================================================
# Part 2: Dataset Splitting & PyTorch DataLoaders
# ==============================================================================

# Step 1: Split into Train (80%) and Temporary (20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    feats_flattened, mock_labels, train_size=0.8, random_state=42, stratify=mock_labels
)

# Step 2: Split Temporary into Validation (10% total) and Test (10% total)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, train_size=0.5, random_state=42, stratify=y_temp
)

# Convert NumPy arrays into PyTorch Tensors and create TensorDatasets
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))

# Create DataLoaders to feed data in mini-batches
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"DataLoaders created!")
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

# ==============================================================================
# Part 3: Baseline Model Architecture (for comparison)
# ==============================================================================

class SpeechBaselineNet(nn.Module):
    def __init__(self, input_dim=2574, num_classes=35):
        super(SpeechBaselineNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)   # Hidden Layer 1
        self.fc2 = nn.Linear(256, 120)        # Hidden Layer 2
        self.fc3 = nn.Linear(120, num_classes) # Output Layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# ==============================================================================
# Part 4: Optimized Model Architecture (Assignment Requirement)
# ==============================================================================
# Required modifications:
# - Additional linear layer (deeper network)
# - Dropout layers (p=0.3) inside hidden transitions
# - Adam optimizer (lr=0.001)
# - 50 epochs

class OptimizedSpeechNet(nn.Module):
    def __init__(self, input_dim=2574, num_classes=35):
        super(OptimizedSpeechNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

# ==============================================================================
# Hardware & Helper Functions
# ==============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computing Device: {device}")

def train_model(model, optimizer, criterion, epochs, train_loader, val_loader, model_name="Model"):
    """Explicit PyTorch training and validation loop."""
    train_losses = []
    val_losses = []
    val_accuracies = []

    print(f"\nStarting training loop for {model_name}...")
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc = correct / total

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")

    return train_losses, val_losses, val_accuracies

def evaluate_test(model, test_loader, model_name="Model"):
    """Terminal analysis against the clean Test Loader."""
    model.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    test_accuracy = np.sum(np.array(test_preds) == np.array(test_targets)) / len(test_targets)
    print(f"\n{model_name} — Unbiased Final Test Dataset Accuracy: {test_accuracy*100:.2f}%")
    return test_targets, test_preds, test_accuracy

def plot_diagnostics(train_losses, val_losses, val_accuracies, model_name="Model"):
    """Part 5: Diagnostics — Loss and Accuracy Visualization"""
    plt.figure(figsize=(14, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='royalblue', lw=2)
    plt.plot(val_losses, label='Validation Loss', color='crimson', lw=2)
    plt.title(f'{model_name} — Loss Convergence Profiles', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value (NLL)')
    plt.legend()
    plt.grid(True, linestyle='--')

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='forestgreen', lw=2)
    plt.title(f'{model_name} — Validation Accuracy Progression', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Decimal Percentage')
    plt.legend()
    plt.grid(True, linestyle='--')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'{model_name.lower().replace(" ", "_")}_diagnostics.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved diagnostics plot to: {save_path}")

def plot_confusion_matrix(test_targets, test_preds, model_name="Model"):
    """Part 6: Test Evaluation and Confusion Matrices"""
    cm = confusion_matrix(test_targets, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
    plt.title(f'{model_name} — Confusion Matrix (Speech Commands)', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Class Indices')
    plt.ylabel('True Class Indices')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved confusion matrix to: {save_path}")

# ==============================================================================
# RUN BASELINE MODEL (SGD, 30 epochs, shallow)
# ==============================================================================

baseline_model = SpeechBaselineNet(input_dim=num_features, num_classes=35).to(device)
print("\n" + "="*60)
print("BASELINE MODEL ARCHITECTURE")
print("="*60)
print(baseline_model)

baseline_optimizer = optim.SGD(baseline_model.parameters(), lr=0.005, momentum=0.9)
criterion = nn.NLLLoss()

bl_train_losses, bl_val_losses, bl_val_accuracies = train_model(
    baseline_model, baseline_optimizer, criterion, epochs=30,
    train_loader=train_loader, val_loader=val_loader, model_name="Baseline"
)

bl_targets, bl_preds, bl_acc = evaluate_test(baseline_model, test_loader, model_name="Baseline")
plot_diagnostics(bl_train_losses, bl_val_losses, bl_val_accuracies, model_name="Baseline")
plot_confusion_matrix(bl_targets, bl_preds, model_name="Baseline")

# ==============================================================================
# RUN OPTIMIZED MODEL (Adam, 50 epochs, deeper, Dropout)
# ==============================================================================

optimized_model = OptimizedSpeechNet(input_dim=num_features, num_classes=35).to(device)
print("\n" + "="*60)
print("OPTIMIZED MODEL ARCHITECTURE")
print("="*60)
print(optimized_model)

optimized_optimizer = optim.Adam(optimized_model.parameters(), lr=0.001)

opt_train_losses, opt_val_losses, opt_val_accuracies = train_model(
    optimized_model, optimized_optimizer, criterion, epochs=50,
    train_loader=train_loader, val_loader=val_loader, model_name="Optimized"
)

opt_targets, opt_preds, opt_acc = evaluate_test(optimized_model, test_loader, model_name="Optimized")
plot_diagnostics(opt_train_losses, opt_val_losses, opt_val_accuracies, model_name="Optimized")
plot_confusion_matrix(opt_targets, opt_preds, model_name="Optimized")

# ==============================================================================
# Final Summary
# ==============================================================================

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Baseline Test Accuracy:  {bl_acc*100:.2f}%")
print(f"Optimized Test Accuracy: {opt_acc*100:.2f}%")
print(f"Improvement:             {(opt_acc - bl_acc)*100:.2f} percentage points")
print("="*60)
