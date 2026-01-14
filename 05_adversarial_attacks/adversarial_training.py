import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Paths
DATA_PATH = "C:/Users/dell/Desktop/Adversarial_ML_Cybersecurity/02_data/NSL_KDD_train_preprocessed.csv"
MODEL_PATH = "C:/Users/dell/Desktop/Adversarial_ML_Cybersecurity/03_models/saved/adversarial_trained_model.pth"

# Define Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# FGSM Attack
def fgsm_attack(model, X, y, epsilon=0.1):
    X.requires_grad = True
    outputs = model(X)
    loss = nn.CrossEntropyLoss()(outputs, y)
    model.zero_grad()
    loss.backward()

    # Create adversarial example
    X_adv = X + epsilon * X.grad.sign()
    X_adv = torch.clamp(X_adv, 0, 1)  # Keep within valid range
    return X_adv.detach()

# Load & Preprocess Data
def preprocess_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Convert labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Unique classes
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    print(f"Unique class labels in dataset: {unique_classes}")
    print(f"Final detected classes: {num_classes}")

    if num_classes == 0:
        raise ValueError("No valid classes detected. Check dataset labels!")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, num_classes

# Train with Adversarial Examples
def train_adversarial_model(model, optimizer, criterion, X_train, y_train, model_path, epsilon=0.1, epochs=10):
    print(" Adversarial training in progress...")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Standard forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Generate adversarial examples
        X_adv = fgsm_attack(model, X_train_tensor, y_train_tensor, epsilon)

        # Forward pass on adversarial examples
        outputs_adv = model(X_adv)
        loss_adv = criterion(outputs_adv, y_train_tensor)

        # Combine losses
        total_loss = (loss + loss_adv) / 2
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}")

    print(" Adversarial training complete. Saving model...")
    torch.save(model.state_dict(), model_path)
    print(f" Model saved at {model_path}")

# Main Execution
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, num_classes = preprocess_data(DATA_PATH)
    
    # Initialize model
    model = SimpleNN(X_train.shape[1], num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train with adversarial examples
    train_adversarial_model(model, optimizer, criterion, X_train, y_train, MODEL_PATH, epsilon=0.1, epochs=10)

    print(" Adversarially trained model is ready.")
