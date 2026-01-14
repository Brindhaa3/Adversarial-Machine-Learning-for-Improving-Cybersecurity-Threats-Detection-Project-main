import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

# Paths
DATA_PATH = "C:/Users/dell/Desktop/Adversarial_ML_Cybersecurity/02_data/NSL_KDD_train_preprocessed.csv"
MODEL_PATH = "C:/Users/dell/Desktop/Adversarial_ML_Cybersecurity/03_models/saved/neural_network.pth"

# Define Simple Neural Network
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

# Load or Retrain Model
def load_or_train_model(model_path, input_size, num_classes, X_train, y_train):
    model = SimpleNN(input_size, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        checkpoint_classes = checkpoint['fc2.weight'].shape[0]

        if checkpoint_classes == num_classes:
            model.load_state_dict(checkpoint)
            print(" Model loaded successfully.")
        else:
            print(f" Model was trained with {checkpoint_classes} classes, but dataset has {num_classes} classes.")
            print(" Retraining model with new class count...")

            # Reinitialize last layer
            model.fc2 = nn.Linear(64, num_classes)

            # Train new model
            train_model(model, optimizer, criterion, X_train, y_train, model_path)
    else:
        print(" No existing model found. Training a new model...")
        train_model(model, optimizer, criterion, X_train, y_train, model_path)

    model.eval()
    return model

# Train Model
def train_model(model, optimizer, criterion, X_train, y_train, model_path, epochs=10):
    print(" Training model...")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    print(" Model training complete. Saving model...")
    torch.save(model.state_dict(), model_path)
    print(f" Model saved at {model_path}")

# Main Execution
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, num_classes = preprocess_data(DATA_PATH)
    model = load_or_train_model(MODEL_PATH, X_train.shape[1], num_classes, X_train, y_train)

    print(" Model ready for adversarial attack.")
