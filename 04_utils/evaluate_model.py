import os
import pickle
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
MODEL_PATH = "03_models/saved/random_forest.pkl"
TEST_DATA_PATH = "02_data/NSL_KDD_train_preprocessed.csv"

def load_model():
    """Load the trained model from file."""
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found: {MODEL_PATH}")
        return None

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully.")
    return model

def load_test_data():
    """Load and prepare test data."""
    if not os.path.exists(TEST_DATA_PATH):
        logging.error(f"Test data file not found: {TEST_DATA_PATH}")
        return None, None

    df = pd.read_csv(TEST_DATA_PATH)
    X_test = df.iloc[:, :-1]  # Drop label column
    y_test = df.iloc[:, -1]   # Last column is the label
    
    # Convert labels to integer (binary classification)
    y_test = (y_test >= 0.5).astype(int)  # Ensure labels are 0 or 1
    
    logging.info(f"Test data loaded successfully: {X_test.shape[0]} samples")
    return X_test, y_test

def evaluate_model():
    """Evaluate the trained model on the test dataset."""
    model = load_model()
    if model is None:
        return

    X_test, y_test = load_test_data()
    if X_test is None or y_test is None:
        return

    # Predictions
    y_pred = model.predict(X_test)
    
    # Convert predictions to integer (if needed)
    y_pred = (y_pred >= 0.5).astype(int)  

    # Performance Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")

    logging.info(f"Model Evaluation:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")

    # Print Classification Report
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
