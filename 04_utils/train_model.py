import logging
import os
import sys
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Ensure `04_utils` is accessible
sys.path.append(os.path.abspath("04_utils"))

try:
    from preprocess import preprocess_data  # Import the preprocessing function
except ImportError as e:
    logging.error(f"Error importing preprocess module: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
DATA_PATH = "02_data/NSL_KDD_train.csv"
PREPROCESSED_DATA_PATH = "02_data/NSL_KDD_train_preprocessed.csv"
MODEL_DIR = "03_models/saved"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")


def train_model():
    """Loads and preprocesses data, trains a RandomForest model, and saves it."""

    logging.info("Starting model training...")

    # Ensure dataset exists
    if not os.path.exists(DATA_PATH):
        logging.error(f"Dataset not found: {DATA_PATH}")
        return

    # Preprocess data
    try:
        preprocess_data(DATA_PATH, save_path=PREPROCESSED_DATA_PATH)
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        return

    # Load preprocessed dataset
    try:
        df = pd.read_csv(PREPROCESSED_DATA_PATH)
    except Exception as e:
        logging.error(f"Error loading preprocessed data: {e}")
        return

    # Split features and labels
    X, y = df.iloc[:, :-1], df.iloc[:, -1]  # Assuming last column is the label

    # FIX: Convert continuous labels to categorical (classification requires discrete labels)
    if y.dtype in ["float64", "int64"]:
        y = y.round().astype(int)  # Convert float labels to integers
        y = y.astype("category")   # Ensure labels are categorical

    logging.info(f"Target labels converted: {y.unique()}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    logging.info("Training Random Forest model...")
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return

    # Save trained model
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model training completed. Model saved at: {MODEL_PATH}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")


if __name__ == "__main__":
    train_model()
