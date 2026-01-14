import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Configure logging
logging.basicConfig(level=logging.INFO)

def train_and_save_rf_model(model_path="03_models/saved/random_forest.pkl"):
    logging.info("Training Random Forest model...")

    # Fix: Increase n_informative to at least 3
    X, y = make_classification(
        n_samples=5000, 
        n_features=41, 
        n_informative=5,  
        n_redundant=2,
        n_classes=5,  
        random_state=42
    )

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    logging.info(f"Random Forest model saved at {model_path}")

if __name__ == "__main__":
    train_and_save_rf_model()
