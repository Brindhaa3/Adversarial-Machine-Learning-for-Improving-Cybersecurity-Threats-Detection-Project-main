import pickle
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_rf_model(model_path="03_models/saved/random_forest.pkl"):
    logging.info("Loading Random Forest model...")

    # Load the model from the .pkl file
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logging.info("Random Forest model loaded successfully!")
    return model

if __name__ == "__main__":
    # Load the model
    model = load_rf_model()

    # Example test input (Random feature values)
    x_test = np.random.rand(1, 41)  # 41 features (same as training data)
    
    # Make a prediction
    prediction = model.predict(x_test)
    
    logging.info(f"Test Prediction: {prediction}")
