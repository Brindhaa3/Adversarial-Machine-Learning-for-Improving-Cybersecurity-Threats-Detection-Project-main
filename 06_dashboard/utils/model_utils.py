import pickle
import numpy as np

# Load the ML model
def load_model():
    with open("models/trained_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Predict function
def predict(model, features):
    features = np.array(features).reshape(1, -1)  # Reshape input
    prediction = model.predict(features)
    return prediction.tolist()
