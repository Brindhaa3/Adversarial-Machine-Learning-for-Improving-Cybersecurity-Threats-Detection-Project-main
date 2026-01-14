import torch
import torch.nn as nn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=41, hidden_size=64, output_size=5):  
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def load_neural_network_model(model_path="03_models/saved/neural_network.pth"):
    logging.info("Loading Neural Network model...")

    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))  # Ensure it runs on CPU
    model.eval()

    logging.info("Neural Network model loaded successfully!")
    return model

if __name__ == "__main__":
    model = load_neural_network_model()
    sample_data = torch.randn(1, 41)  # Example input
    prediction = model(sample_data)
    logging.info(f"Prediction: {prediction}")
