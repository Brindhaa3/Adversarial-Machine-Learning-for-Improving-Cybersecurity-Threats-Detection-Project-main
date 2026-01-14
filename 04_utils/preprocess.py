import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

def preprocess_data(file_path, save_path="02_data/NSL_KDD_train_preprocessed.csv"):
    """Loads, encodes categorical features, scales numeric data, and saves preprocessed dataset."""

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print("Dataset Loaded Successfully! Shape:", df.shape)

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"Found {len(categorical_cols)} Categorical Columns: {list(categorical_cols)}")

    # Convert categorical data to numeric using Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

    print("Categorical Columns Encoded Successfully!")

    # Apply MinMaxScaler (only on numeric columns)
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns].astype(float))

    print("Scaling Completed!")

    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save preprocessed data
    df.to_csv(save_path, index=False)
    print(f"Preprocessing Completed! File saved at {save_path}")

    # Show sample of processed data
    print("\nFirst 5 rows of preprocessed data:\n", df.head())

# Run preprocessing
if __name__ == "__main__":
    preprocess_data("02_data/NSL_KDD_train.csv")
