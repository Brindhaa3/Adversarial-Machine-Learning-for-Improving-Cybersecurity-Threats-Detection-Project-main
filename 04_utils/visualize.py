import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed dataset
file_path = "02_data/NSL_KDD_train_preprocessed.csv"

try:
    df = pd.read_csv(file_path)
    print("Preprocessed dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: Preprocessed file '{file_path}' not found! Run preprocessing first.")
    exit()

# Example visualization: Plot class distribution
plt.figure(figsize=(10, 5))
sns.countplot(x=df.iloc[:, -1])  # Assuming last column is the target label
plt.title("Class Distribution in Preprocessed Data")
plt.xlabel("Attack Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
