import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

#  Define File Paths (Update these paths if needed)
train_file = "KDDTrain+.txt"  # Place this file in the same folder as your script
test_file = "KDDTest+.txt"

# Load NSL-KDD dataset
def load_data(file_path):
    column_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
                    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
                    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
                    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
                    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
                    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
                    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
                    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                    "dst_host_srv_rerror_rate", "label"]

    data = pd.read_csv(file_path, names=column_names, encoding='ISO-8859-1')
    return data

# Preprocess NSL-KDD dataset
def preprocess_data(data):
    label_encoder = LabelEncoder()
    
    # Encode categorical features
    for col in ["protocol_type", "service", "flag"]:
        data[col] = label_encoder.fit_transform(data[col])

    # Convert labels to binary classification (Normal = 0, Attack = 1)
    data['label'] = data['label'].apply(lambda x: 0 if x == 'normal' else 1)

    # Separate features and labels
    X = data.drop(columns=['label'])
    y = data['label']

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

#  Load and preprocess data
train_data = load_data(train_file)
test_data = load_data(test_file)

X_train, y_train = preprocess_data(train_data)
X_test, y_test = preprocess_data(test_data)

#  Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)

#  Build Neural Network Model
def build_nn_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

#  Train Neural Network IDS
nn_model = build_nn_model(X_train.shape[1])
nn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
loss, nn_acc = nn_model.evaluate(X_test, y_test)

#  Visualization: Accuracy Comparison
models = ['Random Forest', 'Neural Network']
accuracies = [rf_acc, nn_acc]

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=accuracies, palette="coolwarm")
plt.ylim(0, 1)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison (Random Forest vs Neural Network)")
plt.show()

#  Visualization: Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

plot_confusion_matrix(y_test, y_pred_rf, "Confusion Matrix - Random Forest")
plot_confusion_matrix(y_test, nn_model.predict(X_test).round(), "Confusion Matrix - Neural Network")
