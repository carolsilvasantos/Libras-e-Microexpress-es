import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def train_model(data_path="data/alphabet_dataset.csv", model_path="models/alphabet_classifier.pkl"):
    """Trains a Random Forest classifier on the collected alphabet dataset."""
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please collect data first.")
        return

    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    if len(df) == 0:
        print("Dataset is empty. Please collect data first.")
        return

    # Split features and labels
    X = df.drop("label", axis=1)
    y = df["label"]

    print(f"Dataset loaded. Total samples: {len(df)}")
    print(f"Classes: {y.unique()}")

    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 and min(y.value_counts()) > 1 else None)

    print("Training Random Forest Classifier...")
    # Initialize and train the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    if len(X_test) > 0:
        print("Evaluating model...")
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        
        # Only print classification report if there are multiple classes
        if len(y.unique()) > 1:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

    # Save the model and labels list
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model_data = {
        "model": clf,
        "labels": y.unique().tolist()
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"Model saved successfully to {model_path}.")

if __name__ == "__main__":
    train_model()
