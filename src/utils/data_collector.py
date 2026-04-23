import os
import csv
import numpy as np

class DataCollector:
    """Collects and saves feature vectors to a CSV file for training."""

    def __init__(self, output_file="data/alphabet_dataset.csv", num_features=225):
        self.output_file = output_file
        self.num_features = num_features
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(self.output_file):
            self._write_headers()

    def _write_headers(self):
        """Writes the CSV headers: label, feature_0, feature_1, ..."""
        headers = ["label"] + [f"feature_{i}" for i in range(self.num_features)]
        with open(self.output_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def save_sample(self, label: str, features: np.ndarray):
        """Saves a single sample to the CSV file."""
        if features is None or len(features) != self.num_features:
            return False

        row = [label] + features.tolist()
        with open(self.output_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        return True
