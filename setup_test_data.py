import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

def create_synthetic_model():
    """Creates a basic model so the user can see the system working immediately."""
    print("Criando modelo inicial para testes...")
    
    # 225 features (pose + hands)
    num_features = 225
    labels = ['A', 'B', 'C', 'L', 'I', 'B', 'R', 'A', 'S']
    
    X = []
    y = []
    
    # Generate random but distinct patterns for a few letters
    for i, label in enumerate(['A', 'B', 'C']):
        for _ in range(20):
            # Base features + some noise unique to each letter index
            features = np.random.randn(num_features) * 0.1
            features += i * 0.5  # shift patterns so they are separable
            X.append(features)
            y.append(label)
            
    X = np.array(X)
    y = np.array(y)
    
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X, y)
    
    os.makedirs("models", exist_ok=True)
    model_path = "models/alphabet_classifier.pkl"
    
    model_data = {
        "model": clf,
        "labels": list(set(y))
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
        
    print(f"DONE: Modelo de teste criado em: {model_path}")
    print("Agora o sistema deve exibir 'ALFABETO: N/A' ou predicoes randomicas em vez de avisos de erro.")

if __name__ == "__main__":
    create_synthetic_model()
