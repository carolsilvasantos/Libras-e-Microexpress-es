import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import sys

def train_model(data_path="data/alphabet_dataset.csv", model_path="models/alphabet_classifier.pkl"):
    """Loads CSV and trains the Scikit-Learn model."""
    print("\n--- INICIANDO TREINAMENTO DA IA ---")
    
    if not os.path.exists(data_path):
        print(f"ERRO: Arquivo '{data_path}' nao encontrado.")
        print("DICA: Rode o 'main.py' e colete pelo menos algumas letras primeiro.")
        return

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"ERRO ao ler CSV: {e}")
        return

    if len(df) < 5:
        print(f"DADOS INSUFICIENTES: Voce tem apenas {len(df)} amostras.")
        print("DICA: Colete pelo menos 20-50 amostras para cada letra para ter resultado.")
        return

    print(f"Lendo {len(df)} amostras do banco de dados...")
    
    X = df.drop("label", axis=1)
    y = df["label"]
    
    unique_labels = y.unique()
    print(f"Letras detectadas no banco: {list(unique_labels)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=y if len(unique_labels) > 1 else None
    )

    print("Treinando Floresta Aleatoria (Random Forest)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"--- SUCESSO! ACURACIA: {acc*100:.1f}% ---")

    # Save
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "labels": list(unique_labels)}, f)
    
    print(f"Modelo salvo em: {model_path}")
    print("\nAgora voce pode rodar o 'main.py' e a IA reconhecera suas maos!\n")

if __name__ == "__main__":
    train_model()
