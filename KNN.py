import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def prepare_data(file_path, random_state=42):
    data = pd.read_csv(file_path)
    X = data.drop("class", axis=1)
    y = data["class"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=random_state
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manual_knn_predict(X_train, y_train, X_test, k=13):
    preds = []
    for test_point in X_test:
        dists = [euclidean(test_point, x) for x in X_train]
        k_idx = np.argsort(dists)[:k]
        k_labels = [y_train.iloc[i] for i in k_idx]
        preds.append(max(set(k_labels), key=k_labels.count))
    return np.array(preds)

def tune_k(X_train, y_train, X_valid, y_valid, k_range=(1, 30)):
    k_values = range(k_range[0], k_range[1] + 1)
    accuracies = []
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        acc = accuracy_score(y_valid, preds)
        accuracies.append(acc)
        print(f"K={k}: Accuracy={acc:.3f}")
    best_k = k_values[np.argmax(accuracies)]
    print(f"\nBest K={best_k} Accuracy={max(accuracies):.3f}")
    plt.plot(k_values, accuracies, marker='o')
    plt.title('Accuracy vs K')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    return best_k

def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_data("telescope_data/telescope_data.csv")
    best_k = tune_k(X_train, y_train, X_valid, y_valid, k_range=(1, 25))

    print("\nRunning Manual KNN...")
    manual_preds = manual_knn_predict(X_train, y_train, X_test, k=best_k)
    evaluate_model(f"Manual KNN (k={best_k})", y_test, manual_preds)

    print("\nRunning Scikit-Learn KNN...")
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    sk_preds = knn.predict(X_test)
    evaluate_model(f"Scikit-Learn KNN (k={best_k})", y_test, sk_preds)
