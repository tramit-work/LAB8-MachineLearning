from flask import Flask, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.datasets import load_wine

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/baitapnhenhang')
def baitapnhenhang():
    TN = 50
    FP = 10
    FN = 5
    TP = 30

    accuracy = round((TP + TN) / (TP + TN + FP + FN), 2)
    recall = round(TP / (TP + FN), 2)
    specificity = round(TN / (TN + FP), 2)
    precision = round(TP / (TP + FP), 2)
    f1 = round((2 * precision * recall) / (precision + recall), 2)

    return render_template(
        'baitapnhenhang.html', accuracy=accuracy, recall=recall, specificity=specificity, precision=precision, f1=f1
    )

@app.route('/baitapnangcao')
def baitapnangcao():
    TN = 50
    FP = 10
    FN = 5
    TP = 30

    accuracy = round((TP + TN) / (TP + TN + FP + FN), 2)
    recall = round(TP / (TP + FN), 2)
    specificity = round(TN / (TN + FP), 2)
    precision = round(TP / (TP + FP), 2)
    f1 = round((2 * precision * recall) / (precision + recall), 2)
    balanced_accuracy = round((recall + specificity) / 2, 2)
    mcc = round((TP * TN - FP * FN) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5), 2)
    fmi = round((precision * recall) ** 0.5, 2)
    bias = round((FP - FN) / (TP + TN + FP + FN), 2)

    return render_template(
        'baitapnangcao.html', accuracy=accuracy, recall=recall, specificity=specificity, precision=precision, f1=f1,
        balanced_accuracy=balanced_accuracy, mcc=mcc, fmi=fmi, bias=bias
    )

@app.route('/baitapvandung')
def baitapvandung():

    np.random.seed(42)
    data_size = 1000

    X_class0 = np.random.multivariate_normal([2, 2], [[1.5, 0.75], [0.75, 1.5]], data_size // 2)
    X_class1 = np.random.multivariate_normal([4, 4], [[1.5, 0.75], [0.75, 1.5]], data_size // 2)
    X = np.vstack((X_class0, X_class1))
    y = np.hstack((np.zeros(data_size // 2), np.ones(data_size // 2)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def knn_predict(X_train, y_train, X_test, k=5):
        y_pred = []
        for test_point in X_test:
            distances = [euclidean_distance(test_point, x) for x in X_train]
            k_indices = np.argsort(distances)[:k]
            k_nearest_labels = [y_train[i] for i in k_indices]
            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
            y_pred.append(most_common)
        return np.array(y_pred)

    def confusion_matrix(y_true, y_pred):
        TP = ((y_true == 1) & (y_pred == 1)).sum()
        TN = ((y_true == 0) & (y_pred == 0)).sum()
        FP = ((y_true == 0) & (y_pred == 1)).sum()
        FN = ((y_true == 1) & (y_pred == 0)).sum()
        return np.array([[TN, FP], [FN, TP]])

    def evaluate_model(y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return accuracy, recall, specificity, precision, f1

    y_pred_knn = knn_predict(X_train, y_train, X_test, k=5)
    accuracy, recall, specificity, precision, f1 = evaluate_model(y_test, y_pred_knn)
    accuracy = round(accuracy, 2)
    recall = round(recall, 2)
    specificity = round(specificity, 2)
    precision = round(precision, 2)
    f1 = round(f1, 2)

    return render_template(
        'baitapvandung.html', accuracy=accuracy, recall=recall, specificity=specificity, precision=precision, f1=f1
    )

@app.route('/baitapvenha')
def baitapvenha():

    data = load_wine()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    accuracy = round(accuracy, 2)
    recall = round(recall, 2)
    precision = round(precision, 2)

    return render_template(
        'baitapvenha.html', accuracy=accuracy, recall=recall, precision=precision
    )

if __name__ == '__main__':
    app.run(debug=True)
