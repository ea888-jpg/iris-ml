
import csv
import math
import random
from collections import Counter
from pathlib import Path

DATA_PATH = Path(__file__).with_name("iris.csv")
LABELS = ("setosa", "versicolor", "virginica")


def load_data(path=DATA_PATH):
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    X = [[float(row[name]) for name in (
        "sepal_length", "sepal_width", "petal_length", "petal_width"
    )] for row in rows]
    y = [row["species"] for row in rows]
    return X, y


def split_data(X, y, test_fraction=0.25, seed=0):
    indices = list(range(len(X)))
    random.Random(seed).shuffle(indices)
    test_count = math.ceil(len(indices) * test_fraction)
    test_ids, train_ids = indices[:test_count], indices[test_count:]
    return ([X[i] for i in train_ids], [X[i] for i in test_ids],
            [y[i] for i in train_ids], [y[i] for i in test_ids])


def euclidean_distance(a, b):
    return math.sqrt(sum((x - z) ** 2 for x, z in zip(a, b)))


class KNNClassifier:
    def __init__(self, k=1):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k має бути додатним цілим числом")
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        if not X or len(X) != len(y) or self.k > len(X):
            raise ValueError("Неправильні навчальні дані або k")
        self.X_train = [list(row) for row in X]
        self.y_train = list(y)
        return self

    def predict_one(self, sample):
        if not self.X_train or len(sample) != len(self.X_train[0]):
            raise ValueError("Модель не навчена або розмір ознак неправильний")
        distances = sorted(
            ((euclidean_distance(sample, row), label)
             for row, label in zip(self.X_train, self.y_train)),
            key=lambda item: item[0]
        )
        neighbors = [label for _, label in distances[:self.k]]
        counts = Counter(neighbors)
        return max(counts, key=lambda label: counts[label])

    def predict(self, samples):
        return [self.predict_one(sample) for sample in samples]

    def score(self, X, y):
        predictions = self.predict(X)
        return sum(a == b for a, b in zip(predictions, y)) / len(y)


def confusion_matrix(actual, predicted, labels=LABELS):
    result = [[0] * len(labels) for _ in labels]
    for true, guess in zip(actual, predicted):
        result[labels.index(true)][labels.index(guess)] += 1
    return result


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = KNNClassifier(k=1).fit(X_train, y_train)
    prediction = model.predict_one([5.0, 2.9, 1.0, 0.2])
    guesses = model.predict(X_test)
    correct = sum(a == b for a, b in zip(y_test, guesses))
    print(f"Дані: {len(X)} квіток, {len(X[0])} ознаки")
    print(f"Навчальна вибірка: {len(X_train)}; тестова: {len(X_test)}")
    print("Новий ірис [5.0, 2.9, 1.0, 0.2]:", prediction)
    print(f"Правильних прогнозів: {correct}/{len(y_test)}")
    print(f"Accuracy: {model.score(X_test, y_test):.4f}")
    print("Матриця помилок (рядки = факт, стовпці = прогноз):")
    for label, row in zip(LABELS, confusion_matrix(y_test, guesses)):
        print(f"{label:11s} {row}")


if __name__ == "__main__":
    main()
