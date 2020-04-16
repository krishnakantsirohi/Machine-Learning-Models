# Krishnakant Singh Sirohi
# 1001668969

import numpy as np
import sys

class knn_classify:
    def __init__(self, k):
        self.classes = None
        self.k = k
        self.X_train = None
        self.y_train = None
        self.n_classes = None
        self.list = [None] * self.k
        self.tg = {}

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.n_classes = len(np.unique(y))
        self.classes = np.unique(y)

    def predict_proba(self, X):
        return np.asarray([self._predict(x) for x in X])

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def accuracy_score(self, y_test, y_pred):
        return np.mean(np.where(y_test == y_pred, 1, 0))

    def _predict(self, X):
        distances = sorted([[self._distance(X, self.X_train[i]), i] for i in range(self.X_train.shape[0])])
        probs = np.zeros(self.n_classes)
        for i in range(0, self.k):
            index = self.y_train[distances[i][1]]
            probs[index] += 1
        return probs

    def _distance(self, x, y):
        return np.sqrt(np.sum(np.square(x - y)))

if __name__ == '__main__':
    train = np.loadtxt(sys.argv[1])
    test = np.loadtxt(sys.argv[2])
    k = int(sys.argv[3])

    np.random.seed(10)

    X_train = train[:, :-1]
    y_train = train[:, -1].astype(np.int32)

    X_test = test[:, :-1]
    y_test = test[:, -1].astype(np.int32)

    X_train = (X_train - np.mean(X_train, axis=0))/np.std(X_train, axis=0)
    X_test = (X_test - np.mean(X_test, axis=0))/np.std(X_test, axis=0)

    knn = knn_classify(k)
    unique = np.unique(y_train)
    for index, val in zip(range(0, len(unique)), unique):
        knn.tg[index] = val
        y_train[y_train == val] = index
        y_test[y_test == val] = index

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    correct = 0
    for actual, target, index in zip(y_pred, y_test, range(1, X_test.shape[0] + 1)):
        correct += np.where(actual == target, 1, 0)
        print('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2lf' % (
        index, knn.tg[actual], knn.tg[target], correct / index))
    print('classification accuracy=%6.4f' % knn.accuracy_score(y_test, y_pred))