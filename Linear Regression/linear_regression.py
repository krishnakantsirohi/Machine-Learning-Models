

import sys
import numpy as np


class linear_regression:
    def __init__(self, num_of_features=1, degree=1, regularization=0):
        self.weights = np.random.random((num_of_features * degree + 1, 1))
        self.degree = degree
        self.regularization = regularization

    def predict(self, X):
        X_hat = np.insert(X, 0, 1, axis=1)
        for d in np.arange(2, self.degree + 1):
            X_hat = np.hstack((X_hat, X_hat[:, 1:] ** d))
        return X_hat @ self.weights

    def train(self, X, y):
        X_hat = np.insert(X, 0, 1, axis=1)
        for d in np.arange(2, self.degree + 1):
            X_hat = np.hstack((X_hat, X_hat[:, 1:] ** d))
        self.weights = np.matmul(np.matmul(np.linalg.pinv(
            np.add((self.regularization * np.identity(self.weights.shape[0])), np.matmul(np.transpose(X_hat), X_hat))),
            np.transpose(X_hat)), y)

    def test(self, X, y):
        pred = np.array(self.predict(X))
        e = np.array(np.square(y - pred))
        for i in range(0, len(pred)):
            print('ID=%5d, output=%14.4f, target value = %10.4f, squared error = %.4f' % (i + 1, pred[i], y[i], e[i]))

    def get_weights(self):
        for i in np.arange(0, len(self.weights)):
            print('w%s = %.5f' % (i, self.weights[i]))


if __name__ == '__main__':
    train = np.loadtxt(sys.argv[1])
    degree = int(sys.argv[2])
    regularization = int(sys.argv[3])
    test = np.loadtxt(sys.argv[4])

    X_train = np.array(train[:, :-1])
    y_train = np.array(train[:, -1]).reshape(train.shape[0], 1)

    X_test = np.array(test[:, :-1])
    y_test = np.array(test[:, -1]).reshape(test.shape[0], 1)

    lr = linear_regression(num_of_features=X_train.shape[1], degree=degree, regularization=regularization)
    lr.train(X=X_train, y=y_train)
    pred = lr.predict(X=X_test)
    lr.get_weights()
    lr.test(X=X_test, y=y_test)
