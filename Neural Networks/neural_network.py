# Krishnakant Singh Sirohi
# 1001668969

import numpy as np
import sys

class neural_network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.biases = [np.ones((r, 1)) for r in sizes[1:]]
        self.weights = [np.random.uniform(low=-.05, high=.05, size=(c, r)) for r, c in zip(sizes[:-1], sizes[1:])]
        self.dimensions = sizes[0]
        self.accuracy = []
        self.tg = {}

    def sigmoid(self, X):
        return 1. / (1. + np.exp(-X))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def predict(self, X):
        for b, w in zip(self.biases, self.weights):
            X = self.sigmoid(np.dot(w, X) + b)
        return X

    def train(self, X_train, y_train, epochs=1, lr=1.):
        for epoch in range(epochs):
            for sample, target in zip(X_train, y_train):
                nw, nb = self.backprop(sample.reshape(sample.shape[0], 1), target.reshape(target.shape[0], 1))
                for w, b, nw1, nb1 in zip(self.weights, self.biases, nw, nb):
                    w -= lr * nw1
                    b -= lr * nb1
            lr *= 0.98

    def backprop(self, x, y):
        ddelta_db = [np.zeros(b.shape) for b in self.biases]
        ddelta_dw = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]
        outputs = []
        for b, w in zip(self.biases, self.weights):
            output = np.dot(w, activation) + b
            outputs.append(output)
            activation = self.sigmoid(output)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * activations[-1] * (1 - activations[-1])
        ddelta_db[-1] = delta
        ddelta_dw[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            input = outputs[-l]
            output = self.sigmoid_prime(input)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * output
            ddelta_db[-l] = delta
            ddelta_dw[-l] = np.dot(delta, activations[-l - 1].transpose())
        return ddelta_dw, ddelta_db

    def test(self, X, y):
        temp = self.predict(X.T).T
        correct = 0.
        for sample, target, index in zip(temp, y, range(1, len(y)+1)):
            predicted = np.where(sample == np.max(sample))
            if target in predicted:
                correct = correct + (1. / len(predicted))
            self.accuracy.append(correct / index)
            print('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f' % (index, self.tg[np.max(predicted[0])], self.tg[target], self.accuracy[-1]))

if __name__ == '__main__':
    train = np.loadtxt(sys.argv[1])
    test = np.loadtxt(sys.argv[2])
    layers = int(sys.argv[3])
    num_nodes = int(sys.argv[4])
    epochs = int(sys.argv[5])

    np.random.seed(22)

    input_dimensions = train.shape[1] - 1

    X_train = (np.array(train[:, :-1]) / np.max(train[:, :-1]))
    X_test = (np.array(test[:, :-1]) / np.max(test[:, :-1]))

    y_train = train[:, -1].astype(np.int32)
    y_test = test[:, -1].astype(np.int32)

    sizes = [input_dimensions]
    for layer in range(layers-2):
        sizes.append(num_nodes)
    sizes.append(len(np.unique(y_train)))

    nn = neural_network(sizes)

    unique = np.unique(y_train)
    for index, val in zip(range(0, len(unique)), unique):
        nn.tg[index] = val
        y_train[y_train == val] = index
        y_test[y_test == val] = index

    y_train = np.eye(np.max(y_train) + 1)[y_train]

    nn.train(X_train=X_train, y_train=y_train, epochs=epochs)
    nn.test(X_test, y_test)
    print('classification accuracy=%6.4f' % np.average(nn.accuracy))