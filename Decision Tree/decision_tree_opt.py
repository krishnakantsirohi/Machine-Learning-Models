# Krishnakant Singh Sirohi
# 1001668969

import numpy as np
import sys

class Node:
    def __init__(self, nodeid):
        self.info_gain = -1
        self.predicted_class = -1
        self.feature_index = -1
        self.threshold = -1
        self.left = None
        self.right = None
        self.id = nodeid

class decision_tree:
    def __init__(self, option, pruning_threshold, id=1, forest=False, numoftrees=1):
        self.option = option
        self.pruning_threshold = pruning_threshold
        self.id = id
        self.head = Node(1)
        self.forest = forest
        self.numoftrees = numoftrees
        self.randomforest = [decision_tree(option, pruning_threshold, id=i+1) for i in range(self.numoftrees)] if self.forest else None
        self.tg = {}

    def entropy(self, values):
        targets, counts = np.unique(values, return_counts=True)
        entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(targets))])
        return entropy

    def informationgain(self, X, y, attribute, threshold):
        total_entropy = self.entropy(y)
        indices = X[:, attribute] < threshold
        ld = y[indices]
        rd = y[~indices]
        ig_1, ig_2 = self.entropy(y[indices]), self.entropy(y[~indices])
        entropy = ((ig_1 * (ld.size/y.shape[0])) + (ig_2 * (rd.size / y.shape[0])))
        return total_entropy - entropy

    def choose_attribute(self, X, y, attributes):
        max_gain = -1
        best_attrib = -1
        best_thresh = -1

        if self.option == 'optimized':
            for attrib in attributes:
                for threshold in np.unique(X[:, attrib]):
                    gain = self.informationgain(X, y, attrib, threshold)
                    if gain > max_gain:
                        max_gain = gain
                        best_attrib = attrib
                        best_thresh = threshold

        elif self.option == 'randomized':
            attrib = np.random.choice(attributes)
            for threshold in np.unique(X[:, attrib]):
                gain = self.informationgain(X, y, attrib, threshold)
                if gain > max_gain:
                    max_gain = gain
                    best_attrib = attrib
                    best_thresh = threshold

        return max_gain, best_attrib, best_thresh

    def fit(self, X, y):
        self.n_classes = len(set(y))
        attributes = [i for i in range(X.shape[1])]

        if self.forest:
            for i in range(0, self.numoftrees):
                self.randomforest[i].head = self._fit(X, y, attributes)
            return self.randomforest
        else:
            self.head = self._fit(X, y, attributes)
            return self.head

    def _fit(self, X, y, attributes, nodeid=1):
        node = Node(nodeid=nodeid)
        node.predicted_class = np.argmax([np.sum(y == c) for c in range(self.n_classes)])
        if X.shape[0] <= self.pruning_threshold:
            node.feature_index = self.choose_attribute(X, y, attributes)[1]
            return node
        elif len(np.unique(y)) <= 1:
            node.feature_index = self.choose_attribute(X, y, attributes)[1]
            return node
        elif len(attributes) == 0:
            return node
        else:
            gain, idx, thr = self.choose_attribute(X, y, attributes)
            node.info_gain = gain
            node.threshold = thr
            node.feature_index = idx
            indices = X[:, idx] < thr
            X_left, y_left = X[indices], y[indices]
            X_right, y_right = X[~indices], y[~indices]
            node.left = self._fit(X_left, y_left, attributes, nodeid=nodeid*2)
            node.right = self._fit(X_right, y_right, attributes, nodeid=(nodeid*2)+1)
        return node

    def predict(self, X):
        if self.forest:
            probs = self.predict_proba(X) / self.n_classes
            return np.argmax(probs, axis=1)
        else:
            return [self._predict(x) for x in X]

    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], self.n_classes)).astype(np.int32)
        if self.forest:
            for tree in self.randomforest:
                predictions = [self._predict(x, tree) for x in X]
                probs += self.to_categorical(predictions, self.n_classes)
        else:
            predictions = [self._predict(x) for x in X]
            probs = self.to_categorical(predictions, self.n_classes)
        return probs

    def _predict(self, X, tree=None):
        if self.forest:
            self.head = tree.head
        node = self.head
        while node.left:
            if X[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def print(self):
        if self.forest:
            for tree in self.randomforest:
                self._print_tree(tree)
        else:
            self._print_tree(self)

    def _print_tree(self, tree):
        p = []
        q = []
        p.append(tree.head)
        while (p):
            node = p[0]
            p.remove(node)
            print('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f' % (tree.id, node.id, node.feature_index, node.threshold, node.info_gain))
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
            if not p:
                p = q
                q = []

    def to_categorical(self, y, num_classes=None, dtype='float32'):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=np.int32)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def accuracy(self, X, y):
        actuals = self.predict(X)
        correct = np.where(actuals == y, 1, 0)
        return np.average(correct)

if __name__ == '__main__':
    train = np.loadtxt(sys.argv[1])
    test = np.loadtxt(sys.argv[2])
    option = sys.argv[3]
    pruning_thr = float(sys.argv[4])

    np.random.seed(10)
    X_train = train[:, :-1]
    y_train = train[:, -1]

    X_test = test[:, :-1]
    y_test = test[:, -1]

    forest = option.startswith('forest')
    if forest:
        numTrees = int(option[6:])

    if forest:
        rf = decision_tree(option='randomized', pruning_threshold=pruning_thr, forest=forest, numoftrees=numTrees)
        unique = np.unique(y_train)
        for index, val in zip(range(0, len(unique)), unique):
            rf.tg[index] = val
            y_train[y_train == val] = index
            y_test[y_test == val] = index
        rfit = rf.fit(X_train, y_train)
        rf.print()
        predictions = rf.predict_proba(X_test)
        correct = 0
        for acutal, target, index in zip(predictions, y_test, range(y_test.shape[0])):
            predicted = np.where(acutal == np.max(acutal))
            if target in predicted[0]:
                correct += (1 / len(predicted[0]))
            print(
                'ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f' % (
                index + 1, rf.tg[np.max(predicted[0])], rf.tg[target], correct / (index + 1)))
        print('classification accuracy=%6.4f' % rf.accuracy(X_test, y_test))
    else:
        dt = decision_tree(option=option, pruning_threshold=pruning_thr)
        unique = np.unique(y_train)
        for index, val in zip(range(0, len(unique)), unique):
            dt.tg[index] = val
            y_train[y_train == val] = index
            y_test[y_test == val] = index

        tree = dt.fit(X_train, y_train)
        dt.print()
        predictions = dt.predict_proba(X_test)
        correct = 0
        for acutal, target, index in zip(predictions, y_test, range(y_test.shape[0])):
            predicted = np.where(acutal == np.max(acutal))
            if target in predicted[0]:
                correct += (1 / len(predicted[0]))
            print(
                'ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f' % (
                index + 1, dt.tg[np.max(predicted[0])], dt.tg[target], correct / (index + 1)))
        print('classification accuracy=%6.4f' % dt.accuracy(X_test, y_test))