import numpy as np


def featureScale(x):
    return (x - x.mean()) / x.std()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self):
        self.theta = 0

    def costFunction(self, X, y, Lambda):
        m, n = np.shape(X)
        y = np.reshape(y, (m, 1))
        hyp = sigmoid(X.dot(self.theta))
        diff=hyp-y
        error = np.sum(y*np.log(hyp) + (1 - y)*np.log(1 - hyp))
        reg = 0
        tempTheta = 0
        if Lambda != 0:
            tempTheta = self.theta
            tempTheta[0] = 0
            reg = (Lambda / 2) * np.square(np.sum(tempTheta))
        J = (-1 / m) * (error + reg)
        grad = (1 / m) * (X.T.dot(diff)) + (Lambda / m) * tempTheta
        return J, grad

    def train(self, X, y, alpha=0.01, iters=10000, Lambda=0):
        m, n = np.shape(X)
        X = np.append(np.ones((m, 1)), X, axis=1)
        n = n + 1
        self.theta = np.zeros((n, 1))
        Jvec = np.zeros(iters)
        for e in range(iters):
            J, grad = self.costFunction(X, y, Lambda)
            self.theta = self.theta - alpha * grad
            Jvec[e] = J
        return Jvec, self.theta

    def predict(self, x):
        m = np.shape(x)[0]
        x = np.append(np.ones((m, 1)), x, axis=1)
        return sigmoid(x.dot(self.theta).reshape(m))
