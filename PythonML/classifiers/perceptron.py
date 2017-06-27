__author__ = 'Shawn'

import numpy as np

from classifiers.classifier import Classifier


class Perceptron(Classifier):

    """
    Parameters
    -------------
    eta : float
        Learning rate (between 0.0 and 1.0)

    n_iter : int
        Passes over the training set

    Attributes
    ------------
    w_ : weights

    errors_ : number of misclassification
    """
    def __init__(self, eta=0.01, n_iter=10):
        self._eta    = eta
        self._n_iter = n_iter

    """

    """
    def _fit_binary(self, X, Y):
        w = np.zeros(1 + X.shape[1])

        for i in range(self._n_iter):
            error = 0

            for x, y in zip(X, Y):
                y_     = self._predict_binary(x, w)
                delta  = self._eta * (y - y_)
                w[0]  += delta
                w[1:] += delta * x
                error += int(delta != 0.0)

        return w


    """

    """
    def _predict_binary(self, X, w):
        return np.where(np.dot(X, w[1:]) + w[0] >= 0.0, 1, -1)


    """
        fitting multiple class using One-vs-Rest strategy

        (1) if there are [1,...,i,...,n] n classes in the data, build n classifiers

        (2) for each class i, build a classifier based on i vs. -i
    """
    def _fit_multiple(self, X, Y):
        return super(Perceptron, self)._fit_multiple(X, Y)


    """
        make decision by applying all classifier to an unseen example and predict the label for which the corresponding
        classifier reports highest confidence. In perceptron learning algorithm, the one-vs-rest strategy indicates if
        one classifier generates the high score among others, it has the highest confidence to assign to this classifier.
    """
    def _predict_multiple(self, X, w):
        return super(Perceptron, self)._predict_multiple(X, w)