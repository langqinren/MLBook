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
        self.__eta    = eta
        self.__n_iter = n_iter


    def __fit_binary(self, X, Y):
        w = np.zeros(1 + X.shape[1])

        for i in range(self.__n_iter):
            error = 0

            for x, y in zip(X, Y):
                y_     = self.__predict_binary(x, w)
                delta  = self.__eta * (y - y_)
                w[0]  += delta
                w[1:] += delta * x
                error += int(delta != 0.0)

        return w


    """
        fitting multiple class using One-vs-Rest strategy

        (1) if there are [1,...,i,...,n] n classes in the data, build n classifiers

        (2) for each class i, build a classifier based on i vs. -i
    """
    def __fit_multiple(self, X, Y):
        w = []

        for pos_label in np.unique(Y):
            Y_ = np.copy(Y)

            pos_idx = (Y == pos_label)
            neg_idx = (Y != pos_label)

            Y_[pos_idx] = +1
            Y_[neg_idx] = -1
            w.append(self.__fit_binary(X, Y_))

        return w


    def __predict_binary(self, X, w):
        return np.where(np.dot(X, w[1:]) + w[0] >= 0.0, 1, -1)


    """
        make decision by applying all classifier to an unseen example and predict the label for which the corresponding
        classifier reports highest confidence. In perceptron learning algorithm, the one-vs-rest strategy indicates if
        one classifier generates the high score among others, it has the highest confidence to assign to this classifier.
    """
    def __predict_multiple(self, X, w):
        dist = []

        for i in range(len(w)):
            dist.append(np.dot(X, w[i,1:]) + w[i,0])

        return np.argmax(dist, axis=0)