__author__ = 'Shawn'

import numpy as np

class Classifier(object):
    """
    """
    def fit(self, X, Y):
        class_num = len(np.unique(Y))

        if class_num == 2:
            self.__w = np.copy(self._fit_binary(X, Y))
        elif class_num > 2:
            self.__w = np.copy(self._fit_multiple(X, Y))

        return self

    """
    """
    def predict(self, X):
        shape = self.__w.shape

        if shape[0] == 1:
            return self._predict_binary(X, self.__w)
        elif shape[0] > 1:
            return self._predict_multiple(X, self.__w)

    """
    """
    def _fit_multiple(self, X, Y):
        w = []

        for pos_label in np.unique(Y):
            Y_ = np.copy(Y)

            pos_idx = (Y == pos_label)
            neg_idx = (Y != pos_label)

            Y_[pos_idx] = +1
            Y_[neg_idx] = -1
            w.append(self._fit_binary(X, Y_))

        return w

    """
    """
    def _predict_multiple(self, X, w):
        dist = []

        for i in range(len(w)):
            dist.append(np.dot(X, w[i,1:]) + w[i,0])

        return np.argmax(dist, axis=0)


    def _fit_binary(self, X, Y):
        raise NotImplementedError

    def _predict_binary(self, X, w):
        raise NotImplementedError
