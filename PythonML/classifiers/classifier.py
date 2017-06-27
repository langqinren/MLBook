__author__ = 'Shawn'

import numpy as np

from abc import ABC, abstractmethod


class Classifier(ABC):
    @classmethod
    def fit(self, X, Y):
        class_num = len(np.unique(Y))

        if class_num == 2:
            self.__w = np.copy(self.__fit_binary(X, Y))
        elif class_num > 2:
            self.__w = np.copy(self.__fit_multiple(X, Y))

        return self

    @classmethod
    def predict(self, X):
        shape = self.__w.shape

        if shape[0] == 1:
            return self.__predict_binary(X, self.__w)
        elif shape[0] > 1:
            return self.__predict_multiple(X, self.__w)

    @abstractmethod
    def __fit_multiple(self, X, Y):
        pass

    @abstractmethod
    def __fit_binary(self, X, Y):
        pass

    @abstractmethod
    def __predict_multiple(self, Y, w):
        pass

    @abstractmethod
    def __predict_binary(self, Y, w):
        pass