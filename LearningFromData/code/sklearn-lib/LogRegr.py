__author__ = 'Shawn'

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition, datasets, metrics
from sklearn.model_selection import train_test_split

model = linear_model.Perceptron()

iris = datasets.load_iris()
X    = iris.data[:, :3]
Y    = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print Y_pred
print Y_test

#print metrics.precision_score(Y_test, Y_pred,  average='macro')
#print metrics.recall_score(Y_test, Y_pred,  average='macro')