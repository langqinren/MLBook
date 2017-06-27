__author__ = 'Shawn'

import numpy             as np
import matplotlib.pyplot as plt

from sklearn                 import datasets
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler


iris = datasets.load_iris()
X    = iris.data[:, [2,3]]
Y    = iris.target
#Y    = np.where(Y == 0, -1, 1)

sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

# split training & testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# normalization feature

# training
weights = []
Cs      = []
accurs  = []

for i in np.arange(-5, 5):
    c   = 10**float(i)
    ppn = LogisticRegression(C=c, random_state=2)
    ppn.fit(X_train, Y_train)

    Y_pred = ppn.predict(X_test)
    accur  = accuracy_score(Y_test, Y_pred)

    weights.append(ppn.coef_[1])
    Cs.append(c)
    accurs.append(accur)

f1 = plt.figure()
plt.plot(Cs, accurs)
plt.xscale('log')
plt.xlabel('parameter C')
plt.ylabel('testing accuracy')


weights = np.array(weights)
f2 = plt.figure()
plt.plot(Cs, weights[:,0], label='petal length')
plt.plot(Cs, weights[:,1], label='petal width', linestyle='--')
plt.legend(loc='upper left')
plt.xscale('log')
plt.xlabel('parameter C')
plt.ylabel('weight')


plt.show()
