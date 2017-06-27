__author__ = 'Shawn'

import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from sklearn import datasets
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score
from classifiers.perceptron  import Perceptron
#from sklearn.linear_model import Perceptron
#from sklearn.linear_model import LogisticRegression


# load training & testing data
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
ppn = Perceptron(C=1.0, random_state=2)
ppn.fit(X_train, Y_train)


# testing & evalulation
Y_pred = ppn.predict(X_test)
accur  = accuracy_score(Y_test, Y_pred)
print('Accuracy %.2f' % accur)


x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
res = 0.1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, res), np.arange(x2_min, x2_max, res))


colors = ('lightgreen', 'gray', 'cyan')
cmap   = cl.ListedColormap(colors[:len(np.unique(Y))])

bound = np.array(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = ppn.predict(bound)
Z = Z.reshape(xx1.shape)
plt.contourf(xx1,xx2, Z, aplha=0.4, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())


se_idx = (Y == 0)
ve_idx = (Y == 1)
vi_idx = (Y == 2)

plt.scatter(X[se_idx, 0], X[se_idx, 1], color='red',   marker='o', label='setosa')
plt.scatter(X[ve_idx, 0], X[ve_idx, 1], color='blue',  marker='x', label='versicolor')
plt.scatter(X[vi_idx, 0], X[vi_idx, 1], color='green', marker='^', label='virginica')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()
