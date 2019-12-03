import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

gamma = 0.1

def Dd(P, Q):
    s = 0
    if len(P) != len(Q):
        return 0
    
    for i in range(1, len(P) + 1):
        s += math.acos(math.sqrt(P[i]) * math.sqrt(Q[i]))
    return s

def D(P, Q):
    s = 0
    if len(P) != len(Q):
        return 0
    for i in range(0, len(P)):
        s += (P[i] - Q[i])*(P[i] - Q[i])

    return math.sqrt(s)

def K(P, Q):
    return math.exp(-1 * math.pow(D(P,Q),2) / gamma)

# for LOWBOW: each row of P contains the unweighted sum over BOLH
def my_kernel(A, B):
    matrix = [[0 for i in range(len(B))] for j in range(len(A)) ]
    for i in range(len(A)):
        for j in range(len(B)):
            matrix[i][j] = K(A[i], B[j])
    return matrix


h = .02  # step size in the mesh

# we create an instance of SVM and fit out data.
clf = svm.SVC(kernel=my_kernel)
clf.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
plt.title('3-Class classification using Support Vector Machine with custom'
          ' kernel')
plt.axis('tight')
plt.show()
