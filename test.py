import numpy as np
from scipy.spatial.distance import cdist

x = np.array([[[1,2,3,4,5],
               [5,6,7,8,5],
               [5,6,7,8,5]],
              [[11,22,23,24,5],
               [25,26,27,28,5],
               [5,6,7,8,5]]])
i,j,k = x.shape

xx = x.reshape(i,j*k).T


y = np.array([[[31,32,33,34,5],
               [35,36,37,38,5],
               [5,6,7,8,5]],
              [[41,42,43,44,5],
               [45,46,47,48,5],
               [5,6,7,8,5]]])


yy = y.reshape(i,j*k).T


X = np.vstack([xx,yy])
# print(X.T)
V = [[ 3.11317942,  1.29638747], [ 1.29638747,  0.58241432]]
V = np.array(V)
VI = np.linalg.inv(V)
print(V, '\n', VI)
print(np.dot(V, VI))
print(np.dot(VI, V))
# print np.diag(np.sqrt(np.dot(np.dot((xx-yy),VI),(xx-yy).T)))