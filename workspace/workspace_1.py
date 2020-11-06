
import numpy as np

import caerus


def cross_entropy(p, q):

    error = np.sum(-(p * np.log2(q)))/p.shape[0]

    return error


dist1 = np.array([[0.8, 0.1, 0.1], [0.9, 0.05, 0.05], [0.7, 0.1, 0.2], [0.7, 0.2, 0.1]])
dist2 = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])
dist3 = np.array([[.9, 0.1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])

softmax = caerus.activations.Softmax()
ce = caerus.errors.CrossEntropy()

print("Cross entropy:\n", ce(dist2, dist3))
print()
print("Softmax:\n", softmax(dist1.T))
print(np.argmax(softmax(dist1.T), axis=1))

dist1 = np.array([np.diagflat(dist.reshape(-1,1)) - np.dot(dist.reshape(-1,1), dist.reshape(-1,1).T) for dist in dist1])

dist1 = (1/dist1.shape[0]) * np.sum(dist1, axis=0)

print(dist1)
print(dist1.shape)
