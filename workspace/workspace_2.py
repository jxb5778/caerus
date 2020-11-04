
import numpy as np


arr1 = np.array([1, 2, 3])
arr2 = np.array([[1,1], [2, 2], [3, 3]])

shuffler = np.random.permutation(arr2.shape[0])

arr1_shuffled = arr1[shuffler]
arr2_shuffled = arr2[shuffler]

print("Array 1: ", arr1_shuffled)
print("Array 2: ", arr2_shuffled)
