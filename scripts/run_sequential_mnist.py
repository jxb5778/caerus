
from mlxtend.data import loadlocal_mnist
import numpy as np

import caerus


DIRECTORY = 'C:/Users/bergj/Documents/Geroge Mason/Courses/2020- Fall/CS 688/Project_2/mnist/extract/'

X_train, y_train = loadlocal_mnist(
    images_path='{}{}'.format(DIRECTORY, 'train-images.idx3-ubyte'),
    labels_path='{}{}'.format(DIRECTORY, 'train-labels.idx1-ubyte')
)


X_train = (1/255.0) * X_train
y_train = np.array([np.bincount(np.array([y]), minlength=10) for y in y_train])

X_test, y_test = loadlocal_mnist(
    images_path='{}{}'.format(DIRECTORY, 't10k-images.idx3-ubyte'),
    labels_path='{}{}'.format(DIRECTORY, 't10k-labels.idx1-ubyte')
)

X_test = (1/255.0) * X_test
y_test = np.array([np.bincount(np.array([y]), minlength=10) for y in y_test])

print("X train shape: ", X_train.shape)
print("Y train shape: ", y_train.shape)
print("X test shape: ", X_test.shape)
print()

clf = caerus.models.Sequential([
    caerus.layers.Input(input_shape=(1, 784)),
    caerus.layers.Dense(units=300, activation='sigmoid'),
    caerus.layers.Dense(units=100, activation='sigmoid'),
    caerus.layers.Dense(units=10, activation='softmax')
])

sgd = caerus.optimizers.SGD(learning_rate=0.2, beta_1=0.9, grad_clip=100)

clf.compile(loss='crossentropy', optimizer=sgd)

print(clf.summary())

history = clf.fit(X=X_train, y=y_train, epochs=2, batch_size=1)

preds = clf.predict(X_test)

print(preds)
print(np.sum(preds, axis=1, keepdims=True))
print(preds.shape)

error = caerus.errors.CrossEntropy()

print("Test error rate: ", error(yhat=preds, y=y_test))
