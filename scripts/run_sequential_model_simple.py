
import pandas as pd

import caerus


DIRECTORY = 'C:/Users/bergj/Documents/Geroge Mason/Courses/2020- Fall/CS 688/Assignment 2/twoclassData/twoclassData/'

train_file = '{}{}'.format(DIRECTORY, 'set9.train')
test_file = '{}{}'.format(DIRECTORY, 'set.test')

train_df = pd.read_csv(train_file, header=None, sep=' ')
test_df = pd.read_csv(test_file, header=None, sep=' ')

cols_a = ['x', 'y', 'label', 'blank']
cols_b = ['x', 'y', 'label']

train_df.columns = cols_b
test_df.columns = cols_a

train_df = train_df[cols_b]
test_df = test_df[cols_b]

train_df['label'] = train_df['label'].map(int)
test_df['label'] = test_df['label'].map(int)

clf = caerus.models.Sequential([
    caerus.layers.Input(input_shape=(1, 2)),
    caerus.layers.Dense(units=2, activation='sigmoid'),
    caerus.layers.Dense(units=1, activation='sigmoid')
])

sgd = caerus.optimizers.SGD(learning_rate=0.3, beta_1=0.9, grad_clip=100)

clf.compile(loss='mse', optimizer=sgd)

print(clf.summary())

history = clf.fit(X=train_df[['x', 'y']], y=train_df['label'], epochs=2750, batch_size=10)

preds = clf.predict(test_df[['x', 'y']])[0]

preds = [1 if pred > 0.5 else 0 for pred in preds]

mask = [y == yhat for y, yhat in zip(preds, test_df['label'].tolist())]

accuracy = sum(mask) / len(preds)

print("Accuracy: ", accuracy)
print(preds)
