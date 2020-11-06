
import pandas as pd

import caerus


DIRECTORY = 'C:/Users/bergj/Documents/Geroge Mason/Courses/2020- Fall/CS 688/Assignment 2/twoclassData/twoclassData/'

train_file = '{}{}'.format(DIRECTORY, 'set10.train')
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

cls = caerus.models.MLP(layers=[2, 2, 1])

loss_func = caerus.errors.MeanSquaredError()
sgd = caerus.optimizers.SGD(learning_rate=0.3, beta_1=0.9, grad_clip=100)

cls.compile(loss=loss_func, optimizer=sgd)

print(cls.summary())

history = cls.fit(X=train_df[['x', 'y']], y=train_df['label'], epochs=2750, batch_size=5)

preds = cls.predict(test_df[['x', 'y']])[0]

preds = [1 if pred > 0.5 else 0 for pred in preds]

mask = [y == yhat for y, yhat in zip(preds, test_df['label'].tolist())]

accuracy = sum(mask) / len(preds)

print("Accuracy: ", accuracy)
print(preds)
