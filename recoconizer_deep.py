from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, RMSprop

#import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# Setup
np.random.seed(1337)  # for reproducibility
batch_size = 128
nb_epoch = 20

# Read Data
print('Reading data...')
train = pd.read_csv('data/train.csv')
labels = train.ix[:, 0].values.astype('int32')
X_train = (train.ix[:, 1:].values).astype('float32')
X_test = (pd.read_csv('data/test.csv').values).astype('float32')

# pre-processing
y_train = np_utils.to_categorical(labels)
scale = np.max(X_train)
X_train /= scale
X_test /= scale
mean = np.std(X_train)
X_train -= mean
X_test -= mean
input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

# Cleaning data
labels = train.ix[:, 0].values.astype('int32')
X_train = (train.ix[:, 1:].values).astype('float32')
y_train = np_utils.to_categorical(labels)

# Model
model = Sequential()
model.add(Dense(784, 128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 10))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

print("Training...")
fitlog = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2)


print("Generating test predictions ...")
preds = model.predict_classes(X_test, verbose=1)
print(preds)


def save_to_csv(preds, fname):
    """
    Save the results into a csv file.
    """
    pd.DataFrame({"ImageId": list(range(1, len(preds) + 1)), "Label": preds}).to_csv(fname, index=False, header=True)

# Save results
print('Saving results to .csv')
save_to_csv(preds, "deep_nn.csv")
