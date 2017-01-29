#!/usr/bin/env python

import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


if __name__ == '__main__':
    np.random.seed(42)

    train = pd.read_csv('Data/train.csv')
    labels = train.ix[:, 0].values.astype('int32')
    X_train = train.ix[:, 1:].values.astype('float32')
    X_test = pd.read_csv('Data/test.csv').values.astype('float32')
    y_train = np_utils.to_categorical(labels)

    X_train /= 255
    X_test /= 255

    model = Sequential()

    model.add(Dense(1500, input_dim=784, init="normal", activation="relu"))
    model.add(Dense(900, init="normal", activation="relu"))
    model.add(Dense(10, init="normal", activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="SGD",
                  metrics=["accuracy"])

    model.fit(X_train, y_train, batch_size=100, nb_epoch=150,
              validation_split=0.2, verbose=1)

    predicts = model.predict_classes(X_test, verbose=1)

    pd.DataFrame({"ImageId": range(1, len(predicts) + 1),
                  "Label": predicts}).to_csv('Data/result.csv', index=False,
                                             header=True)
