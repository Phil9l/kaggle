#!/usr/bin/env python

import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


def fix_data(data, get_result=False):
    data['Gender'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Embarked"] = data["Embarked"].fillna('S') \
                                       .map({'S': 0, 'C': 1, 'Q': 2}) \
                                       .astype(int)
    data = data.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'],
                     axis=1)
    if get_result:
        result = data['Survived']
        data = data.drop(['Survived'], axis=1)
        return data.values.astype('int32'), result

    return data.values.astype('int32')

if __name__ == '__main__':
    train_data, train_res = fix_data(pd.read_csv('Data/train.csv', header=0),
                                     get_result=True)
    test_csv = pd.read_csv('Data/test.csv', header=0)
    passenger_ids = test_csv['PassengerId']
    test_data = fix_data(test_csv)

    model = Sequential()
    model.add(Dense(1000, input_dim=7, init='uniform', activation='relu'))
    model.add(Dense(700, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer="SGD",
                  metrics=["accuracy"])

    model.fit(train_data, train_res, batch_size=200, nb_epoch=100,
              validation_split=0.2, verbose=1)

    predicts = [i[0] for i in model.predict_classes(test_data, verbose=1)]

    pd.DataFrame({"PassengerId": passenger_ids, "Survived": predicts}) \
        .to_csv('Data/result.csv', index=False, header=True)
