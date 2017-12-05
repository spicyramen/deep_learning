FILENAME = 'pima-indians-diabetes.csv'
_LEARNING_RATE = 0.003
_NUM_FEATURES = 8
_NUM_LABELS = 1
_NUM_EPOCHS = 150
_BATCH_SIZE = 10

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

FILENAME = '../data/pima-indians-diabetes.csv'


def import_data(filename, batch_size):
    if not filename:
        raise ValueError('Invalid filename')
    dataset = np.loadtxt(filename, delimiter=",")
    training_data = dataset[:, 0:6].astype(float)
    training_label = dataset[:, 7]

    return train_test_split(training_data, training_label, test_size=0.2, random_state=42)



