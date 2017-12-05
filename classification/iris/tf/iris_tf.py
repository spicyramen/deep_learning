#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Example of DNNClassifier for Iris plant dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split

import tensorflow as tf

SEED = 42
test_size = 0.2
LABELS = {0: 'Iris-setosa', 1: 'Iris-versicolor,', 2: 'Iris-virginica '}


def set_label(value):
    """Return correct value for LABELS."""
    if value in LABELS:
        return LABELS[value]
    return 'Invalid'


def get_training_data():
    """Loads and get training data.

    Returns:
        (tuple) with training, labels, test, test_labels
    """
    # Loads dataset from default datasets.
    iris = datasets.load_iris()
    return train_test_split(iris.data,
                            iris.target,
                            test_size=test_size,
                            random_state=SEED)


def main(argv):
    del argv  # Unused argv

    # Load dataset and get 'training' and 'test' datasets.
    training_data, test_data, train_labels, test_labels = get_training_data()

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
        training_data)
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=3)

    # Train model.
    classifier.fit(training_data, train_labels, steps=200)
    # Evaluate test data.
    accuracy_score = classifier.evaluate(x=test_data, y=test_labels)["accuracy"]
    print("Accuracy: {0:f}".format(accuracy_score))
    # Predict test data.
    predictions = list(classifier.predict(test_data, as_iterable=True))
    _predictions = [set_label(prediction) for prediction in predictions]
    print("Predictions: {}".format(_predictions))
    score = metrics.accuracy_score(test_labels, predictions)
    print("Accuracy Rate, which is calculated by accuracy_score() is: %f" % score)


if __name__ == '__main__':
    tf.app.run()
