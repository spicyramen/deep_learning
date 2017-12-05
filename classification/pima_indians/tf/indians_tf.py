from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split

import tensorflow as tf


def main(unused_argv):
    # Load dataset.
    diabetes = datasets.load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.2, random_state=42)

    # Build 3 layer DNN with 8, 12, 10 units respectively.
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
        x_train)
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        activation_fn=tf.nn.relu,
        hidden_units=[8, 12, 10],
        n_classes=2,
        optimizer='adam')

    # Fit and predict.
    classifier.fit(x_train, y_train, steps=200)
    predictions = list(classifier.predict(x_test, as_iterable=True))
    score = metrics.accuracy_score(y_test, predictions)
    print('Accuracy: {0:f}'.format(score))

if __name__ == '__main__':
    tf.app.run()
