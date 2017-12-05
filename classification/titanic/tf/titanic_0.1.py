import pandas as pd
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('data/titanic.csv')
random.seed(42)

_N_CLASSES = 2
_LEARNING_RATE = 0.05
_TRAIN_STEPS = 200

CONTINUOUS_COLUMNS = ['Age', 'SibSp', 'Fare']
LABEL_COLUMN = 'Survived'

# Data collection
y, X = data[LABEL_COLUMN], data[CONTINUOUS_COLUMNS].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature columns. Get Numerical values
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=_LEARNING_RATE)

classifier = tf.contrib.learn.LinearClassifier(n_classes=_N_CLASSES,
                                               feature_columns=feature_columns,
                                               optimizer=optimizer)


# TF model
def make_input_fn(features, target):
    def input_fn():
        feature_cols = {k: tf.constant(features[k].values) for k in CONTINUOUS_COLUMNS}
        feature_dict = dict(feature_cols)
        label = tf.constant(target.values)
        return feature_dict, label
    return input_fn

input_fn = make_input_fn(X_train, y_train)
classifier.fit(input_fn=make_input_fn(X_train, y_train), steps=_TRAIN_STEPS)

#print accuracy_score(y_test, classifier.predict(X_test, as_iterable=False))
#classifier.fit(X_train, y_train, batch_size=10, steps=100)


# Traditional SKlearn model
#lr = LogisticRegression()
#lr.fit(X_train, y_train)
# Accuracy score
#print accuracy_score(y_test, lr.predict(X_test))
