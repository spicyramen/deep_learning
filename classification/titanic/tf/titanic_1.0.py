import random
import pandas
import numpy as np
# from sklearn import metrics, cross_validation
from sklearn import metrics
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn

random.seed(42)

data = pandas.read_csv('data/titanic.csv')

"""
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
"""

X = data[['Embarked']]  # Embarked is the port (3 classes).
y = data['Survived']  # 0 or 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

embarked_classes = X_train['Embarked'].unique()
n_classes = len(embarked_classes) + 1
print('Number of classes: ', n_classes)
print('Embarked has next classes: ', embarked_classes)

cat_processor = learn.preprocessing.CategoricalProcessor()
X_train = np.array(list(cat_processor.fit_transform(X_train)))
X_test = np.array(list(cat_processor.transform(X_test)))

### Embeddings
EMBEDDING_SIZE = 3


def categorical_model(features, target):
    target = tf.one_hot(target, 2, 1.0, 0.0)
    features = learn.ops.categorical_variable(
        features, n_classes, embedding_size=EMBEDDING_SIZE, name='embarked')
    prediction, loss = learn.models.logistic_regression(tf.squeeze(features, [1]), target)
    train_op = layers.optimize_loss(loss,
                                    tf.contrib.framework.get_global_step(), optimizer='SGD', learning_rate=0.05)
    return tf.argmax(prediction, dimension=1), loss, train_op


classifier = learn.Estimator(model_fn=categorical_model)
classifier.fit(X_train, y_train, steps=1000)

print("Accuracy: {0}".format(metrics.accuracy_score(classifier.predict(X_test), y_test)))
print("ROC: {0}".format(metrics.roc_auc_score(classifier.predict(X_test), y_test)))


### One Hot

def one_hot_categorical_model(features, target):
    features = tf.one_hot(features, n_classes, 1.0, 0.0)
    target = tf.one_hot(target, 2, 1.0, 0.0)
    prediction, loss = learn.models.logistic_regression(
        tf.squeeze(features, [1]), target)
    train_op = layers.optimize_loss(loss,
                                    tf.contrib.framework.get_global_step(), optimizer='SGD',
                                    learning_rate=0.01)
    return tf.argmax(prediction, dimension=1), loss, train_op


classifier = tf.estimator.Estimator(model_fn=one_hot_categorical_model)
classifier.fit(X_train, y_train, steps=1000)

print("Accuracy: {0}".format(metrics.accuracy_score(classifier.predict(X_test), y_test)))
print("ROC: {0}".format(metrics.roc_auc_score(classifier.predict(X_test), y_test)))
