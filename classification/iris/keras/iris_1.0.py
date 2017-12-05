import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

seed = 7  # Fix random seed for reproducibility.
numpy.random.seed(seed)
EPOCHS = 200
BATCH_SIZE = 5
VERBOSE = 0
FILENAME = '../data/iris.csv'
LABELS = {0: 'Iris-setosa', 1: 'Iris-versicolor,', 2: 'Iris-virginica '}


def get_training_data():
    """Get training and test data.

    Original file: 5.1,3.5,1.4,0.2,Iris-setosa
    We convert the label data to a categorical value.
    First using  encoder.transfor we convert Iris-setosa,Iris-versicolor,Iris-virginica to [0,1,2].
    Then with np_utils.to_categorical Converts [0,1,2] to [ 1.  0.  0.], [ 0.  1.  0.],  [ 0.  0.  1.].

    Returns:
        A tuple of numpy.ndarray. Training data and training label.
    """

    # Load dataset.
    dataframe = pandas.read_csv(FILENAME, header=None)
    dataset = dataframe.values
    training_data = dataset[:, 0:4].astype(float)  # 5.1,3.5,1.4,0.2,
    training_label = dataset[:, 4]  # Iris-setosa

    # Encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(training_label)
    encoded_label = encoder.transform(training_label)

    # Convert integers to one hot encoded.
    training_categorical_label = np_utils.to_categorical(encoded_label)
    print type(training_data), type(training_categorical_label)
    return training_data, training_categorical_label


def baseline_model():
    """Create a baseline model.

    4 inputs -> [4 hidden nodes] -> 3 outputs.

    Returns:
        A Keras Sequential model (model).
    """

    model = Sequential()
    model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
    model.add(Dense(3, init='normal', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Get training data and labels.
training_data, training_label = get_training_data()
estimator = KerasClassifier(build_fn=baseline_model, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)
# k-fold cross validation.
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# Evaluate our model (estimator) on our dataset.
results = cross_val_score(estimator, training_data, training_label, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
# Fit the classifier.
estimator.fit(training_data, training_label)
# Classify two new flower samples.
samples = numpy.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
predictions = list(estimator.predict(samples))
print("Predictions: {}".format(str(predictions)))
