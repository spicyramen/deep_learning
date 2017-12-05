"""Analyze Playfull reviews dataset for number of reviews prediction.

This is a dataset that describes how many weekly reviews exists for
all Play Vertical.
We will predict reviews for next week.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import numpy
import pandas
import matplotlib.pyplot as plt

# Dataset parameters.
FILENAME = '../data/reviews.csv'
_TRAINING_SIZE = 0.67
_LOOP_BACK_WINDOW = 1

# Neural Network parameters.
_HPARAMS = {'epochs': 100, 'batch_size': 1}
_VERBOSE_LEVEL = 2
_SEED = 7


def _IsEmpty(dataframe):
    """Check if Pandas dataFrame is empty or not.

    Args:
        dataframe: (pandas.DataFrame) Dataframe to process.

    Returns:
        A boolean. True if the given data is interpreted as empty
    """

    if dataframe is not None and not dataframe.empty:
        return False
    return True


def ExtractDataSet(dataset, look_back_window=1):
    """Convert an array of values into a dataset matrix.

    Args:
      dataset: (numpy.array)
      look_back_window: (int)

    Returns:
      numpy.array
    """

    features, target = [], []
    for idx in range(len(dataset) - look_back_window - 1):
        time_slot = dataset[idx:(idx + look_back_window), 0]
        features.append(time_slot)
        target.append(dataset[idx + look_back_window, 0])
    # reshape input to be [samples, time steps, features]
    return numpy.array(features), numpy.array(target)


def SplitDataSet(dataset):
    """Get training and test dataset.

    Args:
      dataframe: (Pandas.Dataframe)

    Returns:
      (tuple of numpy.array)

    Raises:
      ValueError: Empty dataset.
    """

    # Split input into training and test datasets
    train_size = int(len(dataset) * _TRAINING_SIZE)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print(len(train), len(test))
    return train, test


def LoadDataFromFile(filename):
    """Reads filename and generates a Pandas DataFrame.

    Args:
      filename(str): filename to read data from CNS.

    Returns:
      dataframe A Pandas.Dataframe.

    Raises:
      FileError: Not able to read file.
      ValueError: Invalid file, No data in file.
    """

    if not filename:
        raise ValueError('Invalid filename')

    # Fix random seed for reproducibility.
    numpy.random.seed(_SEED)
    # Generate a numpy array from data in file.
    dataframe = pandas.read_csv(filename, usecols=[1], engine='python', skipfooter=3)
    if _IsEmpty(dataframe):
        raise ValueError('No data in file')

    return dataframe


def CreateModel(loop_back_window):
    """Create Keras Model.

    Args:
      loop_back_window: (int) Loop back window.
    Returns:
      A Keras Model.
    """

    # Create model.
    model = Sequential()
    model.add(LSTM(4, input_shape=(None, loop_back_window)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def main():
    loop_back_window = _LOOP_BACK_WINDOW
    verbose_level = _VERBOSE_LEVEL

    dataframe = LoadDataFromFile(FILENAME)

    # Generate a numpy array from data in file.
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # Get training and test datasets.
    training_dataset, test_dataset = SplitDataSet(dataset)
    # Extract Features and Target.
    training_features, training_target = ExtractDataSet(training_dataset, loop_back_window)
    test_features, test_target = ExtractDataSet(test_dataset, loop_back_window)

    # Reshape input to be [samples, time steps, features]
    training_features = numpy.reshape(training_features, (training_features.shape[0], 1, training_features.shape[1]))
    test_features = numpy.reshape(test_features, (test_features.shape[0], 1, test_features.shape[1]))

    # Create Model.
    model = CreateModel(loop_back_window)
    model.fit(
        training_features,
        training_target,
        epochs=_HPARAMS['epochs'],
        batch_size=_HPARAMS['batch_size'],
        verbose=verbose_level)

    # Generate predictions for training and test data.
    train_predictions = model.predict(training_features)
    test_predictions = model.predict(test_features)

    # Invert predictions. (Scale back the data to the original representation)
    train_predictions = scaler.inverse_transform(train_predictions)
    training_target = scaler.inverse_transform([training_target])
    test_predictions = scaler.inverse_transform(test_predictions)
    test_target = scaler.inverse_transform([test_target])

    # Scores.
    training_score = math.sqrt(mean_squared_error(training_target[0], train_predictions[:, 0]))
    print('Train Score: %.2f RMSE' % (training_score))
    test_score = math.sqrt(mean_squared_error(test_target[0], test_predictions[:, 0]))
    print('Test Score: %.2f RMSE' % (test_score))

    # Shift training predictions for plotting.
    training_predictions_plot = numpy.empty_like(dataset)
    training_predictions_plot[:, :] = numpy.nan
    training_predictions_plot[loop_back_window:len(train_predictions) + loop_back_window, :] = train_predictions

    # Shift test predictions for plotting.
    test_predictions_plot = numpy.empty_like(dataset)
    test_predictions_plot[:, :] = numpy.nan
    test_predictions_plot[len(train_predictions) + (loop_back_window * 2) + 1:len(dataset) - 1, :] = test_predictions

    # Scale back the data to the original representation
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(training_predictions_plot)
    plt.plot(test_predictions_plot)
    # Plot baseline and predictions. Blue=Whole Dataset, Green=Training, Red=Predictions.
    plt.show()


main()

# Training Score: 3800420.20 MSE (1949.47 RMSE)
# Test Score: 2577974.75 MSE (1605.61 RMSE)
