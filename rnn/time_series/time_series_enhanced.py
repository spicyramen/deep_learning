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

import numpy
import pandas
import matplotlib.pyplot as plt

# Dataset parameters.
FILENAME = 'data/reviews.csv'
_TRAINING_SIZE = 0.67
_LOOP_BACK_WINDOW = 10

# Neural Network parameters.
_HPARAMS = {'epochs': 200, 'batch_size': 2}
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
    return numpy.array(features), numpy.array(target)


def SplitDataSet(dataframe):
    """Get training and test dataset.

    Args:
      dataframe: (Pandas.Dataframe)

    Returns:
      (tuple of numpy.array)

    Raises:
      ValueError: Empty dataset.
    """

    if _IsEmpty(dataframe):
        raise ValueError('Empty dataframe')

    dataset = dataframe.values
    dataset = dataset.astype('float32')
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
    if not isinstance(dataframe, pandas.DataFrame):
        raise ValueError('No data in file')

    return dataframe


def CreateModel(dataset_features, dataset_target, loop_back_window,
                verbose_level):
    """Create Keras Model.

    Args:
      dataset_features: (numpy.array)
      dataset_target: (numpy.array)
      loop_back_window: (int) Loop back window.
      verbose_level: (int) Verbose level for Model fit method.

    Returns:
      A Keras Model.
    """

    # Create model.
    model = Sequential()
    model.add(Dense(8, input_dim=loop_back_window, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(
        dataset_features,
        dataset_target,
        epochs=_HPARAMS['epochs'],
        batch_size=_HPARAMS['batch_size'],
        verbose=verbose_level)
    return model


def main():
    loop_back_window = _LOOP_BACK_WINDOW
    verbose_level = _VERBOSE_LEVEL
    # Read CSV file and extract training and test datasets.
    # Fix random seed for reproducibility.
    numpy.random.seed(_SEED)
    # Generate a numpy array from data in file.
    dataframe = LoadDataFromFile(FILENAME)
    training_dataset, test_dataset = SplitDataSet(dataframe)
    # Extract Features and target.
    training_features, training_target = ExtractDataSet(training_dataset, loop_back_window)
    test_features, test_target = ExtractDataSet(test_dataset, loop_back_window)

    model = CreateModel(training_features, training_target, loop_back_window, verbose_level)

    # Estimate model performance.
    training_score = model.evaluate(training_features, training_target, verbose=0)
    print('Training Score: %.2f MSE (%.2f RMSE)' % (training_score,
                                                    math.sqrt(training_score)))
    test_score = model.evaluate(test_features, test_target, verbose=1)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (test_score,
                                                math.sqrt(test_score)))
    # Generate predictions for training and test data.
    train_predictions = model.predict(training_features)
    test_predictions = model.predict(test_features)

    # Shift train predictions for plotting.
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    training_predictions_plot = numpy.empty_like(dataset)
    training_predictions_plot[:, :] = numpy.nan
    training_predictions_plot[loop_back_window:len(train_predictions) + loop_back_window, :] = train_predictions
    # shift test predictions for plotting.
    test_predictions_plot = numpy.empty_like(dataset)
    test_predictions_plot[:, :] = numpy.nan
    test_predictions_plot[len(train_predictions) + (loop_back_window * 2) + 1:len(dataset) - 1, :] = test_predictions
    # plot baseline and predictions.
    plt.plot(dataset)
    plt.plot(training_predictions_plot)
    plt.plot(test_predictions_plot)
    plt.show()


main()

# Training Score: 3800420.20 MSE (1949.47 RMSE)
# Test Score: 2577974.75 MSE (1605.61 RMSE)
