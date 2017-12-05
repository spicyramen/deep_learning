# Multilayer Perceptron to Predict International Airline Passengers (t+1, given t)
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense


def CreateDataSet(dataset, look_back=1):
    """Convert an array of values into a dataset matrix.

    dataset (numpy): dataset either training or testing
    look_back (int): look back window. Example: look_back = 1 => Day 0 - Day 1
    """

    data_features, data_target = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        data_features.append(a)
        data_target.append(dataset[i + look_back, 0])
    return numpy.array(data_features), numpy.array(data_target)

# Fix random seed for reproducibility.
numpy.random.seed(7)

# Load the dataset.
dataframe = pandas.read_csv('data/reviews.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

# Split into train and test sets.
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print(len(train), len(test))

look_back = 10
trainX, trainY = CreateDataSet(train, look_back)
testX, testY = CreateDataSet(test, look_back)

# Create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)

# Estimate model performance.
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: % .2f MSE(% .2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: % .2f MSE (% .2f RMSE)' % (testScore, math.sqrt(testScore)))

# Generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
# plot baseline and predictions
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# Train Score:  4162216.70 MSE( 2040.15 RMSE)
# Test Score:  4159027.75 MSE ( 2039.37 RMSE)