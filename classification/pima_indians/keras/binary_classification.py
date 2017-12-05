# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy

FILENAME = '../data/pima-indians-diabetes.csv'


# Fix random seed for reproducibility.
seed = 7
numpy.random.seed(seed)

# Load Pima indians dataset.
dataset = numpy.loadtxt(FILENAME, delimiter=',')
# Split into input (X) and output (Y) variables.
training_data = dataset[:, 0:8]
training_targets = dataset[:, 8]

#  Create model:
#   'relu' Activation function for input layer.
#   We use a 'sigmoid' activation function on the output layer to ensure our network output
#   is between 0 and 1 and easy to map to either a probability of class 1 or snap to a hard classification.
#   of either class with a default threshold of 0.5.
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training_data, training_targets, epochs=150, batch_size=10)
scores = model.evaluate(training_data, training_targets)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
