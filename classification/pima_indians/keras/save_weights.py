# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
import json
from keras.models import model_from_json

FILENAME = '../data/pima-indians-diabetes.csv'

LOSS_FUNCTION = 'binary_crossentropy'
OPTIMIZER = 'adam'
INIT = 'uniform'
ACTIVATION = 'relu'
ACTIVATION_OUTPUT = 'sigmoid'

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# Load pima indians dataset
dataset = numpy.loadtxt(FILENAME, delimiter=',')
print type(dataset)
# split into input (X) and output (Y) variables
features = dataset[:, 0:8]
labels = dataset[:, 8]

####### CREATE MODEL #######
model = Sequential()

# RELU Activation function for input layer
model.add(Dense(12, input_dim=8, init=INIT, activation=ACTIVATION))

# RELU Activation function for input layer
model.add(Dense(8, init=INIT, activation=ACTIVATION))

# We use a sigmoid activation function on the output layer to ensure our network output
# is between 0 and 1 and easy to map to either a probability of class 1 or snap to a hard classification
# of either class with a default threshold of 0.5
# Sigmoid Activation function for output layer
model.add(Dense(1, init=INIT, activation=ACTIVATION_OUTPUT))
####### COMPILE MODEL #######
model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=['accuracy'])  # Fit the model
####### FIT MODEL #######
model.fit(features, labels, epochs=5, batch_size=1)
####### EVALUATE MODEL #######
scores = model.evaluate(features, labels)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# Save data
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
print type(model)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
print type(loaded_model)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(features, labels, verbose=0)
print "%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100)
