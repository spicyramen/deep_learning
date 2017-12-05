# MLP with manual validation set
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load pima indians dataset
FILENAME = '../data/pima-indians-diabetes.csv'

dataset = np.loadtxt(FILENAME, delimiter=',')
# split into input (X) and output (Y) variables
features = dataset[:, 0:8]
labels = dataset[:, 8]
# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=seed)  # create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=10)
model.summary()