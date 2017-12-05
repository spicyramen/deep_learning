# Plot ad hoc CIFAR10 instances.
from keras.datasets import cifar10
from PIL import Image
# Bad hack
Image.Image.tostring = Image.Image.tobytes

from matplotlib import pyplot
from scipy.misc import toimage

# Load data.
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Create a grid of 3x3 images.
for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(toimage(X_train[i]))

# Show the plot.
pyplot.show()
