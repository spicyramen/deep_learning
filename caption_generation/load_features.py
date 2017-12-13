from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

# Directory and file information.
_ROOT = '/usr/local/src/data/Flickr8k/'
_FOLDER = _ROOT + 'Flickr8k_images'
_PKL_FILE = 'features.pkl'


def extract_features(directory):
    """Extract features from each photo in the directory.

    :param directory:
    :return:
    """
    # Load the model Oxford Model.
    model = VGG16()
    # Re-structure the model.
    # Remove the last layer from the loaded model, as this is the model used to predict a classification for a photo.
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # Summarize Model.
    print(model.summary())
    # Extract features from each photo.
    features = {}
    for name in listdir(directory):
        # Load an image from file.
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        # Convert the image pixels to a numpy array.
        image = img_to_array(image)
        # Reshape data for the model.
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # Prepare the image for the VGG model.
        image = preprocess_input(image)
        # Get features.
        feature = model.predict(image, verbose=0)
        # Get image id.
        image_id = name.split('.')[0]
        # Store feature.
        features[image_id] = feature
        print('>%s' % name)
    return features


# Extract features from all images.
vgg_features = extract_features(_FOLDER)
print('Extracted Features: %d' % len(vgg_features))
# Save to file.
dump(vgg_features, open(_PKL_FILE, 'wb'))
