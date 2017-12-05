import string

from numpy import array
from pickle import load

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

_DIRECTORY = 'Flicker8k_Dataset'


def load_doc(filename):
    """Load doc into memory.

    :param filename:
    :return:
    """
    # open the file as read only
    with open(filename, 'r') as fp:
        return fp.read()


def load_descriptions(doc):
    """Extract descriptions for images.

    :param doc:
    :return:
    """
    mapping = dict()
    # process lines
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # remove filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # create the list if needed
        if image_id not in mapping:
            mapping[image_id] = list()
        # store description
        mapping[image_id].append(image_desc)
    return mapping  # parse descriptions


def clean_descriptions(descriptions):
    """Prepare translation table for removing punctuation.

    :param descriptions:
    :return:
    """
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case.
            desc = [word.lower() for word in desc]
            # remove punctuation from each token.
            desc = [w.translate(None, string.punctuation) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word) > 1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] = ' '.join(desc)  # clean descriptions.


def to_vocabulary(descriptions):
    """Convert the loaded descriptions into a vocabulary of words.

    :param descriptions:
    :return:
    """
    # build a list of all description strings
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    with open(filename, 'w') as fp:
        fp.write(data)


def load_set(filename):
    """Load a pre-defined list of photo identifiers.

    :param filename:
    :return:
    """
    doc = load_doc(filename)
    dataset = []
    # Process line by line.
    for line in doc.split('\n'):
        # Skip empty lines.
        if len(line) < 1:
            continue
        # Get the image identifier.
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


def load_clean_descriptions(filename, dataset):
    """Load clean descriptions into memory.

    :param filename:
    :param dataset:
    :return:
    """

    doc = load_doc(filename)
    descriptions = {}
    for line in doc.split('\n'):
        # Split line by white space.
        tokens = line.split()
        # Split id from description.
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set.
        if image_id in dataset:
            # create list.
            if image_id not in descriptions:
                descriptions[image_id] = []
            # Wrap description in tokens.
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions


def load_photo_features(filename, dataset):
    """load photo features.


    :param filename:
    :param dataset:
    :return:
    """
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features


def to_lines(descriptions):
    """Convert a dictionary of clean descriptions to a list of descriptions.

    :param descriptions:
    :return:
    """
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


def create_tokenizer(descriptions):
    """fit a tokenizer given caption descriptions

    :param descriptions:
    :return:
    """
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer  # prepare tokenizer


def create_sequences(tokenizer, max_length, descriptions, photos):
    """Creates sequences of images, input sequences and output words for an image.

    :param tokenizer:
    :param max_length:
    :param descriptions:
    :param photos:
    :return:
    """
    X1, X2, y = [], [], []
    # Walk through each image identifier.
    for key, desc_list in descriptions.items():
        # walk through each description for the image
        for desc in desc_list:
            # Encode the sequence.
            seq = tokenizer.texts_to_sequences([desc])[0]
            # Split one sequence into multiple X,Y pairs.
            for i in range(1, len(seq)):
                # Split into input and output pair.
                in_seq, out_seq = seq[:i], seq[i]
                # Pad input sequence.
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # Encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store.
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)


def max_length(descriptions):
    """Calculate the length of the description with the most words.

    :param descriptions:
    :return:
    """
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


def define_model(vocab_size, max_length):
    """Define the captioning model.

    :param vocab_size:
    :param max_length:
    :return:
    """
    # Feature extractor model.
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # Sequence model.
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # Decoder model.
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # Tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # Summarize model
    print(model.summary())
    # plot_model(model, to_file='model.png', show_shapes=True)
    return model


# Train dataset:

# Loads training dataset (6K).
train_images_filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(train_images_filename)
print('Dataset: %d images.' % len(train))

# Load Descriptions.
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# Photo features.
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))

# Determine the maximum sequence length.
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# Prepare sequences.
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)

# dev dataset:

# Load test set.
test_images_filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
test = load_set(test_images_filename)
print('Dataset: %d' % len(test))

# Descriptions.
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))

# Photo features.
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))

# Prepare sequences.
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)

# Fits Model:

# Define the model.
model = define_model(vocab_size, max_length)
# define checkpoint callback
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Fit model.
model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint],
          validation_data=([X1test, X2test], ytest))
