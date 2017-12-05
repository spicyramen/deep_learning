import string


def load_doc(filename):
    """Load doc into memory.

    :param filename:
    :return:
    """
    # open the file as read only.
    with open(filename, 'r') as fp:
        return fp.read()


def load_descriptions(doc):
    """Extract descriptions for images.

    :param doc:
    :return:
    """
    mapping = {}
    # Process lines.
    for line in doc.split('\n'):
        # Split line by white space.
        tokens = line.split()
        if len(line) < 2:
            continue
        # Take the first token as the image id, the rest as the description.
        image_id, image_desc = tokens[0], tokens[1:]
        # Extract filename from image id.
        image_id = image_id.split('.')[0]
        # Convert description tokens back to string.
        image_desc = ' '.join(image_desc)
        # Create the mapping list if needed.
        if image_id not in mapping:
            mapping[image_id] = []
        # Store description.
        mapping[image_id].append(image_desc)
    return mapping


def clean_descriptions(descriptions):
    """Prepare translation table for removing punctuation.

    :param descriptions:
    :return:
    """
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # Tokenize.
            desc = desc.split()
            # Convert to lower case.
            desc = [word.lower() for word in desc]
            # Remove punctuation from each token.
            desc = [w.translate(None, string.punctuation) for w in desc]
            # Remove hanging 's' and 'a'.
            desc = [word for word in desc if len(word) > 1]
            # Remove tokens with numbers in them.
            desc = [word for word in desc if word.isalpha()]
            # Store as string.
            desc_list[i] = ' '.join(desc)


def to_vocabulary(descriptions):
    """Convert the loaded descriptions into a vocabulary of words.

    :param descriptions:
    :return:
    """
    # Build a list of all description strings.
    all_desc = set()
    for key in descriptions.iterkeys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


def save_descriptions(descriptions, filename):
    """Save descriptions to file, one per line.

    :param descriptions:
    :param filename:
    :return:
    """
    lines = list()
    for key, desc_list in descriptions.iteritems():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# Load descriptions from Token file.
doc = load_doc('/usr/local/src/data/Flickr8k/Flickr8k.token.txt')

# Parse descriptions.
image_descriptions = load_descriptions(doc)
print('Loaded: %d images' % len(image_descriptions))

# Clean descriptions.
clean_descriptions(image_descriptions)

# Summarize vocabulary.
vocabulary = to_vocabulary(image_descriptions)
print('Vocabulary Size: %d' % len(vocabulary))

# Save to file.
save_descriptions(image_descriptions, 'descriptions.txt')
