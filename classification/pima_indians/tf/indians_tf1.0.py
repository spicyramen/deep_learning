# # Create your first MLP in Keras

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

print("Tensorflow version: " + tf.__version__)
tf.set_random_seed(0)

FILENAME = '../data/pima-indians-diabetes.csv'
NLABELS = 2
_LEARNING_RATE = 0.003
_NUM_FEATURES = 8
_NUM_EPOCHS = 150
_BATCH_SIZE = 10

sess = tf.InteractiveSession()


def get_training_data():
    dataset = np.loadtxt(FILENAME, delimiter=",")
    training_data = dataset[:, 0:8].astype(float)
    training_label = dataset[:, 8]
    return train_test_split(training_data, training_label, test_size=0.2, random_state=42)


# Create the model
x = tf.placeholder(tf.float32, [None, 784], name='x-input')
W = tf.Variable(tf.zeros([784, NLABELS]), name='weights')
b = tf.Variable(tf.zeros([NLABELS], name='bias'))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# Add summary ops to collect data
_ = tf.histogram_summary('weights', W)
_ = tf.histogram_summary('biases', b)
_ = tf.histogram_summary('y', y)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, NLABELS], name='y-input')

# More name scopes will clean up the graph representation
with tf.name_scope('cross_entropy'):
    cross_entropy = -tf.reduce_mean(y_ * tf.log(y))
    _ = tf.scalar_summary('cross entropy', cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(10.).minimize(cross_entropy)

with tf.name_scope('test'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    _ = tf.scalar_summary('accuracy', accuracy)

# Merge all the summaries and write them out to /tmp/mnist_logs
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter('logs', sess.graph_def)
tf.initialize_all_variables().run()

# Train the model, and feed in test data and record summaries every 10 steps

for i in range(1000):
    if i % 10 == 0:  # Record summary data and the accuracy
        training_data, test_data, training_labels, test_labels = get_training_data()
        labels = training_labels
        feed = {x: training_data, y_: labels}

        result = sess.run([merged, accuracy, cross_entropy], feed_dict=feed)
        summary_str = result[0]
        acc = result[1]
        loss = result[2]
        writer.add_summary(summary_str, i)
        print('Accuracy at step %s: %s - loss: %f' % (i, acc, loss))
sess.run(train_step, feed_dict=feed)
