
import tensorflow as tf

tf.get_variable(name='x', shape=[5, 0], dtype=tf.int32)
x = tf.constant(4.0)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print sess.run(x)
