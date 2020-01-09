import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
a = tf.constant(1.5)
b = tf.constant(2.5)
with tf.Session() as sess:
    x = sess.run(a+b)
    print(x)
    print('GPU:', tf.test.is_gpu_available())
