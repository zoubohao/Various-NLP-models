import tensorflow as tf
from tensorflow import keras
import numpy as np

a = tf.convert_to_tensor(np.array(np.random.randn(3,5,6),dtype=np.float32))
b = tf.convert_to_tensor(np.array(np.random.randn(3,5,6),dtype=np.float32))
print(tf.matmul(a,b,transpose_b=True))
dense = keras.layers.Dense(64)
print(dense(a))



