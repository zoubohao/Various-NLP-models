import tensorflow as tf
from tensorflow import keras
import numpy as np

maskNp = [[[0, 0, 0, 0, -1e10],
           [0, 0, 0, 0, -1e10],
           [0, 0, 0, 0, -1e10],
           [0, 0, 0, 0, -1e10],
           [-1e10, -1e10, -1e10, -1e10, -1e10]] for _ in range(3)]
maskTest = tf.convert_to_tensor(np.array(maskNp, dtype=np.float32), dtype=tf.float32)
x = tf.zeros(shape=[5,5],dtype=tf.float32)
print(tf.add(x,maskTest))
