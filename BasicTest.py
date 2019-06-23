import tensorflow as tf
from tensorflow import keras
import numpy as np

a = [1,2,3,4]
b = [2,3,4,5]
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions so that we can add the padding
    # to the attention logits.
    # tf.newaxis is an place holder, it can add in one of tensor shape and changes the shape of this tensor.
    # But, it can only add 1 in shape of one tensor.
    return seq[:,tf.newaxis,tf.newaxis :]  # (batch_size, 1, 1, seq_len)
print("Create Mask : " , create_padding_mask([[4,3,2,0,0],[3,4,5,6,1]]))
print("Reshape Tensor : ",tf.reshape(create_padding_mask([[4,3,2,0,0],[3,4,5,6,1]]),[5,2]))
print("Concat Tensor : ",tf.concat((a,b),axis=0))


### It can transform the python language to tensorflow engine machine code by using the decorator tf.function.
@tf.function
def pythonControlFlow(x,threshold):
    x = tf.maximum(x,threshold,name="Maximum")
    return x
#testTensor = tf.convert_to_tensor(np.ones(shape=[2,3],dtype=np.float32),dtype=tf.float32)



class MyDense(keras.layers.Layer):

    def __init__(self,units):
    ### dynamic: Set this to `True` if your layer should only be run eagerly, and
    ### should not be used to generate a static computation graph.
    ### This would be the case for a Tree-RNN or a recursive network,
    ### for example, or generally for any layer that manipulates tensors
    ### using Python control flow. If `False`, we assume that the layer can
    ### safely be used to generate a static computation graph.
        super(MyDense,self).__init__(dynamic=False)
        self.units = units
        self.w = None
        self.b = None
        self.bn = keras.layers.BatchNormalization(axis=-1,epsilon=0.0001)

    def build(self, input_shape):
        batchNumber , cells = input_shape[0] , input_shape[1]
        self.w = self.add_weight(name="Weight",shape=[cells,self.units],dtype=tf.float32,
                                 initializer=keras.initializers.GlorotNormal(),
                                 regularizer=keras.regularizers.L1L2(l1=0.001,l2=0.002),
                                 trainable=True)
        self.b = self.add_weight(name="Bias",shape=[self.units],dtype=tf.float32,
                                 initializer=keras.initializers.constant(0.0),
                                 regularizer=keras.regularizers.l2(l=0.02),
                                 trainable=True)
        self.add_loss(0.001 * tf.reduce_mean(self.w))
        super(MyDense,self).build(input_shape)

    def call(self, inputs, training = False):
        x = tf.matmul(inputs,self.w,name="MatrixMul") + self.b
        x = self.bn(x,training)
        x = tf.nn.relu(x)
        return x

class MyNet(keras.Model):

    def __init__(self):
        super(MyNet,self).__init__(name="MLP")
        self.mLayers = []
        for i in range(4):
            self.mLayers.append(MyDense(5))
        self.mLayers.append(keras.layers.Dense(1))
        self.KlLoss = []

    def call(self, inputs, training=False, mask=None):
        for layer in self.mLayers:
            inputs = layer(inputs)
            self.KlLoss.append(tf.multiply(tf.reduce_mean(inputs),0.01))
        return inputs

gru = keras.layers.LSTMCell(units=5)
print(gru.dynamic)
epoch = 50
timesInOneEpoch = 200
myNet = MyNet()
testTensor = tf.convert_to_tensor(np.array([[3,4,5,6,7],[6,7,8,9,10]],dtype=np.float32),dtype=tf.float32)
print("Test myNet result : ",myNet(testTensor,training = False))
lossFun = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam()
# myNet.compile(optimizer=optimizer,loss=lossFun)
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir="D://myNet")
# myNet.fit(
#     testTensor,
#     tf.convert_to_tensor([[-1.0],[1.0]],dtype=tf.float32),
#     batch_size=2,
#     epochs=1,
#     callbacks=[tensorboard_callback])
trainingTimes = 0
for e in range(epoch):
    for t in range(timesInOneEpoch):
        with tf.GradientTape() as tape:
            logits = myNet(testTensor,training = True)
            loss = lossFun(logits,[[-1.0],[1.0]]) + tf.multiply(
                tf.add_n([tf.nn.l2_loss(varias) for varias in myNet.trainable_weights]),0.001) +\
                tf.add_n([lo for lo in myNet.KlLoss])
            gradients = tape.gradient(loss,myNet.trainable_weights)
        optimizer.apply_gradients(zip(gradients,myNet.trainable_weights))
        if trainingTimes % 100. == 0.0 :
            #optimizer = keras.optimizers.Adam()
            print("Times : ",trainingTimes)
            print(myNet(testTensor, training=False))
            print("Loss : ", float(loss))
        trainingTimes += 1




