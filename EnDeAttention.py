import  tensorflow as tf
from tensorflow import keras
import numpy as np



### input shape : [batchSize ,  maxTimeSteps , embeddingNumber]
class EncoderNet(keras.Model):
    """
        ### The for loop is implement in the dimension of maxTimeSteps
        ### Input needs shaped into [maxTimeSteps , batchSize , embeddingNumber]
        ### The shape of states is [batchSize , units]
    """

    def __init__(self,units):
        """

        :param units: ### The units mean that the out space of tensor
        """
        super(EncoderNet,self).__init__()
        self.cell0 = keras.layers.GRUCell(units,dropout=0.2,recurrent_dropout=0.3)
        self.cell1 = keras.layers.GRUCell(units,dropout=0.2,recurrent_dropout=0.3)

    def call(self, inputs, training = None, mask=None,states = None):
        ### The for loop is implement in the dimension of maxTimeSteps
        ### Input needs shaped into [maxTimeSteps , batchSize , embeddingNumber]
        ### The shape of states is [batchSize , units]
        b = inputs.shape[0]
        t = inputs.shape[1]
        n = inputs.shape[2]
        reStates = [tf.identity(states)]
        states = [states]
        inputs = tf.reshape(inputs,[t,b,n])
        h_list = []
        for i in range(t):
            outputThisStep , states = self.cell0(inputs[i],training = training , states = states)
            reOutputThisStep , reStates = self.cell1(inputs[t - 1 - i], training = training, states = reStates)
            c = tf.divide(tf.add(outputThisStep,reOutputThisStep),2.0)
            h_list.append(c)
        return h_list[-1] , tf.stack(h_list,axis=0)

### InputShape : [batchSize , concat(Si_1Tensor.shape[1], HiddenTensors.shape[1])]
class FeedForward(keras.Model) :
    """
    In the call function,
    ### InputShape : [batchSize , concat(Si_1Tensor.shape[1], HiddenTensors.shape[1])]
    """

    def __init__(self,outputDim):
        super(FeedForward,self).__init__()
        self.dense0 = keras.layers.Dense(10)
        self.bn0 = keras.layers.BatchNormalization(axis=-1,epsilon=1e-5)
        self.dense1 = keras.layers.Dense(5)
        self.bn1 = keras.layers.BatchNormalization(axis=-1,epsilon=1e-5)
        self.dense2 = keras.layers.Dense(outputDim)


    def call(self, inputs, training=None, mask=None):
        x = self.dense0(inputs)
        x = self.bn0(x,training=training)
        x = tf.nn.relu(x)
        x = self.dense1(x)
        x = self.bn1(x,training = training)
        x = tf.nn.relu(x)
        x = self.dense2(x)
        return x

### Input is a list : [Si_1Tensor , HiddenTensors]
### Si_1Tensor : [batchSize , sunits]
### HiddenTenors : [maxTimesSteps , batchSize , hunits]
class Attention(keras.Model) :
    """
    In the call function ,
    ### Input is a list : [Si_1Tensor , HiddenTensors]
    ### Si_1Tensor : [batchSize , sunits]
    ### HiddenTenors : [maxTimesSteps , batchSize , hunits]
    """

    def __init__(self):
        super(Attention,self).__init__()
        self.feedforword = FeedForward(1)

    def call(self, inputs, training=None, mask=None):
        Si_1Tensor = inputs[0]
        HiddenTensors = inputs[1]
        t = HiddenTensors.shape[0]
        b = HiddenTensors.shape[1]
        hunits = HiddenTensors.shape[2]
        AiList = []
        for hj in HiddenTensors:
            concatTensor = tf.concat((Si_1Tensor,hj),axis=1)
            aij = self.feedforword(concatTensor,training=training)
            AiList.append(aij)
        AiTensor = tf.concat(AiList,axis=0)
        AiSoftmax = tf.nn.softmax(AiTensor,axis=0)
        AijList = tf.unstack(AiSoftmax,axis=0)
        hjList = tf.unstack(HiddenTensors,axis=0)
        Ci = tf.zeros(shape=[b,hunits],dtype=tf.float32)
        for i in range(t) :
            Ci = Ci + tf.scalar_mul(tf.squeeze(AijList[i]),hjList[i])
        return Ci

### Input is a list : [S0 , HiddenTensors]
### Si_1Tensor : [batchSize , sunits]
### HiddenTenors : [maxTimesSteps , batchSize , hunits]
class DecoderNet(keras.Model) :
    """
    In the call funciton ,
    ### Input is a list : [S0 , HiddenTensors]
    ### Si_1Tensor : [batchSize , sunits]
    ### HiddenTenors : [maxTimesSteps , batchSize , hunits]
    """

    def __init__(self,units,times):
        super(DecoderNet,self).__init__()
        self.times = times
        self.cell0 = keras.layers.GRUCell(units,dropout=0.2,recurrent_dropout=0.3)
        self.attention = Attention()

    def call(self, inputs, training=None, mask=None):
        Zi = [tf.identity(inputs[0])]
        HiddenTensors = inputs[1]
        b = HiddenTensors.shape[1]
        hunits = HiddenTensors.shape[2]
        thisStep = tf.zeros(shape=[b,hunits],dtype=tf.float32)
        for i in range(self.times):
            Ci = self.attention([Zi[0] , HiddenTensors],training)
            thisStep , Zi = self.cell0(tf.add(Ci,thisStep),training = training,states = Zi)
        return thisStep

class EDAttentionNet(keras.Model) :

    def __init__(self,units , times , outDim):
        super(EDAttentionNet,self).__init__()
        self.encoder = EncoderNet(units)
        self.decoder = DecoderNet(units,times)
        self.dense = keras.layers.Dense(outDim)

    def call(self, inputs, training=None, mask=None,states = None):
        z0 , hiddenTensors = self.encoder(inputs,training = training, states = states)
        finalStepOutTensor = self.decoder([z0,hiddenTensors],training = training)
        out = self.dense(finalStepOutTensor)
        return out


if __name__ == "__main__" :
    ### input data shape : [batch , maxTimeSteps , wordEmbedding]
    inputTest = np.array(np.random.randn(3, 4, 6), dtype=np.float32)
    ### initial states shape : [batch , units]
    initialState = tf.convert_to_tensor(np.zeros(shape=[3, 7], dtype=np.float32), dtype=tf.float32)
    ### Encoder and decoder with attention RNN model
    model = EDAttentionNet(7,4,1)
    output = model(inputTest,training = False, states = initialState)
    print("Test output is ",output)
    lossFun = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam()
    trainingTimes = 0
    epoch = 50
    timesInOneEpoch = 200
    for e in range(epoch):
        for k in range(timesInOneEpoch):
            with tf.GradientTape() as tape:
                logits = model(inputTest, training=True,states = initialState)
                loss = lossFun(logits, [[-1.0], [1.0],[-1.0]]) + tf.multiply(
                    tf.add_n([tf.nn.l2_loss(varias) for varias in model.trainable_weights]), 0.001)
                gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            if trainingTimes % 100. == 0.0:
                # optimizer = keras.optimizers.Adam()
                print("Times : ", trainingTimes)
                print(model(inputTest, training=True,states = initialState))
                print("Loss : ", float(loss))
            trainingTimes += 1


