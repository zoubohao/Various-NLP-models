import tensorflow as tf
from tensorflow import keras
import numpy as np


### Q:[ b, maxTimes , wordEmbedding ]
### K:[ b, maxTimes , wordEmbedding ]
### V:[ b, maxTimes , wordEmbedding ]
class SelfAttention(keras.Model) :
    """
    ### Q:[ b, maxTimes , wordEmbedding ]
    ### K:[ b, maxTimes , wordEmbedding ]
    ### V:[ b, maxTimes , wordEmbedding ]
    """
    def __init__(self):
        """
    ### mask : it is an mask which o prevent rightward information flow in the decoder to
    ### preserve the auto-regressive property
    ### As usual, it is an SUB-MATRIX in the whole matrix.
    ### if the time is at 4 step , the max times is 5 . The mask needs to be like this:
    ###[[0, 0, 0, 0, -1e10],
    ### [0, 0, 0, 0, -1e10],
    ### [0, 0, 0, 0, -1e10],
    ### [0, 0, 0, 0, -1e10],
    ### [-1e10,-1e10,-1e10,-1e10,-1e10]]
        """
        super(SelfAttention,self).__init__()

    ### mask : it is an mask which o prevent rightward information flow in the decoder to
    ### preserve the auto-regressive property
    ### As usual, it is an SUB-MATRIX in the whole matrix.
    ### if the time is at 4 step , the max times is 5 . The mask needs to be like this:
    ###[[0, 0, 0, 0, -1e10],
    ### [0, 0, 0, 0, -1e10],
    ### [0, 0, 0, 0, -1e10],
    ### [0, 0, 0, 0, -1e10],
    ### [-1e10,-1e10,-1e10,-1e10,-1e10]]
    def call(self, inputs, training=None, mask=None):
        Q , K , V = inputs
        if len(Q.shape) == 2:
            dk = Q.shape[1]
        else:
            dk = Q.shape[2]
        ### x : [m , m]
        x = tf.scalar_mul(1. / dk , tf.matmul(Q,K,transpose_b=True))
        ### this is an added operation to add mask into x tensor.
        ### if x is three dimensions, the mask need also to be three dimension.
        if mask is not None :
            assert len(x.shape) == len(mask.shape) , "The dimensions of x and mask are not same !!!"
            x = tf.add(x,mask)
        x = tf.nn.softmax(x)
        ### V : [m , dv]
        x = tf.matmul(x , V)
        return x

class MultiHeadAttention(keras.Model) :

    def __init__(self,h,dk,outDim):
        super(MultiHeadAttention,self).__init__()
        self.dk = dk
        self.dv = dk
        self.h = h
        self.QDenses = [keras.layers.Dense(self.dk) for _ in range(self.h)]
        self.KDenses = [keras.layers.Dense(self.dk) for _ in range(self.h)]
        self.VDenses = [keras.layers.Dense(self.dv) for _ in range(self.h)]
        self.selfAttentions = [SelfAttention() for _ in range(self.h)]
        self.finalLiner = keras.layers.Dense(outDim)

    def call(self, inputs, training=None, mask=None):
        Q , K , V = inputs
        tensors = []
        for i in range(self.h) :
            tensors.append(self.selfAttentions[i]((self.QDenses[i](Q),
                                                   self.KDenses[i](K),
                                                   self.VDenses[i](V)),mask = mask))
        concatTensor = tf.concat(tensors,axis=-1)
        return self.finalLiner(concatTensor)

class FeedForward(keras.Model) :

    def __init__(self,interMediumDim , outDim):
        super(FeedForward,self).__init__()
        self.dense0 = keras.layers.Dense(interMediumDim)
        self.bn0 = keras.layers.BatchNormalization()
        self.prelu0 = keras.layers.PReLU()
        self.dropout0 = keras.layers.Dropout(rate=0.5)
        self.dense1 = keras.layers.Dense(interMediumDim)
        self.bn1 = keras.layers.BatchNormalization()
        self.prelu1 = keras.layers.PReLU()
        self.dropout1 = keras.layers.Dropout(rate=0.5)
        self.dense2 = keras.layers.Dense(outDim)


    def call(self, inputs, training=None, mask=None):
        x = self.dense0(inputs)
        x = self.bn0(x,training = training)
        x = self.prelu0(x)
        x = self.dropout0(x,training=training)
        x = self.dense1(x)
        x = self.bn1(x,training = training)
        x = self.prelu1(x)
        x = self.dropout1(x,training=training)
        return self.dense2(x)

class TransformerEncoder(keras.Model) :
    """
    In the call function , ### The inputs consist of Q, K, V.
    """

    def __init__(self,h,interMediumDim,dk):
        super(TransformerEncoder,self).__init__()
        self.multiHead = MultiHeadAttention(h = h, dk=dk, outDim=dk)
        self.feedforward = FeedForward(interMediumDim,dk)
        self.layerNorm0 = keras.layers.LayerNormalization()
        self.layerNorm1 = keras.layers.LayerNormalization()

    ### The inputs consist of Q, K, V.
    def call(self, inputs, training=None, mask=None):
        Q, K, V = inputs
        oriMatrix = tf.identity(Q)
        mutiHeadMatrix = self.multiHead((Q,K,V),training = training,mask = mask)
        addedTensor0 = tf.add(oriMatrix,mutiHeadMatrix)
        addedNormal0 = self.layerNorm0(addedTensor0)
        addedIdentity = tf.identity(addedNormal0)
        feedforwardM = self.feedforward(addedNormal0,training = training)
        addedTensor1 = tf.add(addedIdentity,feedforwardM)
        addedNormal1 = self.layerNorm1(addedTensor1)
        return addedNormal1

class TransformerDecoder(keras.Model) :
    """
    In the call function ,
    ### the inputs are composed by original matrix and encoded states which encoded from encoder.
    ### inputs = (origMatrix , encodedMatrix)
    """

    def __init__(self,h,interMediumDim,dk):
        super(TransformerDecoder,self).__init__()
        self.maskedMultiHead = MultiHeadAttention(h,dk,dk)
        self.multiHead = MultiHeadAttention(h,dk,dk)
        self.feedforward = FeedForward(interMediumDim,dk)
        self.layerNorm0 = keras.layers.LayerNormalization()
        self.layerNorm1 = keras.layers.LayerNormalization()
        self.layerNorm2 = keras.layers.LayerNormalization()

    ### the inputs are composed by original matrix and encoded states which encoded from encoder.
    ### inputs = (origMatrix , encodedMatrix)
    def call(self, inputs, training=None, mask=None):
        origMatrix, encodedMatrix = inputs
        Q0 = tf.identity(origMatrix)
        K0 = tf.identity(origMatrix)
        V0 = tf.identity(origMatrix)
        maskedMultiMatrix0 = self.maskedMultiHead((Q0,K0,V0),training = training , mask = mask)
        addedNormal0 = self.layerNorm0(tf.add(maskedMultiMatrix0,origMatrix))
        Q1 = tf.identity(addedNormal0)
        K1 = tf.identity(encodedMatrix)
        V1 = tf.identity(encodedMatrix)
        multiHead1 = self.multiHead((Q1,K1,V1),training = training)
        addedNormal1 = self.layerNorm1(tf.add(addedNormal0,multiHead1))
        feedforward = self.feedforward(addedNormal1,training = training)
        return self.layerNorm2(tf.add(addedNormal1,feedforward))

class Transformer(keras.Model) :
    """
    In the call function ,
    ### The inputs are composed with inputsEmbeddingMatrix and outputEmbeddingMatrix
    ### inputs = (inputsEmbeddingMatrix, outputEmbeddingMatrix)
    """

    def __init__(self,interMediumDim,dimOfWordEmbedding,outDim):
        super(Transformer,self).__init__()
        self.encoder0 = TransformerEncoder(h=8,dk=dimOfWordEmbedding,interMediumDim=interMediumDim)
        self.encoder1 = TransformerEncoder(h=8,dk=dimOfWordEmbedding,interMediumDim=interMediumDim)
        self.decoder0 = TransformerDecoder(h=8,dk=dimOfWordEmbedding,interMediumDim=interMediumDim)
        self.decoder1 = TransformerDecoder(h=8,dk=dimOfWordEmbedding,interMediumDim=interMediumDim)
        self.outDense = keras.layers.Dense(outDim)

    ### The inputs are composed with inputsEmbeddingMatrix and outputEmbeddingMatrix
    ### inputs = (inputsEmbeddingMatrix, outputEmbeddingMatrix)
    def call(self, inputs, training=None, mask=None):
        inputsEmbeddingMatrix , outputEmbeddingMatrix = inputs
        encoderS0 = self.encoder0((inputsEmbeddingMatrix,
                                   inputsEmbeddingMatrix,
                                   inputsEmbeddingMatrix),training,mask = None)
        encoderS1 = self.encoder1((encoderS0,
                                   encoderS0,
                                   encoderS0),training,mask = None)
        decoderS0 = self.decoder0((outputEmbeddingMatrix,encoderS1),training,mask = mask)
        decoderS1 = self.decoder1((decoderS0,encoderS1),training,mask = mask)
        mT = decoderS1.shape[1]
        dM = decoderS1.shape[2]
        flattenTensor = tf.reshape(decoderS1,shape=[-1,mT * dM])
        x = self.outDense(flattenTensor)
        x = tf.nn.softmax(x)
        return x



if __name__ == "__main__":
    ### This is an test.
    ### The shape of testInputs is [batchSize , maxTimes , wordEmbedding]
    ### The shape of testOutputs is the same as testInputs.
    ### The mask is at the 4-th step and the max times are 5.
    ### The thisStepsLabel is to simulate which word in the dictionary should be selected.
    maskNp = [[[0, 0, 0, 0, -1e10],
               [0, 0, 0, 0, -1e10],
               [0, 0, 0, 0, -1e10],
               [0, 0, 0, 0, -1e10],
               [-1e10, -1e10, -1e10, -1e10, -1e10]]  for _ in range(3)]
    maskTest = tf.convert_to_tensor(np.array(maskNp, dtype=np.float32), dtype=tf.float32)
    testInputs = np.array(np.random.randn(3, 5, 8), dtype=np.float32)
    testOutputs = np.array(np.random.randn(3, 5 ,8),dtype=np.float32)
    transformer = Transformer(8,8,5)
    thisStepsLabel = [[0.,0.,0.,1.,0.],
                      [0.,1.,0.,0.,0.],
                      [0.,0.,0.,0.,1.]]
    lossFun = keras.losses.MeanAbsoluteError()
    optimizer = keras.optimizers.Adam()
    trainingTimes = 0
    epoch = 50
    timesInOneEpoch = 200
    for e in range(epoch) :
        for ti in range(timesInOneEpoch) :
            with tf.GradientTape() as tape:
                logits = transformer((testInputs,testOutputs),training = True , mask = maskTest)
                losses = lossFun(logits,thisStepsLabel) + \
                         tf.multiply(tf.add_n([tf.nn.l2_loss(varias) for varias in transformer.trainable_weights]),0.0005)
                gradients = tape.gradient(losses, transformer.trainable_weights)
            if trainingTimes % 100. == 0.0:
                print("Times : ", trainingTimes)
                print(transformer((testInputs,testOutputs), training=False,mask = maskTest))
                print("Loss : ", float(losses))
            optimizer.apply_gradients(zip(gradients, transformer.trainable_weights))
            trainingTimes += 1





