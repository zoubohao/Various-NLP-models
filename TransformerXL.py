import tensorflow as tf
import numpy as np
import Transformer as tsf
from tensorflow import keras

class TransformerXLEncoder(keras.Model) :

    def __init__(self,sLength,
                 numberOfLayers,selfAttentionSize,interMediumDim,
                 wordEmbeddingSize):
        super(TransformerXLEncoder,self).__init__()
        self.L = numberOfLayers
        self.tsfEncoderList = [tsf.TransformerEncoder(h=selfAttentionSize,
                                                      interMediumDim=interMediumDim,
                                                      dk=wordEmbeddingSize) for _ in range(numberOfLayers)]
        self.cacheMatrix = [tf.zeros(shape=[sLength,wordEmbeddingSize],dtype=tf.float32)
                            for _ in range(numberOfLayers)]
        self.WQ = [keras.layers.Dense(wordEmbeddingSize) for _ in range(numberOfLayers)]
        self.WK = [keras.layers.Dense(wordEmbeddingSize) for _ in range(numberOfLayers)]
        self.WV = [keras.layers.Dense(wordEmbeddingSize) for _ in range(numberOfLayers)]

    ### this inputs is inputsEmbeddingMatrix
    ### shape of it is [b , max , wordEmbedding]
    def call(self, inputs, training=None, mask=None):
        sentenceMatrixList = tf.unstack(inputs,axis=0)
        outList = []
        ### operation in batch dimension.
        for ht in sentenceMatrixList:
            ### operations in n transformer encoder layers.
            for i in range(self.L):
                ht_1 = self.cacheMatrix[i]
                ht_1 = tf.stop_gradient(ht_1)
                concatTensor = tf.concat([ht_1,ht],axis=-1)
                QMatrix = self.WQ[i](concatTensor)
                KMatrix = self.WK[i](concatTensor)
                VMatrix = self.WV[i](concatTensor)
                ht = self.tsfEncoderList[i]((QMatrix,KMatrix,VMatrix),training,mask = mask)
                self.cacheMatrix[i] = ht
            outList.append(ht)
        encoderStates = tf.stack(outList,axis=0)
        return encoderStates



### cache ht_1 : [n , hiddenStates]
### new ht : [n , hiddenState]
### hiddenStates : [maxTimes , hiddenUnits]
### if concat ht_1 and ht , the length will be overflowed the maxTimes size.
### For dealing this problem, we need to use an W. to down sample the double maxTimes size into maxTimes size.

class TransformerXL(keras.Model) :

    def __init__(self,sLength,
                 numberOfLayers,selfAttentionSize,interMediumDim,
                 wordEmbeddingSize,
                 outDim):
        super(TransformerXL,self).__init__()
        self.L = numberOfLayers
        self.transformerXLEncoder = TransformerXLEncoder(sLength=sLength,
                                                        numberOfLayers=numberOfLayers,
                                                        selfAttentionSize=selfAttentionSize,
                                                        interMediumDim=interMediumDim,
                                                        wordEmbeddingSize=wordEmbeddingSize)
        self.tsfDecoderList = [tsf.TransformerDecoder(h=selfAttentionSize,
                                                      interMediumDim=interMediumDim,
                                                      dk=wordEmbeddingSize) for _ in range(numberOfLayers)]
        self.denseOut = keras.layers.Dense(outDim)

    ### The inputs are composed with inputsEmbeddingMatrix and outputEmbeddingMatrix
    ### inputs = (inputsEmbeddingMatrix, outputEmbeddingMatrix)
    ### the shape of inputsEmbeddingMatrix is [batchSize , sLength , embedding]
    ### The TransformerXL is to use the information from before matrix
    ### and dose not calculate gradients of before matrix .
    ### The shape of output is the same as inputs
    def call(self, inputs, training=None, mask=None):
        inputsEmbeddingMatrix , outputEmbeddingMatrix = inputs
        encoderStates = self.transformerXLEncoder(inputsEmbeddingMatrix,training,mask = None)
        x = tf.identity(outputEmbeddingMatrix)
        for i in range(self.L):
            x = self.tsfDecoderList[i]((x,encoderStates),training,mask = mask)
        sL = x.shape[1]
        wE = x.shape[2]
        x = tf.reshape(x,[-1,sL * wE])
        x = self.denseOut(x)
        return tf.nn.softmax(x)




if __name__ == "__main__":
    maskNp = [[0, 0, 0, 0, -1e10],
               [0, 0, 0, 0, -1e10],
               [0, 0, 0, 0, -1e10],
               [0, 0, 0, 0, -1e10],
               [-1e10, -1e10, -1e10, -1e10, -1e10]]
    maskTest = tf.convert_to_tensor(np.array(maskNp, dtype=np.float32), dtype=tf.float32)
    testInputsEmbedding = np.array(np.random.randn(3, 5, 8), dtype=np.float32)
    testOutputsEmbedding = np.array(np.random.randn(3, 5, 8), dtype=np.float32)
    transformerXL = TransformerXL(sLength=5,numberOfLayers=3,
                                  selfAttentionSize=8,
                                  interMediumDim=15,
                                  wordEmbeddingSize=8,
                                  outDim=5)
    thisStepsLabel = [[0., 0., 0., 1., 0.],
                      [0., 1., 0., 0., 0.],
                      [0., 0., 0., 0., 1.]]
    lossFun = keras.losses.MeanAbsoluteError()
    optimizer = keras.optimizers.Adam()
    trainingTimes = 0
    epoch = 50
    timesInOneEpoch = 200
    for e in range(epoch):
        for ti in range(timesInOneEpoch):
            with tf.GradientTape() as tape:
                logits = transformerXL((testInputsEmbedding, testOutputsEmbedding), training=True, mask=maskTest)
                losses = lossFun(logits, thisStepsLabel) + \
                         tf.multiply(tf.add_n([tf.nn.l2_loss(varias) for varias in transformerXL.trainable_weights]),
                                     0.0005)
                gradients = tape.gradient(losses, transformerXL.trainable_weights)
            if trainingTimes % 100. == 0.0:
                print("Times : ", trainingTimes)
                print(transformerXL((testInputsEmbedding, testOutputsEmbedding), training=False, mask=maskTest))
                print("Loss : ", float(losses))
            optimizer.apply_gradients(zip(gradients, transformerXL.trainable_weights))
            trainingTimes += 1






