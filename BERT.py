import tensorflow as tf
import numpy as np
from tensorflow import keras
import Transformer as tsf


### BERT is just an transformer encoder.
### However, this encoder has prodigious parameters and
### the training of this huge encoder is tough.
### So, this can be an pre-trained model.
class BERT (keras.Model) :

    def __init__(self,numberOfLayers,hiddenSize,numberOfSelfAttentionSize):
        super(BERT,self).__init__()
        self.L = numberOfLayers
        self.TseList = [tsf.TransformerEncoder(h = numberOfSelfAttentionSize,
                                               interMediumDim=4 * hiddenSize,
                                               dk = hiddenSize)  for _ in range(numberOfLayers)]

    ### inputs are consisted of batch of sentences and it must contains some masked words.
    ### In this case, this mask parameter is useless because the inputs have changed some words
    ### in the sentences into symbol of [MASK].
    ### the shape of inputs is [batchSize , maxTimes , wordEmbedding]
    ### the shape of return tensor is the same as inputs .
    def call(self, inputs, training=None, mask=None):
        for l in range(self.L) :
            inputs = self.TseList[l]((inputs,
                                      inputs,
                                      inputs),training)
        return inputs

class PreTrainedModel(keras.Model) :

    def __init__(self,numberOfLayers,hiddenSize,numberOfSelfAttentionSize,outDim):
        super(PreTrainedModel,self).__init__()
        self.bert = BERT(numberOfLayers,hiddenSize,numberOfSelfAttentionSize)
        self.dense = keras.layers.Dense(outDim)

    def call(self, inputs, training=None, mask=None):
        toTensor = self.bert(inputs,training)
        mT = toTensor.shape[1]
        wE = toTensor.shape[2]
        flattenTensor = tf.reshape(toTensor,[-1,mT * wE])
        denseTensor = self.dense(flattenTensor)
        return tf.nn.softmax(denseTensor,axis=-1)

if __name__ == "__main__":
    testInputs = np.array(np.random.randn(3,4,5),dtype=np.float32)
    smallBERT = PreTrainedModel(3,5,8,4)
    testLabels = np.array([[0,0,0,1],
                           [1,0,0,0],
                           [0,1,0,0]],dtype=np.float32)
    loss = keras.losses.MeanAbsoluteError()
    optimizer = keras.optimizers.Adam()
    epoch = 5
    timesInOneEpoch = 200
    trainingTimes = 0
    for e in range(epoch):
        for ts in range(timesInOneEpoch):
            with tf.GradientTape() as tape :
                logits = smallBERT(testInputs,training = True,mask = None)
                losses = loss(logits , testLabels) + \
                    tf.add_n([tf.multiply(0.001,tf.nn.l2_loss(varis))  for varis in smallBERT.trainable_weights])
                gradients = tape.gradient(losses,smallBERT.trainable_weights)
            if trainingTimes % 100 == 0:
                print("Times : ",trainingTimes)
                print("Logits : ", smallBERT(testInputs,False,None))
                print("Losses : ",losses)
            optimizer.apply_gradients(zip(gradients,smallBERT.trainable_weights))
            trainingTimes += 1


