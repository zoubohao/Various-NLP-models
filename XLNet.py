import tensorflow as tf
import numpy as np
import Transformer as tsf
from tensorflow import keras
import TransformerXL as tsfXL


class QueryStreamTransformerXLEncoder (keras.Model) :

    def __init__(self,batchSize,sLength,wordEmbeddingSize,
                 numberOfLayers,selfAttentionSize,interMediumDim):
        super(QueryStreamTransformerXLEncoder,self).__init__()
        self.L = numberOfLayers
        self.B = batchSize
        self.tsfEncoderList = [tsf.TransformerEncoder(h=selfAttentionSize,
                                                      interMediumDim=interMediumDim,
                                                      dk=wordEmbeddingSize) for _ in range(numberOfLayers)]
        self.cacheMatrix = [tf.zeros(shape=[sLength,wordEmbeddingSize],dtype=tf.float32)
                            for _ in range(numberOfLayers)]
        self.WQ = [keras.layers.Dense(wordEmbeddingSize) for _ in range(numberOfLayers)]
        self.WK = [keras.layers.Dense(wordEmbeddingSize) for _ in range(numberOfLayers)]
        self.WV = [keras.layers.Dense(wordEmbeddingSize) for _ in range(numberOfLayers)]


    ### The inputs consist of double component.
    ### The first is a sentence , the shape of it is [b , maxTimes , wordEmbedding]
    ### The second is a G_Matrix. The shape is [b , maxTimes , wordEmbedding], but
    ### it is an trainable weight.
    def call(self, inputs, training=None, mask=None):
        inputsWordEmbedding , G_Matrix = inputs
        sentenceMatrixList = tf.unstack(inputsWordEmbedding,axis=0)
        G_Matrix_List = tf.unstack(G_Matrix,axis=0)
        outList = []
        ### operation in batch dimension.
        for bi in range(self.B):
            ### ht needs to have padding mask to delete the information of ht0
            ht = sentenceMatrixList[bi]
            ### Qi needs to have padding mask to delete the information except Gt0
            Qi = G_Matrix_List[bi]
            ### operations in n transformer encoder layers.
            for i in range(self.L):
                ht_1 = self.cacheMatrix[i]
                ht_1 = tf.stop_gradient(ht_1)
                concatTensorH = tf.concat([ht_1,ht],axis=-1)
                concatTensorQ = tf.concat([ht_1,Qi],axis=-1)
                QMatrix = self.WQ[i](concatTensorQ)
                KMatrix = self.WK[i](concatTensorH)
                VMatrix = self.WV[i](concatTensorH)
                ht = self.tsfEncoderList[i]((QMatrix,KMatrix,VMatrix),training,mask = mask)
                self.cacheMatrix[i] = ht
            outList.append(ht)
        encoderStates = tf.stack(outList,axis=0)
        return encoderStates

class MaskedTwoStreamAttention(keras.Model):

    def __init__(self,batchSize,sLength,wordEmbeddingSize,
                 numberOfLayers,selfAttentionSize,interMediumDim):
        self.B = batchSize
        self.T = sLength
        self.W = wordEmbeddingSize
        super(MaskedTwoStreamAttention,self).__init__()
        self.queryAttention = QueryStreamTransformerXLEncoder(batchSize,sLength,wordEmbeddingSize,
                                                              numberOfLayers,selfAttentionSize,interMediumDim)
        self.contentAttention = tsfXL.TransformerXLEncoder(sLength,numberOfLayers,
                                                           selfAttentionSize,interMediumDim,wordEmbeddingSize)

    ### The inputs consist of three components.
    ### The first is a sentence , the shape of it is [b , maxTimes , wordEmbedding]
    ### The second is a G_Matrix. The shape is [b , maxTimes , wordEmbedding], but
    ### it is an trainable weight.
    ### The third is position that needs to predict.
    ##################################################
    #### Mask is composed by mask0 and mask1.        #
    #### The mask0 is for query stream attention     #
    #### The mask1 is for content stream attention   #
    ##################################################
    def call(self, inputs,training=None, mask=None):
        inputsWordEmbedding, G_Matrix , predictPosition = inputs
        if mask is not None:
            mask0, mask1 = mask
        else:
            mask0 = mask1 = None
        paddingMaskGMatrix = np.zeros(shape=[self.B,self.T,self.W])
        paddingMaskGMatrix[:,predictPosition,:] = [1 for _ in range(self.W)]
        paddingMaskGMatrix = tf.convert_to_tensor(paddingMaskGMatrix,dtype=tf.float32)
        paddingMaskInput = np.ones(shape=[self.B,self.T,self.W])
        paddingMaskInput[:,predictPosition,:] = [0 for _ in range(self.W)]
        paddingMaskInput = tf.convert_to_tensor(paddingMaskInput,dtype=tf.float32)
        gStates = self.queryAttention((tf.multiply(inputsWordEmbedding,paddingMaskInput),
                                       tf.multiply(G_Matrix,paddingMaskGMatrix)),training,mask0)
        hStates = self.contentAttention(inputsWordEmbedding,training,mask1)
        return hStates , gStates


class XLNet (keras.Model) :

    def __init__(self,batchSize,sLength,wordEmbeddingSize,
                 numberOfLayers,selfAttentionSize,interMediumDim,
                 outDim):
        super(XLNet,self).__init__()
        self.G_Matrix = self.add_weight("G_Matrix", shape=[batchSize,sLength,wordEmbeddingSize],dtype=tf.float32,
                                        initializer=keras.initializers.glorot_normal())
        self.maskTwoStreamAttention0 = MaskedTwoStreamAttention(batchSize,sLength,wordEmbeddingSize,
                                                               numberOfLayers,selfAttentionSize,interMediumDim)
        self.maskTwoStreamAttention1 = MaskedTwoStreamAttention(batchSize,sLength,wordEmbeddingSize,
                                                               numberOfLayers,selfAttentionSize,interMediumDim)
        self.dense = keras.layers.Dense(outDim)

    ### the inputs are composed by batch of sentences and a predictPosition.
    ### the shape of inputs is [b , maxTimes , wordEmbedding]
    def call(self, inputs, training=None, mask=None):
        inputEmbedding , predictPosition = inputs
        hStates0 , gStates0 = self.maskTwoStreamAttention0((inputEmbedding,self.G_Matrix,predictPosition),training,mask)
        _ , gStates1 = self.maskTwoStreamAttention1((hStates0,gStates0,predictPosition),training,mask)
        mT = gStates1.shape[1]
        wE = gStates1.shape[2]
        flattenTensor = tf.reshape(gStates1,shape=[-1,mT * wE])
        tTensor = self.dense(flattenTensor)
        return tf.nn.softmax(tTensor)

class LossDevice(keras.Model) :

    def __init__(self):
        super(LossDevice,self).__init__()

    def call(self, inputs, training=None, mask=None):
        logs , label = inputs
        return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(label,logs,
                                                                       tf.ones(logs.shape,dtype=tf.float32)))

if __name__ == "__main__":
    testInput = np.array(np.random.randn(3,4,5),dtype=np.float32)
    testLabels = np.array([[1,0,0,0],
                           [0,0,0,1],
                           [0,0,1,0]],dtype=np.float32)
    mask0Test = tf.convert_to_tensor(np.array([[-1e10,0,-1e10,-1e10],
                                               [0,-1e10,-1e10,-1e10],
                                               [-1e10,-1e10,-1e10,-1e10],
                                               [-1e10,-1e10,-1e10,-1e10]]),dtype=tf.float32)
    mask1Test = tf.convert_to_tensor(np.array([[0,0,-1e10,-1e10],
                                               [0,0,-1e10,-1e10],
                                               [-1e10,-1e10,0,-1e10],
                                               [-1e10,-1e10,-1e10,0]]),dtype=tf.float32)
    Model = XLNet(batchSize=3,sLength=4,wordEmbeddingSize=5,
                  numberOfLayers=2,selfAttentionSize=4,interMediumDim=8,
                  outDim=4)
    loss = LossDevice()
    learningRate = 0.001
    opti = tf.optimizers.Adam(learningRate,epsilon=1e-5,amsgrad=True)
    epoch = 10
    timesInOneEpoch = 1000
    trainingTimes = 0
    for e in range(epoch) :
        for ti in range(timesInOneEpoch) :
            with tf.GradientTape() as tape :
                logits = Model((testInput,2),True,(mask0Test,mask1Test))
                losses = loss((logits,testLabels)) + \
                    tf.add_n([tf.multiply(0.0001,tf.nn.l2_loss(varis))  for varis in Model.trainable_weights])
                gradients = tape.gradient(losses,Model.trainable_weights)
            if trainingTimes % 50 == 0:
                if trainingTimes != 0 :
                    config = opti.get_config()
                    config["learning_rate"] = learningRate * 0.99
                    opti = opti.from_config(config)
                    print("Config LR : ",opti.get_config()["learning_rate"])
                print("Times : ",trainingTimes)
                print("Logits : ",Model((testInput,2),False,(mask0Test,mask1Test)))
                print("Losses : ",losses)
            opti.apply_gradients(zip(gradients,Model.trainable_weights))
            trainingTimes += 1



