import tensorflow as tf
import numpy as np
import Transformer as tsf
from tensorflow import keras
import TransformerXL as tsfXL


class QueryStreamTransformerXLEncoder (keras.Model) :
    """
    In the call function,
    ### The inputs consist of double component.
    ### The first is a sentence , the shape of it is [b , maxTimes , wordEmbedding]
    ### The second is a G_Matrix. The shape is [b , maxTimes , wordEmbedding], but
    ### it is an trainable weight.
    """

    def __init__(self,batchSize,sLength,wordEmbeddingSize,
                 numberOfTransformerLayers,selfAttentionSize,interMediumDim):
        super(QueryStreamTransformerXLEncoder,self).__init__()
        self.L = numberOfTransformerLayers
        self.B = batchSize
        self.tsfEncoderList = [tsf.TransformerEncoder(h=selfAttentionSize,
                                                      interMediumDim=interMediumDim,
                                                      dk=wordEmbeddingSize) for _ in range(numberOfTransformerLayers)]
        self.cacheMatrix = [tf.zeros(shape=[sLength,wordEmbeddingSize],dtype=tf.float32)
                            for _ in range(numberOfTransformerLayers)]
        self.WQ = [keras.layers.Dense(wordEmbeddingSize) for _ in range(numberOfTransformerLayers)]
        self.WK = [keras.layers.Dense(wordEmbeddingSize) for _ in range(numberOfTransformerLayers)]
        self.WV = [keras.layers.Dense(wordEmbeddingSize) for _ in range(numberOfTransformerLayers)]


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
    """
    In the call function,
    ### The inputs consist of four components.
    ### The first is a sentence , the shape of it is [b , maxTimes , wordEmbedding]
    ### The second is a G_Matrix. The shape is [b , maxTimes , wordEmbedding], but
    ### it is an trainable weight.
    ### The third is position that needs to predict.
    ### The four is if the model is in fine-tuning.
    ##################################################
    #### Mask is composed by mask0 and mask1.        #
    #### The mask0 is for query stream attention     #
    #### The mask1 is for content stream attention   #
    ##################################################
    """

    def __init__(self,batchSize,sLength,wordEmbeddingSize,
                 numberOfTransformerLayers,selfAttentionSize,interMediumDim):
        self.B = batchSize
        self.T = sLength
        self.W = wordEmbeddingSize
        super(MaskedTwoStreamAttention,self).__init__()
        self.queryAttention = QueryStreamTransformerXLEncoder(batchSize,sLength,wordEmbeddingSize,
                                                              numberOfTransformerLayers,selfAttentionSize,interMediumDim)
        self.contentAttention = tsfXL.TransformerXLEncoder(sLength,numberOfTransformerLayers,
                                                           selfAttentionSize,interMediumDim,wordEmbeddingSize)

    ### The inputs consist of four components.
    ### The first is a sentence , the shape of it is [b , maxTimes , wordEmbedding]
    ### The second is a G_Matrix. The shape is [b , maxTimes , wordEmbedding], but
    ### it is an trainable weight.
    ### The third is position that needs to predict.
    ### The four is if the model is in fine-tuning.
    ##################################################
    #### Mask is composed by mask0 and mask1.        #
    #### The mask0 is for query stream attention     #
    #### The mask1 is for content stream attention   #
    ##################################################
    def call(self, inputs,training=None, mask=None):
        inputsWordEmbedding, G_Matrix , predictPosition , ifFinetuning = inputs
        if mask is not None:
            mask0, mask1 = mask
        else:
            mask0 = mask1 = None
        if ifFinetuning :
            paddingMaskGMatrix = tf.ones(shape=[self.B, self.T, self.W],dtype=tf.float32)
            paddingMaskInput = tf.ones(shape=[self.B, self.T, self.W],dtype=tf.float32)
        else:
            paddingMaskGMatrix = np.zeros(shape=[self.B, self.T, self.W])
            paddingMaskGMatrix[:, predictPosition, :] = [1 for _ in range(self.W)]
            paddingMaskGMatrix = tf.convert_to_tensor(paddingMaskGMatrix, dtype=tf.float32)
            paddingMaskInput = np.ones(shape=[self.B, self.T, self.W])
            paddingMaskInput[:, predictPosition, :] = [0 for _ in range(self.W)]
            paddingMaskInput = tf.convert_to_tensor(paddingMaskInput, dtype=tf.float32)
        gStates = self.queryAttention((tf.multiply(inputsWordEmbedding,paddingMaskInput),
                                       tf.multiply(G_Matrix,paddingMaskGMatrix)),training,mask0)
        hStates = self.contentAttention(inputsWordEmbedding,training,mask1)
        return hStates , gStates


class XLNetEncoder (keras.Model) :
    """
    In the call fucntion,
    ### the inputs are composed by batch of sentences, a predictPosition and a ifFinetuning parameter.
    ### the shape of batch of sentences is [b , maxTimes , wordEmbedding]
    ### the return is a hiddenTensor which shape is [b , maxTimes, wordEmbedding]
    """

    def __init__(self,batchSize,sLength,wordEmbeddingSize,
                 numberOfTransformerLayers,selfAttentionSize,interMediumDim,
                 numberOfMaskTwoStreamLayer):
        super(XLNetEncoder,self).__init__()
        self.L = numberOfMaskTwoStreamLayer
        self.G_Matrix = self.add_weight("G_Matrix", shape=[batchSize,sLength,wordEmbeddingSize],dtype=tf.float32,
                                        initializer=keras.initializers.glorot_normal())
        self.maskedTwoStreamList = [MaskedTwoStreamAttention(batchSize,sLength,wordEmbeddingSize,
                                                            numberOfTransformerLayers,selfAttentionSize,interMediumDim)
                                   for _ in range(numberOfMaskTwoStreamLayer)]

    ### the inputs are composed by batch of sentences, a predictPosition and a ifFinetuning parameter.
    ### the shape of inputs is [b , maxTimes , wordEmbedding]
    ### the return is a hiddenTensor which shape is [b , maxTimes, wordEmbedding]
    def call(self, inputs, training=None, mask=None):
        inputEmbedding , predictPosition , ifFinetuning = inputs
        hStates, gStates = self.maskedTwoStreamList[0](
            (inputEmbedding, self.G_Matrix, predictPosition, ifFinetuning), training, mask)
        for i in range(self.L - 1):
            hStates, gStates = self.maskedTwoStreamList[i+1](
                (hStates, gStates, predictPosition, ifFinetuning), training, mask)
        return gStates

class XLNetPreTrain(keras.Model) :
    """
    In the call function
    ### the inputs are composed by batch of sentences, a predictPosition and a ifFinetuning parameter.
    ### the shape of batch of sentences is [b , maxTimes , wordEmbedding]
    """

    def __init__(self,batchSize,sLength,wordEmbeddingSize,
                 numberOfTransformerLayers,selfAttentionSize,interMediumDim,
                 numberOfMaskTwoStreamLayer,
                 outDim):
        super(XLNetPreTrain,self).__init__()
        self.encoder = XLNetEncoder(batchSize,sLength,wordEmbeddingSize,
                                    numberOfTransformerLayers,selfAttentionSize,interMediumDim,numberOfMaskTwoStreamLayer)
        self.dense = keras.layers.Dense(outDim)

    ### the inputs are composed by batch of sentences, a predictPosition and a ifFinetuning parameter.
    ### the shape of batch of sentences is [b , maxTimes , wordEmbedding]
    def call(self, inputs, training=None, mask=None):
        encoderState = self.encoder(inputs,training,mask)
        flatten = tf.reshape(encoderState, shape=[-1, encoderState.shape[1] * encoderState.shape[2]])
        decoder = self.dense(flatten)
        return tf.nn.softmax(decoder)

class XLNet(keras.Model) :
    """
    In the call fucntion,
    ### the inputs are composed by batch of sentences , a translation sentence and a ifFinetuning parameter.
    ### the shape of inputs is [b , maxTimes , wordEmbedding]
    ### This mask only has one mask which maintains the auto regression attribute.
    """

    def __init__(self,batchSize,sLength,wordEmbeddingSize,
                 numberOfTransformerLayers,selfAttentionSize,interMediumDim,
                 outDim,numberOfMaskTwoStreamLayer,numberOfDecoder):
        super(XLNet,self).__init__()
        self.dn = numberOfDecoder
        self.encoder = XLNetEncoder(batchSize,sLength,wordEmbeddingSize,
                                    numberOfTransformerLayers,selfAttentionSize,interMediumDim,
                                    numberOfMaskTwoStreamLayer)
        self.decoders = [tsf.TransformerDecoder(selfAttentionSize,interMediumDim,wordEmbeddingSize) for _ in range(numberOfDecoder)]
        self.dense = keras.layers.Dense(outDim)

    ### the inputs are composed by batch of sentences , a translation sentence and a ifFinetuning parameter.
    ### the shape of inputs is [b , maxTimes , wordEmbedding]
    ### This mask only has one mask which maintains the auto regression attribute.
    def call(self, inputs, training=None, mask=None):
        inputEmbedding , outputEmbedding ,ifFinetuning = inputs
        encoderState = self.encoder((inputEmbedding,0 , ifFinetuning),training,None)
        x = self.decoders[0]((outputEmbedding,encoderState),training,mask)
        for i in range(self.dn - 1):
            x = self.decoders[i + 1]((x,encoderState),training,mask)
        flatten = tf.reshape(x,shape=[-1,x.shape[1] * x.shape[2]])
        lx = self.dense(flatten)
        return tf.nn.softmax(lx)


class LossDevice(keras.Model) :

    def __init__(self):
        super(LossDevice,self).__init__()

    def call(self, inputs, training=None, mask=None):
        logs , label = inputs
        return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(label,logs,
                                                                       tf.ones(logs.shape,dtype=tf.float32)))

if __name__ == "__main__":
    # testInput = np.array(np.random.randn(3,4,5),dtype=np.float32)
    # testTranslation = np.array(np.random.randn(3,4,5),dtype=np.float32)
    testLabels = np.array([[1,0,0,0],
                           [0,0,0,1],
                           [0,0,1,0]],dtype=np.float32)
    ### This mask is for query stream attention,it can not see itself.
    mask0Test = tf.convert_to_tensor(np.array([[-1e10,0,-1e10,-1e10],
                                               [0,-1e10,-1e10,-1e10],
                                               [-1e10,-1e10,-1e10,-1e10],
                                               [-1e10,-1e10,-1e10,-1e10]]),dtype=tf.float32)
    ### This mask is for content stream attention, it can see itself .
    mask1Test = tf.convert_to_tensor(np.array([[0,0,-1e10,-1e10],
                                               [0,0,-1e10,-1e10],
                                               [-1e10,-1e10,0,-1e10],
                                               [-1e10,-1e10,-1e10,0]]),dtype=tf.float32)
    ### This mask is for fine-turing .
    mask2Test = tf.convert_to_tensor(np.array([[[0,0,0,-1e10],
                                               [0,0,0,-1e10],
                                               [0,0,0,-1e10],
                                               [-1e10,-1e10,-1e10,-1e10]] for _ in range(3)]),dtype=tf.float32)
    # Model = XLNetPreTrain(batchSize=3,sLength=4,wordEmbeddingSize=5,
    #               numberOfTransformerLayers=3,selfAttentionSize=4,interMediumDim=8,
    #               outDim=4,numberOfMaskTwoStreamLayer=3)
    Model = XLNet(batchSize=3,sLength=4,wordEmbeddingSize=5,
                  numberOfTransformerLayers=6,selfAttentionSize=8,interMediumDim=18,
                  outDim=4,numberOfMaskTwoStreamLayer=7,numberOfDecoder=9)
    loss = LossDevice()
    learningRate = 0.01
    optiA = tf.optimizers.Adam(learningRate,epsilon=1e-5,amsgrad=True)
    optiSGD = tf.optimizers.SGD(learningRate,momentum=0.96,nesterov=True)
    # print("Load weights . ")
    # Model.load_weights("d:\\XLNet")
    # print("Load weights completed .")
    epoch = 10
    timesInOneEpoch = 1000
    trainingTimes = 0
    for e in range(epoch) :
        for ti in range(timesInOneEpoch) :
            with tf.GradientTape() as tape :
                # logits = Model((testInput,2,False),True,(mask0Test,mask1Test))
                logits = Model((np.array(np.random.randn(3,4,5),dtype=np.float32),
                                np.array(np.random.randn(3, 4, 5), dtype=np.float32),True),True,mask2Test)
                losses = loss((logits,testLabels)) + \
                    tf.add_n([tf.multiply(0.0001,tf.nn.l2_loss(varis))  for varis in Model.trainable_weights])
                gradients = tape.gradient(losses,Model.trainable_weights)
            if trainingTimes % 50 == 0:
                print("Config LR : ", learningRate)
                print("Times : ",trainingTimes)
                # print("Logits : ",Model((testInput,2,False),False,(mask0Test,mask1Test)))
                print("Logits : ",Model((np.array(np.random.randn(3,4,5),dtype=np.float32),
                                         np.array(np.random.randn(3, 4, 5), dtype=np.float32),True),False,mask2Test))
                print("Losses : ",losses)
            if trainingTimes <= epoch * timesInOneEpoch // 2 :
                if trainingTimes % 700 == 0 and trainingTimes != 0:
                    learningRate = learningRate * 0.95
                    config = optiA.get_config()
                    config["learning_rate"] = learningRate
                    optiA = optiA.from_config(config)
                optiA.apply_gradients(zip(gradients, Model.trainable_weights))
            else:
                if trainingTimes % 800 == 0 and trainingTimes != 0:
                    learningRate = learningRate * 0.95
                    config = optiSGD.get_config()
                    config["learning_rate"] = learningRate
                    optiSGD = optiSGD.from_config(config)
                optiSGD.apply_gradients(zip(gradients, Model.trainable_weights))
            if trainingTimes % 1000 == 0 :
                print("Saving....")
                Model.save_weights("d:\\XLNet")
                print("Saving completed.")
            trainingTimes += 1



