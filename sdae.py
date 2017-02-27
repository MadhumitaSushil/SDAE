'''
@author madhumita
'''

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import backend as K

import nn_utils
import numpy as np

class StackedDenoisingAE(object):
    '''
    Implements stacked denoising autoencoders in Keras, without tied weights.
    To read up about the stacked denoising autoencoder, check the following paper:
    
    Vincent, Pascal, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol. 
    "Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion." 
    Journal of Machine Learning Research 11, no. Dec (2010): 3371-3408.
    '''
    def __init__(self, n_layers = 3, n_hid = [500], dropout = [0.05], enc_act = ['sigmoid'], dec_act = ['linear'], bias = False, loss_fn = 'mse', batch_size = 32, nb_epoch = 10, optimizer = 'rmsprop', verbose = 1):
        '''
        Initializes parameters for stacked denoising autoencoders
        @param n_layers: number of layers, i.e., number of autoencoders to stack on top of each other.
        @param n_hid: list with the number of hidden nodes per layer. If only one value specified, same value is used for all the layers
        @param dropout: list with the proportion of input nodes to mask at each layer. If only one value is provided, all the layers share the value.
        @param enc_act: list with activation function for encoders at each layer. Typically sigmoid.
        @param dec_act: list with activation function for decoders at each layer. Typically the same as encoder for binary input, linear for real input.
        @param bias: True to use bias value.
        @param loss_fn: The loss function. Typically 'mse' is used for real values. Options can be found here: https://keras.io/objectives/ 
        @param batch_size: mini batch size for gradient update
        @param nb_epoch: number of epochs to train each layer for
        @param optimizer: The optimizer to use. Options can be found here: https://keras.io/optimizers/
        '''
        self.n_layers = n_layers
        
        #if only one value specified for n_hid, dropout, enc_act or dec_act, use the same parameters for all layers.
        self.n_hid, self.dropout, self.enc_act, self.dec_act = self._assert_input(n_layers, n_hid, dropout, enc_act, dec_act)
        
        self.bias = bias
        
        self.loss_fn = loss_fn
        
        self.batch_size = batch_size
        
        self.nb_epoch = nb_epoch
        
        self.optimizer = optimizer
        
        self.verbose = verbose
        
        
    def get_pretrained_sda(self, data_in, data_val, data_test, shuffle = False, seed = 1337):
        '''
        Pretrains layers of a stacked denoising autoencoder to generate low-dimensional representation of data.
        Returns a model with pretrained encoding layers. Additionally, returns dense representation of input, validation and test data. 
        This dense representation is the value of the hidden node of the last layer.
        The model be used in supervised task by adding a classification/regression layer on top, or the dense pretrained data can be used as input of another model.
        @param data_in: input data (scipy sparse matrix supported)
        @param data_val: validation data (scipy sparse matrix supported)
        @param data_test: test data (scipy sparse matrix supported)
        @param shuffle: True to shuffle data before training model
        @param seed: seed for random shuffling
        '''
        
        encoders = []
        
        for cur_layer in range(self.n_layers):
            
            model = Sequential()
            
            # masking input data to learn to generalize, and prevent identity learning
            in_dropout = Dropout(self.dropout[cur_layer], input_shape=(data_in.shape[1],))
            model.add(in_dropout)
    
            encoder = Dense(output_dim = self.n_hid[cur_layer], init = 'glorot_uniform', activation = self.enc_act[cur_layer], name = 'encoder'+str(cur_layer), bias = self.bias)
            model.add(encoder)
            
            n_out = data_in.shape[1] #same no. of output units as input units (to reconstruct the signal)
            
            decoder = Dense(output_dim = n_out, bias = self.bias, init = 'glorot_uniform', activation = self.dec_act[cur_layer], name = 'decoder'+str(cur_layer))
            model.add(decoder)
            
            model.compile(loss = self.loss_fn, optimizer=self.optimizer)
            
            print("Training layer "+ str(cur_layer))
            
            #Early stopping to stop training when val loss increses for 1 epoch
            early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0)
            
            model.fit_generator(generator = nn_utils.batch_generator(data_in, data_in, batch_size = self.batch_size, shuffle = shuffle, seed = seed), callbacks = [early_stopping], nb_epoch=self.nb_epoch, samples_per_epoch = data_in.shape[0], verbose=self.verbose, validation_data  = nn_utils.batch_generator(data_val, data_val, batch_size = self.batch_size, shuffle = shuffle, seed = seed), nb_val_samples = data_val.shape[0])
            print("Layer "+ str(cur_layer) +" has been trained")
            
            encoders.append(encoder)
        
            data_in = self._get_intermediate_output(model, data_in, n_layer = 1, train = 0, n_out = self.n_hid[cur_layer], batch_size = self.batch_size, shuffle, seed) #train = 0 because we do not want to use dropout to get hidden node value, since is a train-only behavior, used only to learn weights. output of first layer: hidden layer
    #         assert data_in.shape[1] == n_hid[cur_layer], "Output of hidden layer not retrieved"
            
            data_val = self._get_intermediate_output(model, data_val, n_layer = 1, train = 0, n_out = self.n_hid[cur_layer], batch_size = self.batch_size, shuffle, seed) #get output of first layer (hidden layer) without dropout
            data_test = self._get_intermediate_output(model, data_test, n_layer = 1, train = 0, n_out = self.n_hid[cur_layer], batch_size = self.batch_size, shuffle, seed)

        pretrained_model = self._build_model_from_encoders(encoders)      
        return pretrained_model, (data_in, data_val, data_test)
        
        
    def _assert_input(self, n_layers, n_hid, dropout, enc_act, dec_act):
        '''
        If the hidden nodes, dropou proportion, encoder activation function or decoder activation function is given, it uses the same parameter for all the layers.
        Errors out if there is a size mismatch between number of layers and parameters for each layer.
        '''
        
        if len(n_hid) == 1:
            n_hid = n_hid * n_layers
            
        if len(dropout) == 1:
            dropout = dropout *n_layers
        
        if len(enc_act) == 1:
            enc_act = enc_act * n_layers
        
        if len(dec_act) == 1:
            dec_act = dec_act * n_layers
                
        assert (n_layers == len(n_hid) == len(dropout) == len(enc_act) == len(dec_act)), "Please specify as many hidden nodes, dropout proportion on input, and encoder and decoder activation function, as many layers are there, using list data structure"
          
        return n_hid, dropout, enc_act, dec_act
    
    def _get_intermediate_output(self, model, data_in, n_layer, train, n_out, batch_size, shuffle, seed):
        '''
        Returns output of a given intermediate layer in a model
        @param model: model to get output from
        @param data_in: sparse representation of input data
        @param n_layer: the layer number for which output is required
        @param train: (0/1) 1 to use training config, like dropout noise.
        @param n_out: number of output nodes in the given layer (pre-specify so as to use generator function with sparse matrix to get layer output)
        @param batch_size: the num of instances to convert to dense at a time
        @param shuffle: True to shuffle data
        @param seed: seed for replicability of shuffled data
        @return value of intermediate layer
        '''
        data_out = np.zeros(shape = (data_in.shape[0],n_out))
        
        x_batch_gen = nn_utils.x_generator(data_in, batch_size = batch_size, shuffle = shuffle, seed = seed)
        stop_iter = int(np.ceil(data_in.shape[0]/batch_size))
        
        for i in range(stop_iter):
            cur_batch, cur_batch_idx = next(x_batch_gen)
            data_out[cur_batch_idx,:] = self._get_nth_layer_output(model, n_layer, X =  cur_batch, train = train)
        
        return data_out
            
    def _get_nth_layer_output(self, model, n_layer, X, train = 1):
        '''
        Returns output of nth layer in a given model.
        @param model: keras model to get an intermediate value out of
        @param n_layer: the layer number to get the value of
        @param X: input data for which layer value should be computed and returned.
        @param train: (1/0): 1 to use the same setting as training (for example, with Dropout, etc.), 0 to use the same setting as testing phase for the model.
        @return the value of n_layer in the given model, input, and setting 
        '''
        get_nth_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                      [model.layers[n_layer].output])
        return get_nth_layer_output([X,train])[0]
    
    def _build_model_from_encoders(self, encoding_layers):
        '''
        Builds a deep NN model that generates low-dimensional representation of input, based on pretrained layers.
        @param encoding_layers: pretrained encoder layers
        @return model with each encoding layer as a layer of a NN
        '''
        model = Sequential()
        for i in range(len(encoding_layers)):
            model.add(encoding_layers[i]) #trained weights are linked to the layer
    
        return model
