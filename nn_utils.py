from keras.models import model_from_json
from keras.utils.visualize_util import plot
import numpy as np
import scipy.sparse as scp

def batch_generator(X, Y, batch_size, shuffle, seed = 1337):
    '''
    Creates batches of data from given dataset, given a batch size. Returns dense representation of sparse input.
    @param X: input features, sparse or dense
    @param Y: input labels, sparse or dense
    @param batch_size: number of instances in each batch
    @param shuffle: If True, shuffle input instances.
    @param seed: fixed seed for shuffling data, for replication
    @return batch of input features and labels
    '''
    number_of_batches = np.ceil(X.shape[0]/batch_size) #ceil function allows for creating last batch off remaining samples
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(sample_index)
    
    sparse = False
    if scp.issparse(X):
        sparse = True
    
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        if sparse:
            x_batch = X[batch_index,:].toarray() #converts to dense array
            y_batch = Y[batch_index,:].toarray() #converts to dense array
        else:
            x_batch = X[batch_index,:]
            y_batch = Y[batch_index,:]
        counter += 1
        yield x_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def x_generator(X, batch_size, shuffle, seed = 1337):
    '''
    Creates batches of data from given input, given a batch size. Returns dense representation of sparse input one batch a time.
    @param X: input features, can be sparse or dense
    @param batch_size: number of instances in each batch
    @param shuffle: If True, shuffle input instances.
    @param seed: fixed seed for shuffling data, for replication
    @return batch of input data, without shuffling
    '''
    number_of_batches = np.ceil(X.shape[0]/batch_size) #ceil function allows for creating last batch off remaining samples
    counter = 0
    sample_index = np.arange(X.shape[0])
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(sample_index)
    
    sparse = False
    if scp.issparse(X):
        sparse = True
        
    while counter < number_of_batches: 
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        if sparse:
            x_batch = X[batch_index,:].toarray() #converts to dense array
        else:
            x_batch = X[batch_index,:]
        yield x_batch, batch_index
        counter += 1

def save_model(model, out_dir, f_arch = 'model_arch.png', f_model = 'model_arch.json', f_weights = 'model_weights.h5'):
    '''
    Saves a Keras model description and model weights
    @param model: a keras model
    @param out_dir: directory to save model architecture and weights to
    @param f_model: file name for model architecture
    @param f_weights: filename for model weights
    '''
    model.summary()
    plot(model, to_file=out_dir+f_arch)
    
    json_string = model.to_json()
    open(out_dir+f_model, 'w').write(json_string)
    
    model.save_weights(out_dir+f_weights, overwrite=True)
    
def load_model(dir_name, f_model = 'model_arch.json', f_weights = 'model_weights.h5' ):
    '''
    Loads a Keras model from disk to memory.
    @param dir_name: directory in which the model architecture and weight files are present
    @param f_model: file name for model architecture
    @param f_weights: filename for model weights
    @return loaded model
    '''
    json_string = open(dir_name + f_model, 'r').read()
    model = model_from_json(json_string)
    
    model.load_weights(f_weights)
    
    return model