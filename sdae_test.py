from sdae import StackedDenoisingAE

from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

from matplotlib import pyplot as plt

n_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = to_categorical(y_train, n_classes)
Y_test = to_categorical(y_test, n_classes)

X_train_50 = X_train[0:50]

n_in = n_out = X_train.shape[1];
n_hid = 576


cur_sdae = StackedDenoisingAE(n_layers = 3, n_hid = [400], dropout = [0.1], nb_epoch = 20, seed = 1337)

model, (dense_train, dense_val, dense_test), train_recon = cur_sdae.get_pretrained_sda(X_train, X_test, X_test, n_classes = n_classes, dir_out = '../output')

X_train_50 *= 255
output_dense = dense_train[0:50,:]
output_dense *= 255

output_recon = train_recon[0:50,:]
output_recon *= 255

plt.figure(figsize=(10, 10))
for i in range(50):
    ax1 = plt.subplot(10, 10, i * 2 + 1)
    ax1.imshow(X_train_50[i].reshape((28, 28)), interpolation='nearest')
    
    ax2 = plt.subplot(10, 10, i * 2 + 2)
    ax2.imshow(output_dense[i].reshape((20, 20)), interpolation='nearest')
plt.savefig('../output/sdae_test_mnist_dense.png')

plt.figure(figsize=(10, 10))
for i in range(50):
    ax1 = plt.subplot(10, 10, i * 2 + 1)
    ax1.imshow(X_train_50[i].reshape((28, 28)), interpolation='nearest')
    
    ax2 = plt.subplot(10, 10, i * 2 + 2)
    ax2.imshow(output_recon[i].reshape((28, 28)), interpolation='nearest')
plt.savefig('../output/sdae_test_mnist_recon.png')

# fit_classifier = cur_sdae.supervised_classification(model=model, x_train=X_train, x_val=X_test, y_train=Y_train, y_val=Y_test, n_classes=n_classes)
# pred = cur_sdae.predict(fit_classifier, X_test, 1337)
# score, conf_matrix, error_idx = model_utils.score(y_true, y_pred, y_pred_score, cfg, n_classes)