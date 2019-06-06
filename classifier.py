# Imports
from parameters import *

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles


###  Dataset  ###
x_train, y_train_raw = make_blobs(n_samples = n_training_samples, 
	n_features = n_features,
	centers = n_classes,
	center_box = center_box,
	cluster_std =  cluster_std, 
	random_state = seed)

# one-hot encodes the y values (categorically encoding the data)
y_train = np.zeros((y_train_raw.shape[0], n_classes))
y_train[np.arange(y_train_raw.size), y_train_raw] = 1

# scatterplot of the training data
plt.scatter(x_train[:, 0], x_train[:, 1], c = y_train_raw, cmap = colors)
plt.draw()
plt.savefig('{}{}'.format(path_plots, 'train_set.png'))
plt.close()


###  Train and Validation Sets  ###
percent_val  = 0.2
percent_index = int(0.2 * int(x_train.shape[0]))

# creates validation and training features
x_val = x_train[-percent_index:]
x_train = x_train[:-percent_index]

# creates validation and training labels
y_val =  y_train[-percent_index:]
y_train = y_train[:-percent_index]

# asserts that the new number of samples of the labels and targets for val/train sets are the same
assert(x_train.shape[0] == y_train.shape[0])
assert(x_val.shape[0] == y_val.shape[0])



###  Neural Network  Model  ###
model = Sequential()

# Model layers
model.add(Dense(l1, input_shape = x_train.shape[1:], activation = 'relu')) # hidden layer 1
model.add(Dropout(0.5))
model.add(Dense(l2, activation = 'relu')) # hidden layer 2
model.add(Dropout(0.5))
model.add(Dense(l3, activation = 'relu')) # hidden layer 3
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation = 'sigmoid')) # output layer

# Prints a summary of the model shape/architecture
print("| Model Summary |")
model.summary()
print()



###  Compile and Train Model  ###
print("| Model Training |\n")

# Creates callback for training that saves the model weights after each epoch
checkpointer = ModelCheckpoint(filepath = 'saved_models/weights-{epoch:02d}.hdf5')
callbacks_list = [checkpointer]

# Compiles the model
model.compile(loss = loss_op, 
	optimizer = Adam(learn_rate),
	metrics = ['accuracy'])

# Trains the model
history = model.fit(x_train, 
	y_train, 
	batch_size = batch_size, 
	epochs = n_epochs, 
	validation_data = (x_val, y_val), 
	verbose = 2,
	callbacks = callbacks_list)

# Plots the train and validation loss
loss_data = history.history['loss']
val_loss_data = history.history['val_loss']
plt.plot(loss_data)
plt.plot(val_loss_data)
plt.legend(['loss', 'val_loss'])
plt.draw()
plt.savefig('{}{}'.format(path_plots, 'train_loss.png'))
plt.close()

# Saves the model
model.save('{}{}'.format(path_models, NAME))
print("\nModel succsefully trained and saved, done executing classifier.py\n")