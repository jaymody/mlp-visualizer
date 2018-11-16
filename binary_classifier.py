import numpy as np
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

###  Hyperparameters  ###
NAME = 'Binary_Classifier{}.{}'.format(time.time(), 'keras')

n_features = 2 # number of input variables (2, for 2d plotting in matplotlib)
l1 = 32
l2 = 32
l3 = 16
n_classes = 4 # output nodes

n_samples = 100
cluster_std = 0.7

seed1 = 202
seed2 = 422

n_epochs = 20
batch_size = 1
learn_rate = 0.001


###  Dataset  ###
x_train, y_train_raw = make_blobs(n_samples = n_samples, 
	n_features = n_features, 
	centers = n_classes, 
	cluster_std = cluster_std, 
	random_state = seed2)

#one-hot encodes the y values (categorically encoding the data)
y_train = np.zeros((y_train_raw.shape[0], n_classes))
y_train[np.arange(y_train_raw.size), y_train_raw] = 1

# print("...data succesfully created... \n")

plt.scatter(x_train[:, 0], x_train[:, 1], c = y_train_raw, cmap = 'coolwarm')
plt.show()


###  Train and Validation Sets  ###
percent_val  = 0.2
percent_index = int(0.2 * int(x_train.shape[0]))

x_val = x_train[-percent_index:]
x_train = x_train[:-percent_index]

y_val =  y_train[-percent_index:]
y_train = y_train[:-percent_index]

assert(x_train.shape[0] == y_train.shape[0])
assert(x_val.shape[0] == y_val.shape[0])

# print("...training and validation sets created... ")
# print('{} data length: {}'.format('Training', len(x_train)))
# print('{} data length: {}'.format('Validation', len(x_val)))
# print()


###  Neural Network  ###
model = Sequential()

model.add(Dense(l1, input_shape = x_train.shape[1:], activation = 'relu'))
model.add(Dense(l2, activation = 'relu'))
model.add(Dense(l3, activation = 'relu'))
model.add(Dense(n_classes, activation = 'sigmoid'))

# print("...model succesfully created...")
model.summary()
# print()


###  Compile and Train Model  ###
model.compile(loss = binary_crossentropy, optimizer = Adam(learn_rate), metrics = ['accuracy'])
# print('...model succesfully compiled...\n')

model.fit(x_train, 
	y_train, 
	batch_size = batch_size, 
	epochs = n_epochs, 
	validation_data = (x_val, y_val), 
	verbose = 2)
# print('...model succesfully trained...\n')

model.save(NAME)
# print('...model succesfully saved as "' + NAME + '"...')


###  Testing Model  ###
x_test, y_test = make_blobs(n_samples = 200, 
	n_features = n_features, 
	centers = n_classes, 
	cluster_std = cluster_std, 
	random_state = seed2)

predictions = model.predict(x_test)
predictions = [predictions[i,:].argmax() for i in range(int(predictions.shape[0]))]

plt.scatter(x_test[:, 0], x_test[:, 1], c = predictions, cmap = 'coolwarm')
plt.show()
plt.scatter(x_test[:, 0], x_test[:, 1], c = y_test, cmap = 'coolwarm')
plt.show()

plt.savefig('fig.png')