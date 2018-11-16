###  Hyperparameters   ##
NAME = 'classifier_model.{}'.format('h5') # name of the saved model
seed = 422 # Seed for the generated sklearn blobs

n_features = 2 # number of input variables (must be  2 for accurate plotting in matplotlib)
l1 = 32 # first hidden layer nodes
l2 = 32 # second hidden layer nodes
l3 = 16 # third hidden layer nodes
n_classes = 4 # output nodes (the number of different classes)

n_training_samples = 100 # number of training samples (num training points)
n_testing_samples = 200 # number of testing samples (num training points)
cluster_std = 0.7 # the standard deviation of the data points from the center of the blob clusters

n_epochs = 20 # number of training iterations on the training set
batch_size = 1
learn_rate = 0.001

loss_op = 'mean_squared_error'

###  Paths  ###
path_plots = 'figs/'
path_models = 'saved_models/'