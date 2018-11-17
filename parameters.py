# Imports
from matplotlib.colors import ListedColormap

###  Hyperparameters   ##
NAME = 'classifier_model.{}'.format('h5') # name of the saved model
seed = 422 # Seed for the generated data (used 422 for git repo)

n_features = 2 # number of input variables (must be  2 for accurate plotting in matplotlib)
l1 = 32 # first hidden layer nodes
l2 = 32 # second hidden layer nodes
l3 = 16 # third hidden layer nodes
n_classes = 4 # output nodes (the number of different classes) max of 6, if you want more you will have to change the cmap colors

n_training_samples = 100 # number of training samples (num training points)
n_testing_samples = 200 # number of testing samples (num training points)
cluster_std = 0.7 # the standard deviation of the data points from the center of the blob clusters
center_box = (-10, 10) # try not to change this as the plots will not look right
noise = 0.10 # for make_moons

colors_list = ['red', 'cyan', 'magenta', 'green', 'black', 'blue']
colors = ListedColormap(colors_list)

n_epochs = 30 # number of training iterations on the training set
batch_size = 1 # should be kept at 1
learn_rate = 0.001 # rate at which the algorithim updates the weights, try increasing or decreasing by factors of 10

points_per_int = 3 # num of points in an int
alpha = 0.1 # transparency of area map points

loss_op = 'mean_squared_error'

###  Paths  ###
path_plots = 'figs/'
path_models = 'saved_models/'
path_networks = 'networks/'