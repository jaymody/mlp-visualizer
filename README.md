#Scikit-Learn Cluster Classifier
##Creator: Jay Mody

Using scikit API make_blob and keras, I created distinct number of clusters (number of classes) of data that I trained using a simple fully connected multilayer-perceptron network. The number of clusters is arbitrary, I chose 4 because 2 is a simple example and more than 4 creates data with moreoverlap.

All the variable paramaters (except the network  optimizer) can be edited in parameters.py

**HOW TO RUN PROGRAM**
**1. Using anaconda, import the environment from environment.yml**
**2. Run classifier.py**
**3. Run tester.py**

After running the program, new plots should be generated under figs/
The trained model should be saved to saved_models/

######Training Data
The training data consists of 4 distinct classes, made up from 100 data points.
![Alt text](figs/train_set.pngraw=true "Training Set")