# Scikit-Learn Cluster Classifier
## Creator: Jay Mody

Using scikit API make_blob and keras, I created distinct number of clusters (number of classes) of data that I trained using a simple fully connected multilayer-perceptron network. 

**HOW TO RUN PROGRAM**

**1. Using anaconda, import the environment from environment.yml**

**2. Run classifier.py**

**3. Run tester.py**

After running the program, new plots should be generated under figs/
The trained model should be saved to saved_models/

Below are the results using my data and network parameters.


## Model Summary
The model is a simple mlp with 3 hidden layers, all using the relu activation function (output layer uses sigmoid). The summary can be found below:

| Model Summary |

Layer (type) | Output Shape | Param #
--- | --- | ---
*Still* | `renders` | **nicely**
dense_1 (Dense) | (None, 32) | 96
dense_2 (Dense) | (None, 32) | 1056
dense_3 (Dense) | (None, 16) | 528
dense_4 (Dense) | (None, 4) | 68

Total params: 1,748

Trainable params: 1,748

Non-trainable params: 0


## Training Data
The training data consists of 4 distinct classes, made up from 100 data points.
![Train Set](/figs/train_set.png)


## Training Loss
Here is a plot of the training and validation loss dring the training process.
The data can be also found under saved_models/old_log.txt.
![Training](/figs/train_loss.png)


## Testing Data
The test data is similar to the training data, however there is more data (200 points). When the model evaluates the training data, the labels should correspond to this graph.
![Test Set](/figs/test_set.png)


Given that the above is the correct labels for the generated data, here is a plot of what the model predicted for the class of the data.
![Prediction Set](/figs/prediction_set.png)


As you can see, the model predicted almost all of the samples correctly, except for the two points that are in between the peach coloured class and red class.

Overall, the network had an accuracy of **99.0%**


## Parameters
Chaning n_features is not advised, as this would make the plots innacurate since they dont show the additional dimensions.

Changing n_classes will change the number of clusters. Numbers more than 4 may run into the problem of overlap in the clusters.

The seed is randomly generated, however you can easily change it to be any number you want if you want to work with the same dataset every time.

The number of nodes for each of the 3 layers are also editable.

For a full list of the editable paramaters, make sure you check out parameters.py
