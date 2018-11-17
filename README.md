# Scikit-Learn Cluster Classifier
## Creator: Jay Mody

Using scikit API make_blob and keras, I created distinct number of clusters (number of classes) of data that I trained using a simple fully connected multilayer-perceptron network. 

**HOW TO RUN PROGRAM**

**1. Using anaconda, import the environment from environment.yml**

**2. Make any desired changes to parameters.py**

**3. Run classifier.py**

**4. Run tester.py**

**5. Check out the figs/ folder for results**

After running the program, new plots should be generated under figs/
The trained model should be saved to saved_models/
A gif of the networks predictions at the various epochs can also be found in figs/ but will also be saved to networks/

Below are the results using my data and network parameters.



## Model Summary
The model is a simple mlp with 3 hidden layers, all using the relu activation function (output layer uses sigmoid). The summary can be found below:

| Model Summary |

Layer (type) | Output Shape | Param #
--- | --- | ---
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
Here is a plot of the training and validation loss dring the training process. The data can be also found under saved_models/old_log.txt.

![Training](/figs/train_loss.png)



## Testing Data
The test data is similar to the training data, however there is more data (200 points). When the model evaluates the training data, the labels should correspond to this graph.

![Test Set](/figs/test_set.png)


## Predictions
Here is a plot of the models predictions over time:

![Predictions GIF](/figs/neural_network_current.gif)

The final result:

![Final Result](/figs/result.png)



As you can see, the model predicted almost all of the samples correctly, except for the two points that are in between the blue class and black class.

Overall, the network had an accuracy of **99.0%**



## Parameters
Chaning n_features is not advised, as this would make the plots innacurate since they dont show the additional dimensions.

Changing n_classes will change the number of clusters. Numbers more than 4 may run into the problem of overlap in the clusters.

The seed is randomly generated, however you can easily change it to be any number you want if you want to work with the same dataset every time.

The number of nodes for each of the 3 layers are also editable.

For a full list of the editable paramaters, make sure you check out parameters.py


## Other Data Examples

![N](/networks/neural_network1542482492.070956.gif)


![N](/networks/neural_network1542424062.898062.gif)


![N](/networks/neural_network1542479334.236445.gif)


![N](/networks/neural_network1542480257.222606.gif)


![N](/networks/neural_network1542426226.878786.gif)


![N](/networks/neural_network1542426470.394302.gif)


![N](/networks/neural_network1542482833.081877.gif)


![N](/networks/neural_network1542426899.121683.gif)