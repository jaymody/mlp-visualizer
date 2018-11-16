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



## Testing Data
The test data is similar to the training data, however there is more data (200 points). When the model evaluates the training data, the labels should correspond to this graph.
![Test Set](/figs/test_set.png)


![Training](/figs/train_loss.png)


![Prediction Set](/figs/prediction_set.png)
