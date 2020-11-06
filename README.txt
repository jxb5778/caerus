
# Justin Berg

# Caerus -> Caesar's Keras -> to help me conquer (learn about) neural networks and backpropagation.

# Repo Modules #
################

- caerus/
    - __init__.py    -> manages imports of classes and functions for the caerus module.
    - activations.py -> contains the activation functions that can be used in the caerus layers.
    - errors.py      -> contains the error functions that can be used in the caerus neural network.
    - layers.py      -> contains the layers than can be used in a caerus neural network.
    - metrics.py     -> contains functions for metrics that can be computed.
    - models.py      -> contains the the models that can be trained using caerus.
    - optimizers.py  -> contained the optimizers that can be used to train caerus neural networks.

- scripts/
    - run_mlp_model.py               -> script to train and evaluate a basic MLP model,
                                        using the data from the perceptron homework. (not used for the project)
    - run_sequential_model_simple.py -> script to train a simple Sequential NN. (not used for the project)
    - run_sequential_mnist.py        -> script to train simple sequential model on MNIST data (script for the project).

- workspace/ -> (not used for the project) scripts to help me understand functionality before incorporating into codebase
    - workspace_1.py -> (not used for the project)
    - workspace_2.py -> (not used for the project)



# Run the project script - run_sequential_mnist.py #
####################################################

The script used to compute the results cited in the project is scripts/run_sequential_mnist.py.
To run the script you'll need to make sure you install the 2 dependancies- numpy and mlxtend.
Numpy is used for matrix operations and mlxtend is used to read the MNIST data into memory.

Script Outline:
- First, the script reads in the trianing and test MNIST data, normalizes the values, and one-hot-encodes the labels
- Next, the network architecture, optimizer, and loss function are defined.
- Then, caerus network is trained for the number of epochs on the training data.
- Finally, labels are inferred on the test data and compared to the target labels, and an accuracy score is computed.

 Notes:
 - I tried to be too fancy with the code and messed up two major places:
    - Backprop doesn't work when there's more than one hidden layer in the network- not able to fully replicate paper's results.
    - Sigmoid activation creates odd behavior in the softmax sequential caerus network- weight values get clipped.


