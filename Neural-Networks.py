import numpy as np

# X = (sleeping hours,studying hours), y = test score
X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

# Normalize
X = X/np.amax(X, axis=0)
y = y/100  # Max score is 100

# Whole Class with additions:


class Neural_Network(object):
    def __init__(self):
        # Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        #Weights (parameters)
        self.W1 = np.random.randn(self.hiddenLayerSize, self.inputLayerSize)
        self.W2 = np.random.randn(self.outputLayerSize, self.hiddenLayerSize)

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
