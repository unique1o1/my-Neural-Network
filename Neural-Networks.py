import numpy as np

# X = (sleeping hours,studying hours), y = test score
X = np.array(([3, 5, 10], [5, 1, 2]), dtype=float)
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

    def forward(self, X):
        # Propagate inputs through network
        self.z2 = np.dot(self.W1, X)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.W2, self.a2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J


nn = Neural_Network()
print(nn.costFunction(X, y))
