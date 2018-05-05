import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
# X = (sleeping hours,studying hours), y = test score
X = np.array(([3, 5, 10, 6], [5, 1, 2, 1.5]), dtype=float)
y = np.array(([75, 82, 93, 70]), dtype=float)

# Normalize
X = X/np.amax(X, axis=1).reshape(2, 1)
y = y/100  

class Neural_Network(object):
    def __init__(self):
        # Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        #Weights (parameters)
        # np.random.randn(self.hiddenLayerSize, self.inputLayerSize)

        self.W1 = np.random.randn(self.hiddenLayerSize, self.inputLayerSize)
        self.W2 = np.random.randn(self.outputLayerSize, self.hiddenLayerSize)
        self.Lambda = 0.001
        # self.W1 = np.reshape(np.arange(6, dtype=np.float), (3, 2))
        # self.W2 = np.reshape(np.arange(3, dtype=np.float), (1, 3))

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

    def sigmoidPrime(self, z):
        # Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)

        J = 0.5*np.sum((y-self.yHat)**2) / \
            X.shape[0] + (self.Lambda/2) * \
            (np.sum(self.W1**2)+np.sum(self.W2**2))
        return J

    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(delta3, self.a2.T)/X.shape[1] + self.Lambda*self.W2

        delta2 = np.dot(self.W2.T, delta3)*self.sigmoidPrime(self.z2)

        dJdW1 = np.dot(delta2, X.T)/X.shape[1] + self.Lambda*self.W1

        return dJdW1, dJdW2

    def getParams(self):
        # Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))

        return params

    def setParams(self, params):
        # Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end],
                             (self.hiddenLayerSize, self.inputLayerSize))

        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(
            params[W1_end:W2_end], (self.outputLayerSize, self.hiddenLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)

        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


def computeNumericalGradient(N, X, y):
    paramsInitial = N.getParams()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4

    for p in range(len(paramsInitial)):
        # Set perturbation vector
        perturb[p] = e
        N.setParams(paramsInitial + perturb)
        loss2 = N.costFunction(X, y)

        N.setParams(paramsInitial - perturb)
        loss1 = N.costFunction(X, y)

        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)

        # Return the value we changed to zero:
        perturb[p] = 0

    # Return Params to original value:
    N.setParams(paramsInitial)

    return numgrad


class trainer(object):
    def __init__(self, N):
        # Make Local reference to network:
        self.N = N
        self.xx = 1

    def callbackF(self, params):
        #         if self.xx:
        #             print(params)
        #             self.xx=0

        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)

        return cost, grad

    def train(self, X, y):
        # Make an internal variable for the callback function:
        self.X = X
        self.y = y

        # Make empty list to store costs:
        self.J = []

        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS',
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res

# numerical gradient checking
# nn = Neural_Network()
# numgrad = computeNumericalGradient(nn, X, y)

# grad = nn.computeGradients(X, y)
# print(grad-numgrad)
# print(np.linalg.linalg.norm(grad-numgrad)/np.linalg.linalg.norm(grad+numgrad))

# a, b = nn.costFunctionPrime(X, y)
# print(a, b)



# Training Data:
trainX = np.array(([3, 5, 10, 6], [5, 1, 2, 1.5]), dtype=float)
trainY = np.array(([75, 82, 93, 70]), dtype=float)

# Testing Data:
testX = np.array(([4, 4.5, 9, 6], [5, 1, 2.5, 2]), dtype=float)
testY = np.array(([65, 89, 85, 75]), dtype=float)

# Normalize:

# Normalize
trainX = trainX/np.amax(trainX, axis=1).reshape(2, 1)
trainY = trainY/100  # Max score is 100

# Normalize:
testX = testX/np.array([[10.], [5.]])

testY = testY/100  # Max test score is 100


# Modify trainer class to check testing error during training:
class trainer_test(object):
    def __init__(self, N):
        # Make Local reference to network:
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
        self.testJ.append(self.N.costFunction(self.testX, self.testY))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)
        return cost, grad

    def train(self, trainX, trainY, testX, testY):
        # Make an internal variable for the callback function:
        self.X = trainX
        self.y = trainY

        self.testX = testX
        self.testY = testY

        # Make empty list to store costs:
        self.J = []
        self.testJ = []

        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS',
                                 args=(trainX, trainY), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        print(_res.x)
        self.optimizationResults = _res


# Train network with new data:
NN = Neural_Network()

T = trainer_test(NN)

T.train(trainX, trainY, testX, testY)
plt.plot(T.J, linewidth=6)
plt.plot(T.testJ)
plt.grid(1)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()
print(NN.forward(np.array([[6.], [4.]])/np.array([[10.], [5.]])))


# best weight for lowest cost function for test data--------[ 0.1977928  -0.0625821   0.73122814 -0.10796751  0.40894123  0.01463633
#   0.49792095  1.11995794  0.78996266]



# NN = Neural_Network()
# NN.setParams([2.92283085,   2.26309647, 15.40567647, -39.72529278,  10.15396877,
#               13.87667802,  11.80661059,   3.27781238, -10.21810612]
#              )
# print(NN.forward(np.array([[10.], [4.]])/np.array([[10.], [5.]])))
