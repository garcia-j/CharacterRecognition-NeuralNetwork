import numpy as np
import random

class Network(object):
    """
    Represents the Neural Network used for Character Recognition.
    
    Attributes:
        num_layers: The number of layers in the network.
        sizes: The number of neurons in respective layers.
        biases: The biases in the network in a list of matrices.
        weights: The weights in the network in a list of matrices.
    """
    
    def __init__(self, sizes):
        """
        Initializes the Neural Network. The number of Layers is initializes from
        the length of the number of neurons in respective layers. Weights and
        biases are initialized randomly.
        
        Arguments:
            sizes: The number of neurons in respective layers.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
    def feedForward(self, x):
        """
        Feeds the Network with the given input and returns the output.
        
        Arguments:
            x: The input to the Network. Must be an (n, 1) numPy array where n
               is the number of input neurons in the Network.
        
        Returns:
            The output of the Network with 'x' as input.
        """
        for bias, weight in zip(self.biases, self.weights):
            x = sigmoid(np.dot(weight, x) + bias)
        return x
    
    def gradientDescent(self, trainingData, generation, miniBatchSize, eta, testData = None):
        """
        Trains the Network utilizing mini-batch stochastic gradient descent.
        
        Arguments:
            trainingData: List of (x,y) tuples representing training inputs and desired outputs.
            generation: The current generation of training.
            miniBatchSize: The amount of training inputs randomly chosen.
            eta: The training rate.
            testData: If provided, the Network will be evaluated after each generation and
                      will print partial progress.
        """
        dataLen = len(trainingData)
        
        for i in range(generation):
            random.shuffle(trainingData)
            miniBatches = [trainingData[j: j + miniBatchSize]
                           for j in range(0,dataLen, miniBatchSize)]
            
            for miniBatch in miniBatches:
                self.update_miniBatch(miniBatch, eta)
            
            if testData:
                numTest = len(testData)
                print ("Generation {}: {} / {}").format(i, self.evaluate(testData), numTest)
            else:
                print ("Generation {0} complete").format(j)
            

        for i in range(generation):
            random.shuffle(trainingData)
            miniBatches = [trainingData[j: j + miniBatchSize]
                            for j in range(0, dataLen, miniBatchSize)]
            
            
def sigmoid(x):
    """
    Calculates the result of applying the sigmoid function to the given argument.
    
    Arguments:
        x: The parameter of the sigmoid function
    """
    return 1.0/(1.0 + np.exp(-x))
        