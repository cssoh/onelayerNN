import numpy as np
from NetworkActivationFunction import layeractivationfunction,outputactivationfunction


def NeuralNetwork(netInput,inputWeight,layerWeight,inputBias,layerBias):
# Feed-forward neural network with one hidden layer,
# netInput=input to NN, (JxN) matrix
# inputWeight=weights between input layer and hidden layer, (IxJ) matrix,
# I=number of hidden neuron
# layerWeight=weights between hidden layer and output layer, (KxI) vector
# K=number of output
# inputBias=bias between input layer and hidden layer, (Ix1) vector
# layerBias=bias between hidden layer and output layer, (Kx1) vector

  J = len(netInput) # number of input variables
  N = len(netInput[0]) # number of input samples

  sumHiddenNode=np.dot(inputWeight,netInput)+np.tile(inputBias,(1,N)) # input to hidden layer, (IxN) matrix
  
  hiddenOutput=layeractivationfunction(sumHiddenNode) # output of hidden layer, (IxN) matrix
  
  sumOutputNode=np.dot(layerWeight,hiddenOutput)+np.tile(layerBias,(1,N)) # input to output node, (KxN) matrix
  
  netOutput=outputactivationfunction(sumOutputNode) # output of neural network, (KxN) matrix
 
  return netOutput

if __name__=="__main__":
  NeuralNetwork(np.array([[0,0,1,1],[0,1,0,1]]),np.array([[0,0],[0,0],[0,0]]),np.array([0,0,0]),np.array([[0],[0],[0]]),np.array([1]))

