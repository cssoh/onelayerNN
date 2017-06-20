# Function to train neural network using back propagation algorithm

import numpy as np
import math as mt
from NetworkActivationFunction import layeractivationfunction, outputactivationfunction, derivativelayeractivation, derivativeoutputactivation
from NeuralNetwork import NeuralNetwork

def trainNeuralNetwork(traininput,trainoutput,inputWeight,layerWeight,inputBias,layerBias):
# function to train feedforward neural network using back propagation
# algorithm
# traininput=training set input, (JxN) matrix
# N=number of examples
# J=number of input variables/predictors
# trainoutput=training set output, (KxN) matrix
# K=number of outputs
# traininput and trainoutput are scaled/normalized
# inputWeight=weights between the input layers and hidden layers, (IXJ) matrix
# layerWeight=weight between the hidden layer and output layers, (KxI) matrix
# inputBias=bias of hidden layer, (Ix1) vector
# layerBias=bias of output layer, (Kx1) vector

  print("Neural network training")
  numhiddenneuron=input("Enter number of hidden neurons: ")
  learnrate=input("Enter learning rate: ")
  learnrate=float(learnrate)
  momentum=input("Enter momentum: ")
  momentum=float(momentum)
  numepochs=input("Enter maximum number of epochs: ")
  numepochs = int(numepochs)

  I= int(numhiddenneuron);
  N = len(traininput[0])
  J = len(traininput)
  K =len(trainoutput)
  print("Number of samples= " + repr(N))
  print("Number of input variables= " + repr(J))
  print("Number of output variables= " + repr(K))

  # Initialize weights
  # if nargin<3 # randomly initialize weights
  inputWeight=np.random.rand(I,J)-0.5*np.ones((I,J),dtype=np.float)
  print(inputWeight)
  layerWeight=np.random.rand(K,I)-0.5*np.ones((K,I),dtype=np.float)
  print(layerWeight)
  inputBias=np.random.rand(I,1)-0.5*np.ones((I,1),dtype=np.float)
  print(inputBias)
  layerBias=np.random.rand(K,1)-0.5*np.ones((K,1),dtype=np.float)
  print(layerBias)
  #end
  meansquareError=1000
  olddeltalayerWeight=np.zeros((K,I),dtype=np.float)
  olddeltainputWeight=np.zeros((I,J),dtype=np.float)
  i=1;
  while i<=numepochs and meansquareError>0.0000000001:
    for n in range(N):
      # Feed-forward
      sumHiddenNode=np.dot(inputWeight,traininput[:,n].reshape(J,1))+inputBias # input to hidden layer, (Ix1) vector
      hiddenOutput=layeractivationfunction(sumHiddenNode) # output of hidden layer, (Ix1) vector
      sumOutputNode=np.dot(layerWeight,hiddenOutput)+layerBias # input to output node, (Kx1) vector
      netOutput=outputactivationfunction(sumOutputNode) # output of neural network, (Kx1) vector
      
      # Back-propagation of error
      # Output layer
      deltaOutput=np.multiply((trainoutput[:,n]-netOutput),derivativeoutputactivation(sumOutputNode)) # (Kx1) vector
      deltalayerWeight=np.multiply(learnrate,np.dot(deltaOutput,np.transpose(hiddenOutput)))+np.multiply(momentum,olddeltalayerWeight) 
	# (KxI) matrix)
      deltalayerBias=np.multiply(learnrate,deltaOutput) # (Kx1) vector
      olddeltalayerWeight=deltalayerWeight # store layer weight changes. (KxI) matrix

      # Hidden layer
      delta_in=np.sum(np.multiply(np.tile(deltaOutput,(1,I)),layerWeight),axis=0) # (KxI) matrix sum column into (1xI) vector
      deltaInput=np.multiply(delta_in,derivativelayeractivation(np.transpose(sumHiddenNode))) # (1xI) vector
      deltainputWeight=np.multiply(learnrate,np.dot(np.transpose(deltaInput),np.transpose(traininput[:,n].reshape(J,1))))+np.multiply(momentum,olddeltainputWeight) # (IxJ) matrix
      deltainputBias=np.multiply(learnrate,np.transpose(deltaInput)) # (Ix1) vector
      olddeltainputWeight=deltainputWeight # store input weight changes, (IxJ) matrix

      #Update weights and biases
      inputWeight=inputWeight+deltainputWeight # update input weights, (IxJ) matrix
      layerWeight=layerWeight+deltalayerWeight # update layer weights, (KxI) matrix
      inputBias=inputBias+deltainputBias # update input bias, (Ix1) vector
      layerBias=layerBias+deltalayerBias # update layer bias, (Kx1) vector

  

    simOutput = NeuralNetwork(traininput,inputWeight,layerWeight,inputBias,layerBias)
    residual = trainoutput-simOutput
    residual = residual[0]
    meansquareError=sum(np.power(residual,2))/N
    i=i+1
    if mt.fmod(i,1000)==0:
      print("i=" + repr(i) + ", MSE=" +repr(meansquareError))

  print("Input Layer to Hidden Layer Weights:")
  print(repr(inputWeight))
  print("Input Bias: ")
  print(repr(inputBias))
  print("Hidden Layer to Output Layer Weights:")
  print(repr(layerWeight))
  print("Hidden Layer Bias:")
  print(repr(layerBias))

  return inputWeight,layerWeight,inputBias,layerBias,meansquareError
#---------------------------------------------------------------------

if __name__=="__main__":
  trainNeuralNetwork(traininput=np.array([[0,0,1,1],[0,1,0,1]]),trainoutput=np.array([0,1,1,0]),inputWeight=np.array([[1,1],[1,1],[1,1],[1,1]]),layerWeight=np.array([1,1,1,1]),inputBias=np.array([[1],[1],[1],[1]]),layerBias=np.array([1]))

