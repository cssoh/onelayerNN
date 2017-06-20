# Function to test neural network training using back propagation algorithm

import numpy as np
from trainNeuralNetwork import trainNeuralNetwork
from NeuralNetwork import NeuralNetwork

def mainTestNNtraining():

  rawtraininput = np.array([[0.0,0.0,1.0,1.0],[0.0,1.0,0.0,1.0]])

  rawtrainoutput = np.array([0.0,1.0,1.0,0.0])

  inputWeight = np.array([[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0]])

  layerWeight = np.array([1,1,1,1])

  inputBias = np.array([[1],[1],[1],[1]])

  layerBias = np.array([1])

  # Normalize input samples with respect to mean and standard deviation
  nvar = len(rawtraininput) # number of input variables per input sample
  input_data_mean = np.mean(rawtraininput,axis=1).reshape(nvar,1)
  input_data_stddev = np.transpose(np.std(rawtraininput,axis=1)).reshape(nvar,1)
  nsamples = len(rawtraininput[0]) # number of input training samples
  traininput = np.divide(rawtraininput-np.tile(input_data_mean,(1,nsamples)),np.tile(input_data_stddev,(1,nsamples)))

  # Normalize output samples to between 0 and 1
  nvar = len(rawtrainoutput) # number of output variables per output sample
  #output_data_mean = np.mean(rawtrainoutput)
  #output_data_stddev = np.std(rawtrainoutput)
  out_max = np.amax(rawtrainoutput) # get the maximum values for each output variables
  out_min = np.amin(rawtrainoutput) # get the minimum values for each output variables
  nsamples = len(rawtrainoutput) # number of output training samples
  #trainoutput = np.divide(rawtrainoutput-np.tile(output_data_mean,(1,nsamples)),np.tile(output_data_stddev,(1,nsamples)))
  trainoutput = np.divide(rawtrainoutput-np.tile(out_min,(1,nsamples)),np.tile(out_max-out_min,(1,nsamples)))  
  print(trainoutput)
  inputWeight,layerWeight,inputBias,layerBias,meansquareError = trainNeuralNetwork(traininput,trainoutput,inputWeight,layerWeight,inputBias,layerBias)

  rawtestinput = np.array([[0.0,0.0,1.0,1.0],[0.0,1.0,0.0,1.0]])
  nsamples = len(rawtestinput[0])
  testInput = np.divide(rawtestinput-np.tile(input_data_mean,(1,nsamples)),np.tile(input_data_stddev,(1,nsamples)))
  netOutput = NeuralNetwork(testInput,inputWeight,layerWeight,inputBias,layerBias)

  #rawOutput = np.multiply(netOutput,np.tile(output_data_stddev,(1,nsamples)))+np.tile(output_data_mean,(1,nsamples))
  # Inverse the normalization
  rawOutput = np.multiply(netOutput,out_max-out_min)+out_min
  print("Net Output (normalized): ")
  print(repr(netOutput))
  print("Output (no normalization): ")
  print(repr(rawOutput))

if __name__=="__main__":
  mainTestNNtraining()
