
import numpy as np

#--------Activation Functions------------------------------------------
def layeractivationfunction(x):
  I = len(x) # I = number of variables per input sample
  N = len(x[0]) # N = number of input samples
  functionOutput=np.divide(np.ones((I,N),dtype=float), (np.ones((I,N),dtype=float)+np.exp(-x))) # binary sigmoid activation function
  return functionOutput

def outputactivationfunction(x):
  I = len(x) # I = number of variables per input sample
  N = len(x[0]) # N = number of input samples
  functionOutput=np.divide(np.tile(1.,(I,N)),(np.tile(1.,(I,N))+np.exp(-x))) # binary sigmoid activation function
  return functionOutput

#-------derivative of activation functions-------------------------
def derivativelayeractivation(x):
  I = len(x) # I = number of variables per input sample
  N = len(x[0]) # N = number of input samples
  derivativevalue=np.multiply(layeractivationfunction(x),(np.ones((I,N),dtype=float)-layeractivationfunction(x)))
  return derivativevalue

def derivativeoutputactivation(x):
  I = len(x) # I = number of variables per input sample
  N = len(x[0]) # N = number of input samples
  derivativevalue=np.multiply(outputactivationfunction(x),(np.tile(1.,(I,N))-outputactivationfunction(x)))
  return derivativevalue
