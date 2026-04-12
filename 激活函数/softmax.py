import numpy as np

def softmax(x):
  x = np.asarray(x, dtype = np.float64)
  x_max = np.max(x, axis = -1, keepdim = True)
  e_x = np.exp(x - x_max)
  return e_x / np.sum(x, axis = -1, keepdim = True)
