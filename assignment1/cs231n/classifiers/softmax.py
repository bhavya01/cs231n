import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)
  for i in range(X.shape[0]):
    loss += -np.log(np.exp(scores[i][y[i]] - np.max(scores[i]))/np.sum(np.exp(scores[i] - np.max(scores[i])))) 
  loss =  loss/X.shape[0] + (reg/2)*np.sum(W**2)

  for i in range(X.shape[0]):
    dW[:,y[i]] += -X[i,:]
    for j in range(W.shape[1]):
      dW[:,j] += X[i,:]*np.exp(scores[i,j] - np.max(scores[i]))/(np.sum(np.exp(scores[i] - np.max(scores[i]))))
  dW = dW/X.shape[0] + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)
  scores = (scores.T - np.max(scores,axis =1)).T
  class_scores = scores[xrange(X.shape[0]),y]
  denom = np.sum(np.exp(scores),axis=1)
  loss = (1/X.shape[0])*np.sum(-np.log(np.exp(class_scores)/denom)) + (reg/2)*np.sum(W**2)

  ind = np.zeros_like(scores)
  ind[xrange(X.shape[0]),y] = 1	
  dW = -np.dot(X.T,ind)
  dW += np.dot(X.T,(np.exp(scores).T/denom).T)
  dW = dW/X.shape[0] + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

