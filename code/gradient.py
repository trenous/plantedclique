'''Implements gradient descent on an objective function whose global maximum
can be used to recover the planted clique.'''

import numpy as np

def derivative(W):
    ''' Functional that returns the first derivative of log f_W with:

          f_W(v) = Prod_{i<j}{ 1 + W[i,j] * sigmoid(v[i]) * sigmoid(v[j]) }

        Which is

          d log f_W / d v[i] = sigmoid(v[i])^2 * sum_j{
                                (W[i,j] * sigmoid(v[j])) /
                                (1 + W[i,j] * sigmoid(v[i]) * sigmoid(v[j]))
                                }
    Args:
       W: Adjacency Matrix of an undirected graph
          W[i,i] = 0 ; W[i,j] = (-1)^{I(i,j in Edge)}

    Returns:
       A function that computes The derivative of log f_W

    '''
    def grad_f(v):
        """Evaluates the derivative of log f_W at point v
           Args:
             v: A vector in R^len(W) at which the derivative will be evaluated
           Returns:
              The derivative of log f_W at point v
        """
        sig = 1.0 /(1+np.exp(-v))
        # prod[i,j] = sig[i] * sig[j] * W[i,j] ]
        prod = ((sig * W).T) * sig
        # c_matrix[i,j] = (sig[i]^2*sig[j]*W[i,j]) / (1 + sig[i]*sig[j]*W[i,j])
        contribution_matrix = (sig * prod) / (1.0 + prod)
        # grad[i] =  d log f_W / d v[i]
        grad = np.sum(contribution_matrix, axis=1)
        return grad

    return grad_f

def gradient_ascent(v_0, step, epsilon, max_iter, gradient):
    """ Gradient ascent with fixed step size.

       Args:
            v_0: Starting position
           step: step size for gradient descent
        epsilon: Stopping criterion
       max_iter: Stopping CRiterion
       gradient: A function that compute sthe gradient of the
                 function to be minimized.

      Returns:
             The minimizere v*
    """
    iteration = 0
    v_old = v_0
    v_new = None
    while iteration < max_iter:
        iteration += 1
        v_new = v_old + step * gradient(v_old)
        delta = np.linalg.norm(v_old - v_new)
        if delta <= epsilon:
            break
        v_old = v_new
    return v_new, iteration
