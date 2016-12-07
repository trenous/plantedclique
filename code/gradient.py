'''Implements gradient descent on an objective function whose global maximum
can be used to recover the planted clique.'''

import numpy as np

def loglike(W):
    ''' Functional that returns log f_W:

          log(f_W(v)) = log(Prod_{i<j}{ 1 + W[i,j] * sigmoid(v[i]) * sigmoid(v[j]) })
    Args:
       W: Adjacency Matrix of an undirected graph
          W[i,i] = 0 ; W[i,j] = (-1)^{I(i,j in Edge)}

    Returns:
       A function that computes The derivative of log f_W
    '''
    def log_f(v):
        """Evaluates the derivative of log f_W at point v
           Args:
             v: A vector in R^len(W) at which the derivative will be evaluated
           Returns:
              The derivative of log f_W at point v
        """
        sig = 1.0 /(1+np.exp(-v))
        # prod[i,j] = sig[i] * sig[j] * W[i,j] ]
        prod = ((sig * W).T) * sig
        # log_matrix[i,j] = log (1 + sig[i]*sig[j]*W[i,j])
        log_matrix = np.log(1.0 + prod)
        # times 0.5 because we only want log_matrix[i,j] for i<j
        log = 0.5*np.sum(log_matrix)
        return log 

    return log_f 




def derivative(W):
    ''' Functional that returns the first derivative of log f_W with:

          f_W(v) = Prod_{i<j}{ 1 + W[i,j] * sigmoid(v[i]) * sigmoid(v[j]) }

        Which is

          d log f_W / d v[i] = sigmoid(v[i])^2 * e^(-v[i]) * sum_j{
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
        expv = np.exp(-v)
        # prod[i,j] = sig[i] * sig[j] * W[i,j] ]
        prod = ((sig * W).T) * sig
        # c_matrix[i,j] = (sig[i]*exp(-v[i])*sig[i]*sig[j]*W[i,j]) / (1 + sig[i]*sig[j]*W[i,j])
        contribution_matrix = (sig * expv * prod / (1.0 + prod)) 
        # grad[i] =  d log f_W / d v[i]
        # NSZ note: when computing c_matrix, each row of (prod/1+prod)
        # should be multiplied by same number sig[i] * expv[i],
        # so it should rather be c_matrix = (sig * expv * prod/(1+prod)).T.
        # But we can also just take sum along axis = 0 instead.
        grad = np.sum(contribution_matrix, axis = 0)

        return grad

    return grad_f

def gradient_ascent(v_0, step, epsilon, max_iter, gradient):
    """ Gradient ascent with fixed step size.

       Args:
            v_0: Starting position
           step: step size for gradient descent
        epsilon: Stopping criterion
       max_iter: Stopping CRiterion
       gradient: A function that computes the gradient of the
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
        #print(delta)
        if delta <= epsilon:
            break
        v_old = v_new
    return v_new, iteration


def gradient_ascent_backtrack(v_0, epsilon, max_iter, gradient, loglike, eta = 0.001, gamma = 0.5):
    """ Gradient ascent with backtrack line search.

       Args:
            v_0: Starting position
           step: step size for gradient descent
        epsilon: Stopping criterion
       max_iter: Stopping CRiterion
       gradient: A function that computes the gradient of the
                 function to be minimized.
        loglike: A function that computes log likelihood. 
            eta: sufficient decrese criteria for backtrack line search.
          gamma: contraction parameter

      Returns:
             The minimizere v*
    """
    iteration = 0
    v_old = v_0
    F_old = loglike(v_old)
    g_old = gradient(v_old)
    v_new = None
    while iteration < max_iter:
        iteration += 1
        alp = 1
        v_new = v_old + alp * g_old
        F_new = loglike(v_new)
        while (F_new  < F_old + alp*eta*np.sum(g_old*g_old)):
            alp = alp*gamma
            v_new = v_old + alp * g_old
            F_new = loglike(v_new)
            print(iteration,alp)

        delta = np.linalg.norm(v_old - v_new)
        # print(delta)
        if delta <= epsilon:
            break
        v_old = v_new
        F_old = loglike(v_old)
        g_old = gradient(v_old)

    return v_new, iteration

def acc(v,clique):
    N = len(v)
    return float(np.sum((clique >0) * (v>0))+ np.sum((clique <1) * (v<0)))/N



