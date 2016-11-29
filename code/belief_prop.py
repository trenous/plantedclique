import numpy as np


def belief_prop_iter(A,theta_N, theta_NN, poly,t):
    ''' Return theta_N and theta_NN for next iteration.
    
    Args:
        A: W/sqrt(N), symmetric
        theta_N: vector of length N
        theta_NN: NxN matirx, assymetric. theta_NN[i,j] = msg i -> j
        poly: a polynomial function 
        t: time step, therefore f = poly(.,t)

        
    Returns:
        theta_N: the new value for next iteration
        theta_NN: the new value for next iteration
     '''
        
    # number of nodes
    N = A.shape[1]

    # f(theta_{l->i})
    F = poly(theta_NN,t)
    F[np.diag_indices(N)] = 0

    # compute theta_N 
    # theta_N[i] = sum_{j!=i} A[j,i] * F[j,i]
    #            = sum( A[,i] * F[,i]),
    # since A[i,i] = F[i,i] = 0
    theta_N = np.sum(F * A,0)

    # compute theta_NN
    # theta_NN[i,] = theta_N[i] - A[,i] * F[,i] 
    theta_NN = np.reshape(np.repeat(theta_N,N),(N,N)) -A * F.T 
   
    return theta_N, theta_NN

def belief_prop(A,T, poly):

    ''' Run belief propagation and return theta_N, theta_NN .

    Args:
        A: W/sqrt(N)
        T: number of rounds, t*
        poly: a polynomial function

    Returns:
        theta_N
        theta_NN
    '''

    # find number of nodes N
    N = A.shape[1]

    # let A have zero diagonal
    A[np.diag_indices(N)] = 0

    # initialize
    theta_N = np.ones(N)
    theta_NN = np.ones((N,N))
    theta_NN[np.diag_indices(N)] = 0

    # run T iterations
    for t in np.arange(T):
        theta_N,theta_NN = belief_prop_iter(A, theta_N,theta_NN, poly, t)

    return  theta_N, theta_NN


