""" Implemementation of the message passing algorithm for planted clique.
(Deshpande & Montanari 2013)
"""

import numpy as np
import scipy.linalg as linalg
from scipy.misc import factorial2 as fac2
from scipy.misc import factorial as fac
from  scipy.special import binom
import math
import sample_graph

def get_poly(d, t, kappa):
    """ Returns a function generating the sequence of polynomials p(l, z)
    Args:
         d: Hyperparameter Controlling the number of moments
         t: Hyperparameter Controlling the number of steps
    Returns:
         The function p(z, l)  as defined in the paper
    """


    # Generate The first d moments of N(0,1)
    moments = np.array([(1 - i % 2) * fac2(i - 1) for i in np.arange(d + 1)])

    def get_moment(mu, k):
        """ Computes E[(mu + Z)^k] with Z~N(0, 1)
        """
        coeffs = np.array([binom(k, i) * pow(mu,  i)
                           for i in np.arange(k + 1)])

        return np.sum(np.multiply(moments[k::-1], coeffs))


    def get_mu(mu_t, L_t):
        """ Computes mu_{t+1}
        """
        coeffs = (1.0 / L_t) * np.array([pow(mu_t, i) / (fac(i))
                                         for i in np.arange(d + 1)])
        polys = np.array([get_moment(mu_t, i) for i in np.arange(d+1)])
        return kappa * sum(np.multiply(coeffs, polys))

    def get_L(mu_t, d):
        """ Returns L_t.
        """
        # ai = u_l^k / k!
        # sum_{i+j is even}} ai aj (i+j-1)!!
        a = np.array([pow(mu_t,k)/fac(k) for k in np.arange(d + 1)])

        # compute all the moments up to 2*d
        E = np.zeros(2*d+1)
        for i in np.arange(d+1):
            E[2*i]=fac2(2*i-1)

        H = np.zeros((d+1,d+1))
        for i in np.arange(d+1):
            for j in np.arange(d+1):
                H[i,j] = E[i+j]

        L2 = np.dot(a,np.dot(H,a))
        return math.sqrt(L2)

    # Compute Mu's , L's
    mu_list = [1.0, 1.0]
    L_list = [None]
    for i in range(1, t + 1):
        L_list.append(get_L(mu_list[i], d))
        mu_list.append(get_mu(mu_list[i], L_list[i]))

    # Define Function
    def poly(z, l):
        """ Computes the sequence of polynomials from the algorithm.

        Args:
            z: np.array to be evaluated
            l: Step Index
        Returns:
            The l-th Polynomial evaluated pointise on z
        """
        if not l:
            return np.ones(z.shape)
        coeffs = (1.0 / L_list[l]) * np.array([pow(mu_list[l], k) / (fac(k))
                                               for k in np.arange(d + 1)])
        pows = np.arange(d+1) * np.ones(z.shape + (d+1,))
        polys = np.power(np.expand_dims(z, 2), pows)
        return np.sum((coeffs * polys).T, 0).T

    print mu_list
    return poly, mu_list, L_list


def POLY(z, t):
    return pow(z, t)

def msg_pass(CN, W, d, t, rho):
    """ msg pass.
    """
    N = W.shape[1]
    poly, mus, ls = get_poly(d, t, CN / math.sqrt(N))

    A = W / math.sqrt(N)
    theta_N = belief_prop(A, t, poly)
    return theta_N, mus, ls



def belief_prop(A, T, poly):

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
        print(theta_N)

    return  theta_N


def belief_prop_iter(A, theta_N, theta_NN, poly, t):
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
    #F = np.zeros((N,N))
    #for i in np.arange(N):
    #for j in np.arange(N):
    #F[i,j] = poly(theta_NN[i,j],t)

    F = poly(theta_NN, t)
    F[np.diag_indices(N)] = 0

    # compute theta_N
    # theta_N[i] = sum_{j!=i} A[j,i] * F[j,i]
    #            = sum( A[,i] * F[,i]),
    # since A[i,i] = F[i,i] = 0
    theta_N = np.sum(F * A, 0)

    # compute theta_NN
    # theta_NN[i,] = theta_N[i] - A[,i] * F[,i]
    theta_NN = np.reshape(np.repeat(theta_N, N), (N, N)) - (A * F).T

    return theta_N, theta_NN


def predict_clique(theta_N, A, CN):
    mean = np.mean(np.abs(theta_N))
    C_tilde = (theta_N > 2*mean).nonzero()
    A_tilde = A[np.ix_(C_tilde, C_tilde)]
    _, eigs = linalg.eigh(A_tilde, eigvals=A.shape - (1,1))
    u_star = eigs[:,-1]
    B_N = np.argpartition(u_star, -CN)[-CN:]
    scores = np.sum(A[:,B_N] > 0, 1)
    is_clique = scores > (B_N.shape[0] / 2.0)
    return is_clique.nonzero()
