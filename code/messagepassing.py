""" Implemementation of the message passing algorithm for planted clique.
(Deshpande & Monatnari 2013, FInding)
"""

import numpy as np
from scipy.misc import factorial2 as fac2
from scipy.misc import factorial as fac
from  scipy.special import binom
import math

def msg_pass(W, d, t, rho):
    """ msg pass.
"""




def get_poly(d, t):
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
        return np.sum(np.multiply(moments[:k + 1], coeffs))


    def get_mu(mu_t, L_t):
        """ Computes mu_{t+1}
        """
        coeffs = (1.0 / L_t) * np.array([pow(mu_t, i) / (fac(i))
                                         for i in np.arange(d + 1)])
        polys = np.array([get_moment(mu_t, i) for i in np.arange(d+1)])
        return np.sum(np.multiply(coeffs, polys))

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
            z: point to evaluate the function at
            l: Step Index
        Returns:
            The l-th Polynomial evaluated at z
        """
        if not l:
            return 1.0
        coeffs = (1.0 / L_list[l]) * np.array([pow(mu_list[l], k) / (fac(k))
                                               for k in np.arange(d + 1)])
        polys = np.array([pow(z, k) for k in np.arange(d + 1)])
        print coeffs
        print polys
        return np.sum(np.multiply(coeffs, polys))

    return poly, L_list, mu_list
