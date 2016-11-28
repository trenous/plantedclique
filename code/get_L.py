import numpy as np
import scipy.misc
import math

def get_L(mu_t, d):
    """ Returns L_t.
    """
    # ai = u_l^k / k!
    # sum_{i+j is odd}} ai aj (i+j-1)!!
    mu_t = float(mu_t)

    A = np.ones(d+1)
    for k in np.arange(d+1):
        A[k] = pow(mu_t,k) / math.factorial(k)
 
    H = np.zeros((d+1,d+1))
    for i in np.arange(d+1):
        for j in np.arange(i,d+1):
            if (i+j) % 2 == 1:
                H[i,j] = 0
            else:
                H[i,j] = scipy.misc.factorial2(i+j-1)
    
    iu = np.triu_indices(d+1,1)
    H.T[iu] = H[iu]
    
    L2 = np.dot(A,np.dot(H,A))
    return math.sqrt(L2)
    

