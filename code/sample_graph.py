'''Randomly sample a graph with planted clique.'''

import numpy as np

# we only need one function, # since WLOG we can 
# always assume the first |clique_size| nodes are in the clique.

def sample_graph(N,p,clique):
    ''' Function that returns adjacency matrix of a random graph
    given the planted clique.

    Args:
        N: number of nodes
        p: the probability of sampling an edge betwen two nodes
        clique: the indicator vector of the planted clique, 1d array of length N

    Returns:
        W: the NxN adjacency matrix of the graph
    '''


    # randomly sample edges between nodes
    W = np.random.binomial(1,p,(N,N))

    # make W symmetric
    iu = np.triu_indices(N,1)
    W[iu] = W.T[iu]

    # plant clique
    idx = np.arange(N)[clique>0]
    W[np.ix_(idx,idx)] = 1

    # make diagonal entries zero
    W[np.diag_indices(N)] = 0

    return W


