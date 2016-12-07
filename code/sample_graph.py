'''Randomly sample a graph with planted clique.'''

import numpy as np

# we only need one function, # since WLOG we can 
# always assume the first |clique_size| nodes are in the clique.

def sample_graph(N,p,clique):
    ''' Function that returns adjacency matrix of a random graph
    given the planted clique.

    Args:
        N: number of nodes.
        p: the probability of sampling an edge betwen two nodes.
        clique: the indicator vector of the planted clique, 1d array of length N.

    Returns:
        W: the NxN adjacency matrix of the graph. 1 indicates connected and -1 unconnected.
    '''


    # randomly sample edges between nodes
    W = np.random.binomial(1,p,(N,N))
    W = 2 * W - 1 # rescale to (-1,1)

    # make W symmetric
    iu = np.triu_indices(N,1)
    W[iu] = W.T[iu]

    # plant clique
    idx = np.arange(N)[clique>0]
    W[np.ix_(idx,idx)] = 1

    # make diagonal entries zero
    W[np.diag_indices(N)] = 0

    return W

def sample_graph_three(N,p1,p2,clique):
    ''' Function that returns adjacency matrix of a random graph
    given the planted clique.

    Args:
        N: number of nodes.
        p1: the probability of sampling an edge betwen clicque node and non-clique node.
        p2: the probability of sampling an edge betwen non-clique nodes.
        clique: the indicator vector of the planted clique, 1d array of length N.

    Returns:
        W: the NxN adjacency matrix of the graph. 1 indicates connected and -1 unconnected.
    '''

    CN = np.sum(clique>0)

    # randomly sample edges between nodes
    W = np.random.binomial(1,p1,(N,N))
    W = 2 * W - 1 # rescale to (-1,1)

    # plant clique
    idx = np.arange(N)[clique>0]
    W[np.ix_(idx,idx)] = 1

    # resample among non-clique nodes
    W2 = np.random.binomial(1,p2,(N-CN,N-CN))
    W2 = 2 * W2 - 1 # rescale to (-1,1)
    iu = np.triu_indices(N-CN,1)
    W2[iu] = W2.T[iu]

    # replace old edge
    idx = np.arange(N)[clique==0]
    W[np.ix_(idx,idx)] = W2


    # make W symmetric
    iu = np.triu_indices(N,1)
    W[iu] = W.T[iu]


    # make diagonal entries zero
    W[np.diag_indices(N)] = 0

    return W


