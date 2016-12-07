import numpy as np
import math
import gradient
import sample_graph

N = 1000
CN = math.ceil(1.1*math.sqrt(N))
clique = np.append(np.ones(CN), np.zeros(N-CN))
W = sample_graph.sample_graph(N, 0.5, clique)
grad_f = gradient.derivative(W)
log_f = gradient.loglike(W)

# run gradient ascent
v_0 = np.random.rand(N)
v1, iter1 = gradient.gradient_ascent(v_0, step = 1, epsilon = 0.001, max_iter = 1000, gradient = grad_f)
gradient.acc(v1,clique)

# check objective function
log_f(v1)
log_f(clique * 100 - 50)


# with backtrack line search
v_0 = np.random.rand(N)
v2, iter2 = gradient.gradient_ascent_backtrack(v_0, epsilon = 0.001, max_iter = 1000, gradient = grad_f, loglike = log_f, eta = 0.01)
gradient.acc(v2,clique)

log_f(v1)
log_f(40*clique-20)


