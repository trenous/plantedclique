import sample_graph
import numpy as np
import math
import messagepassing

N = 1000
#CN = math.ceil(math.sqrt(N))
CN = math.ceil(math.sqrt(N*(1/math.e+0.1)))

clique = np.append(np.ones(CN), np.zeros(N-CN))
W = sample_graph.sample_graph(N, 0.5, clique)


d = 10
t = 8
rho = 1
theta_N, mu_t  = messagepassing.msg_pass(CN,W,d,t,rho)


