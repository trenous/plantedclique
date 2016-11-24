"""Tests for gradient Ascent and derivative computation.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import unittest
import numpy as np
from  code import gradient



class TestGradient(unittest.TestCase):

    def test_gradient_ascent(self):
        f = lambda x: -2*x + 4
        epsilon = 10**(-6)
        step = 0.5
        for _ in range(10):
            v_0= (np.random.rand() - 0.5) * 100
            maximizer, _ = gradient.gradient_ascent(v_0, step, epsilon, 10**5, f)
            self.assertAlmostEqual(maximizer, 2.0, 5)

    def test_derivative(self):
        W = -np.ones((2, 2))
        W[0, 0] = 0
        W[1, 1] = 0
        v = np.array([-1, -1])
        sig = 1/ (1 + np.exp(1))
        expected = -np.array([sig**3 / (1 - sig**2)] * 2)
        grad = gradient.derivative(W)
        np.testing.assert_almost_equal(expected, grad(v))

    def test_grad_ascent_W(self):
        W = -np.ones((10, 10))
        clq = [0, 1, 2]
        for i in clq:
            for j in clq:
                W[i, j] = 1
        for i in range(10):
            W[i, i] = 0
        W_prime = -W
        v = np.random.rand(10)
        grad_W = gradient.derivative(W)
        grad_W_prime = gradient.derivative(W_prime)
        epsilon = 10**(-5)
        step = 0.1
        maximizer_W, _ = gradient.gradient_ascent(v, step, epsilon, 10**5, grad_W)
        maximizer_W_prime, _ = gradient.gradient_ascent(v, step, epsilon, 10**5, grad_W_prime)
        indicator_W = 1 / (1 + np.exp(-maximizer_W))
        indicator_W_prime = 1 / (1 + np.exp(-maximizer_W_prime))
        clique_W = np.zeros(10)
        clique_W[clq] = 1
        clique_W_prime = -clique_W + 1
        rho_W = np.corrcoef(np.array([clique_W, indicator_W]))[0,0]
        rho_W_prime = np.corrcoef(np.array([clique_W_prime, indicator_W_prime]))[0,0]
        self.assertAlmostEqual(rho_W, 1)
        self.assertAlmostEqual(rho_W_prime, 1)



if __name__ == '__main__':
    unittest.main()
