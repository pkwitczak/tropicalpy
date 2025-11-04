import unittest
import numpy as np

import sys
import os

import pyximport
pyximport.install()

import tropicalpy

class TestTropicalPy(unittest.TestCase):

    def test_zeros_min_plus(self):
        Z = tropicalpy.zeros((2,3), max_plus=False)
        self.assertEqual(Z.shape, (2,3))
        self.assertTrue(np.allclose(Z, np.inf))

    def test_zeros_max_plus(self):
        Z = tropicalpy.zeros((2,3), max_plus=True)
        self.assertEqual(Z.shape, (2,3))
        self.assertTrue(np.allclose(Z, -np.inf))

    def test_eye_min_plus(self):
        I = tropicalpy.eye(3, max_plus=False)
        expected = np.full((3, 3), np.inf)
        np.fill_diagonal(expected, 0)
        self.assertTrue(np.allclose(I, expected))

    def test_eye_max_plus(self):
        I = tropicalpy.eye(3, max_plus=True)
        expected = np.full((3, 3), -np.inf)
        np.fill_diagonal(expected, 0)
        self.assertTrue(np.allclose(I, expected))

    def test_schur(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[4, 3], [2, 1]])
        result = tropicalpy.schur(A, B)
        expected = A + B
        self.assertTrue(np.array_equal(result, expected))

    def test_add_min_plus(self):
        A = np.array([[1, 5], [7, 3]])
        B = np.array([[2, 4], [6, 4]])
        result = tropicalpy.add(A, B, max_plus=False)
        expected = np.minimum(A, B)
        self.assertTrue(np.array_equal(result, expected))

    def test_add_max_plus(self):
        A = np.array([[1, 5], [7, 3]])
        B = np.array([[2, 4], [6, 4]])
        result = tropicalpy.add(A, B, max_plus=True)
        expected = np.maximum(A, B)
        self.assertTrue(np.array_equal(result, expected))

    def test_mul_min_plus(self):
        A = np.array([[0, 1], [2, 3]])
        B = np.array([[4, 5], [6, 7]])
        result = tropicalpy.mul(A, B, max_plus=False)
        # calculate expected by tropical min-plus multiplication
        expected = np.empty((2, 2))
        expected[0,0] = min(A[0,0]+B[0,0], A[0,1]+B[1,0])
        expected[0,1] = min(A[0,0]+B[0,1], A[0,1]+B[1,1])
        expected[1,0] = min(A[1,0]+B[0,0], A[1,1]+B[1,0])
        expected[1,1] = min(A[1,0]+B[0,1], A[1,1]+B[1,1])
        self.assertTrue(np.allclose(result, expected))

    def test_mul_max_plus(self):
        A = np.array([[0, 1], [2, 3]])
        B = np.array([[4, 5], [6, 7]])
        result = tropicalpy.mul(A, B, max_plus=True)
        expected = np.empty((2, 2))
        expected[0,0] = max(A[0,0]+B[0,0], A[0,1]+B[1,0])
        expected[0,1] = max(A[0,0]+B[0,1], A[0,1]+B[1,1])
        expected[1,0] = max(A[1,0]+B[0,0], A[1,1]+B[1,0])
        expected[1,1] = max(A[1,0]+B[0,1], A[1,1]+B[1,1])
        self.assertTrue(np.allclose(result, expected))

    def test_pow_identity(self):
        A = np.array([[0, np.inf], [np.inf, 0]])
        result = tropicalpy.pow(A, 0, max_plus=False)
        expected = tropicalpy.eye(2, max_plus=False)
        self.assertTrue(np.array_equal(result, expected))

    def test_pow_one(self):
        A = np.array([[0, 2], [3, 0]])
        result = tropicalpy.pow(A, 1, max_plus=True)
        self.assertTrue(np.array_equal(result, A))

    def test_pow_multiple(self):
        A = np.array([[0, 1], [1, 0]])
        result = tropicalpy.pow(A, 2, max_plus=False)
        expected = tropicalpy.mul(A, A, max_plus=False)
        self.assertTrue(np.allclose(result, expected))

    def test_kleenePlus(self):
        A = np.array([[0, 1], [2, 0]])
        result = tropicalpy.kleenePlus(A, max_plus=False)
        n = A.shape[0]
        expected = A.copy()
        for k in range(2, n+1):
            expected = tropicalpy.add(expected, tropicalpy.pow(A, k, max_plus=False), max_plus=False)
        self.assertTrue(np.allclose(result, expected))


if __name__ == '__main__':
    unittest.main()
