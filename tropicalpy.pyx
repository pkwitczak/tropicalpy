# tropicalpy module
import math
import numpy as np
from scipy.optimize import linprog               # used for tropical eigenvalues
from scipy.optimize import linear_sum_assignment # tropical determinant

from cpython cimport bool
from cython.view cimport array as cvarray
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
cdef float INF = float('inf')

def assertSquare(np.ndarray A):
    # check array is square
    cdef int n = A.shape[0]
    cdef int m = A.shape[1]
    assert n == m
    return 

def zeros(shape,bool max_plus=False):
    # tropical additive identity
    cdef np.ndarray Z = np.zeros(shape,dtype=DTYPE)
    if max_plus:
        Z.fill(-INF)
        return Z
    else:
        Z.fill(INF)
        return Z

def eye(int n,bool max_plus=False):
    # tropical multiplicative identity
    cdef np.ndarray I = zeros((n,n),max_plus)
    for i in range(0,n):
        I[i,i] = 0
    return I

def schur(np.ndarray X,np.ndarray Y):
    # entry-wise tropical multiplication  (Schur product)
    # includes tropical scalar mul via numpy broadcasting
    return X+Y

def add(np.ndarray X,np.ndarray Y,bool max_plus=False):
    # tropical matrix addition
    if not max_plus:
        return np.minimum(X,Y)
    else:
        return np.maximum(X,Y)

def mul(np.ndarray X,np.ndarray Y,bool max_plus=False):
    # tropical matrix multiplication
    assert X.shape[1] == Y.shape[0]

    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int m = Y.shape[1]
    cdef np.ndarray S = X.reshape((n,1,p)) + Y.T.reshape((1,m,p))
    if max_plus:
        return np.max(S,axis=2)
    else:
        return np.min(S,axis=2)

def pow(np.ndarray A,int k,bool max_plus=False):
    # the tropical power of A
    assertSquare(A)
    assert (k > -1)
    cdef np.ndarray temp = A
    if k == 0:
        return eye(A.shape[0],max_plus)
    elif k == 1:
        return A
    else:
        for i in range(0,k-1):
            temp = mul(temp,A,max_plus)
        return temp

def kleenePlus(np.ndarray A, bool max_plus=False):
    # compute tropical A + A^2 + ... + A^n
    # where n is the dimensions of A
    assertSquare(A)
    cdef int n = A.shape[0]
    cdef np.ndarray Aplus = A

    for k in range(2,n+1):
        Aplus = add(Aplus, pow(A,k,max_plus),max_plus)

    return Aplus



def det(DTYPE_t [:,:] A,bool max_plus=False):
    #TODO: implement determinant
    pass
    return None

def eig(DTYPE_t [:,:] A, bool max_plus=False):
    #TODO: implement eigenvalues
    #test
    pass
    return None








