[![Python CI](https://github.com/pkwitczak/tropicalpy/actions/workflows/tests.yml/badge.svg)](https://github.com/pkwitczak/tropicalpy/actions/workflows/tests.yml)


# TropicalPy

A simple module for performing tropical (min-plus and max-plus) linear algebra operations with NumPy arrays (see reference below for more about tropical linear algebra).  Written in cython for a modest speed boost.

Example:

    >>> import numpy as np
    >>> import tropicalpy as tp

    >>> A = np.array([[0,2,3],[4,0,6],[7,8,0]])

    >>> tp.pow(A,3)               # computes the tropical power A * A * A using min-plus algebra
    >>> tp.pow(A,3,max_plus=True) # same but with max-plus algebra


## Notes

Although this module is implemented in cython for some minor speed gains, matrix multiplication is still O(n^3) and will perform badly for large matrices.

To do:
- move to C++
- implement tropical eigenvalues, determinants, and rank.
- improve matrix multiplication
- improve cython code.

## References

This is a fork from https://github.com/lane203j/tropicalpy by Jeremy Lane. 

A good place to start learning about tropical linear algebra is

Maclagan D, Sturmfels B. "Introduction to Tropical Geometry." 2015

# Installation

PIP:
 pip install git+https://github.com/pkwitczak/tropicalpy.git@dev#egg=tropicalpy
