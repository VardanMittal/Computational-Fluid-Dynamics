import numpy as np

def GaussElimination(A:np.array,b:np.array):
    """Function that solves gauss elimination by elementry row operations
    
    Keyword arguments:
    A - np.array(): matrix
    b - np.array(): vector
    Return: A,b
    """
    
    n = len(A)
    for i in range(n-1):
        for j in range(i+1,n):
            factor = float(A[j][i]/A[i][i])
            for k in range(i,n):
                A[j][k] -= factor * A[i][k]
            b[j] -= factor * b[i]

    x = np.zeros(n)
    for i in reversed(range(n)):
        s = 0
        for j in range(i+1,n):
            s += A[i][j] * x[j]
        x[i] = (b[i] - s) / A[i][i]

    return x

def TDMA():
    pass