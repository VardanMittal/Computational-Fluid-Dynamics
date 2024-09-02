import numpy as np

def GaussElimination(A:np.array,b:np.array):
    """Function that solves gauss elimination by elementry row operations
    
    Keyword arguments:
    A - np.array(): matrix
    b - np.array(): vector
    Return: solution vector x
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

def TDMA(d,l,u,b):
    """Function that solves TDMA
    
    Keyword arguments:
    d - np.array(): diagonal vector
    b - np.array(): known vector
    l - np.array(): lower diagonal vector
    u - np.array(): upper diagonal vector
    Return: x -> solution vector
    """
    n = len(d)
    for i in range(1, n):
        factor = l[i-1] / d[i-1]
        d[i] = d[i] - factor * u[i-1]
        b[i] = b[i] - factor * b[i-1]
    x = np.zeros(n)
    x[-1] = b[-1] / d[-1]

    for i in reversed(range(n-1)):
        x[i] = (b[i] - u[i] * x[i+1]) / d[i]

    return x