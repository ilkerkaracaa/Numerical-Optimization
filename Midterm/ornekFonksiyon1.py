import numpy as np
import math

def f(xk):
    x1 = xk[0]
    x2 = xk[1]
    fk = 3 + (x1-1.5*x2)**2 + (x2-2)**2
    return fk

def gradient(xk):
    x1 = xk[0]
    x2 = xk[1]
    gk = np.array([2*(x1-1.5*x2),-3*x1+6.5*x2-4])
    return gk

