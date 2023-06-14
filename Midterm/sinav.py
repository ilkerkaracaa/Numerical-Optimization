import numpy as np
import math

def f(xk):
    x1 = xk[0]
    x2 = xk[1]
    fk = (x1-25)**2 + (x2-36)**2 + 9*x1**2 + 4*x1**2*x2**2
    return fk

def gradient(xk):
    x1 = xk[0]
    x2 = xk[1]
    gk = np.array([8*x1*x2**2+20*x1-50,8*x1**2*x2+2*x2-72])
    return gk