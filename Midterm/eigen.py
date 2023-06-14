import numpy as np
from numpy.linag import eig

A=np.array([[2, -3], [-3, 0]])
eigenvalue, eigenvector = eig(A) # w:eigenvalue, v: eigenvector
print (eigenvalue)