import numpy as np
from numpy.linalg import eig

A = np.array([[5,2],[2,1]])

w,v = eig(A) #w: özdeğer, v: özvektör
print(w)

