import numpy as np
import math
from birinciSoru import ti, yi

def polinomIO(t,x):
    yhat = [x[0]+x[1]*ti+x[2]*ti**2 for ti in t]
    return yhat

numofdata = len(ti)
J = -np.ones((numofdata,1))
J = np.hstack((J,-np.ones((numofdata,1))*np.array(ti).reshape(numofdata,1)))
J = np.hstack((J,-np.ones((numofdata,1))*np.array(ti).reshape(numofdata,1)**2))
A = np.linalg.inv(J.transpose().dot(J))
B = J.transpose().dot(yi)
x = -A.dot(B)
T = np.arange(-3,3,0.1)
yhat = polinomIO(T,x)