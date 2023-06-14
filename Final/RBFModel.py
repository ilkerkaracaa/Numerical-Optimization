import numpy as np
import math
from birinciSoru import ti, yi

def gaussianfunction(t,c,s):
    return math.exp(-(t-c)**2/(s**2))

def RBFIO(t,x,c,s):
    yhat = []
    for ti in t:
        toplam = 0
        for i in range(0,len(x)):
            toplam += x[i] * gaussianfunction(ti,c[i],s[i])
        yhat.append(toplam)
    return yhat

def findxcs(ti,yi,RBFsayisi):
    lenghtofsegment = (max(ti) - min(ti))/(RBFsayisi)
    s = [lenghtofsegment for tmp in range(0,RBFsayisi)]
    c = [min(ti) + lenghtofsegment/2 + lenghtofsegment*i for i in range(0,RBFsayisi)]
    numofdata = len(ti)
    J=np.zeros((numofdata,RBFsayisi))
    for i in range(0,numofdata):
        for j in range(0,RBFsayisi):
            J[i,j] = gaussianfunction(ti[i],c[j],s[j])
    A = np.linalg.inv(J.transpose().dot(J))
    B = J.transpose().dot(yi)
    x = -A.dot(B)
    return x,c,s

trainingindices = np.arange(0,len(ti),1)
traininginput = np.array(ti)[trainingindices]
trainingoutput = np.array(yi)[trainingindices]
validationindices = np.arange(1,len(ti),2)
validationinput = np.array(ti)[validationindices]
validationoutput = np.array(yi)[validationindices]

RBF = []; FV = []
for RBFSayisi in range(1,12):
    x,c,s = findxcs(traininginput,trainingoutput,RBFSayisi)
    yhat = RBFIO(validationinput,x,c,s)
    e = np.array(validationoutput) - np.array(yhat)
    fvalidation = sum(e**2)
    RBF.append(RBFSayisi)
    FV.append(fvalidation)
    print("RBF Sayisi: ",RBFSayisi," Fvalidation: ",fvalidation)