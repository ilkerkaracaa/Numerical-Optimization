import numpy as np

ti = np.array([-2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6])
yi = np.array([-1.5, 1.30, 2.64, 2.94, 2.62, 2.10, 1.80, 2.15, 3.56, 6.47])

def polinomIO(t, x):
    yhat = []
    for ti in t:
        toplam = 0
        for i in range(0, len(x)):
            toplam += x[i]*ti** i
        yhat.append(toplam)
    return yhat 

def findx(ti, yi, polinomderecesi):
    numofdata = len(ti)
    J = -np.ones((numofdata, 1))
    for n in range(1, polinomderecesi+1):
        J = np.hstack((J,-np.ones((numofdata, 1)) * np.array(ti).reshape(numofdata, 1)**n))
    A = np.linalg.inv(J.transpose().dot(J))
    B = J.transpose().dot(yi)
    x = -A.dot(B)
    return x  

trainingindices = np.arange(0, len(ti), 2)
traininginput = np.array(ti)[trainingindices]
trainingoutput = np.array(yi)[trainingindices]
validationindices = np.arange(1, len(ti), 2)
validationinput = np.array(ti)[validationindices]
validationoutput = np.array(yi)[validationindices]

PD = []
FV = []
for polinomderecesi in range(0,5):
    x = findx(traininginput, trainingoutput, polinomderecesi)
    yhat = polinomIO(validationinput, x)
    e = np.array(validationoutput) - np.array(yhat)
    fvalidation = sum(e**2)
    PD.append(polinomderecesi)
    FV.append(fvalidation)
    print("Polinom derecesi:", polinomderecesi, "Fvalidation:", fvalidation)