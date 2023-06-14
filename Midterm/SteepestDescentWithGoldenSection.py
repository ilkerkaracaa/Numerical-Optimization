import numpy as np
import math
from sinav import f,gradient

def GSf(sk,xk,pk):
    sonuc = f(xk+sk*pk)
    return sonuc

def GS(xk,pk):
    salt = 0
    sust = 1
    ds = 0.0001
    alpha = (1+math.sqrt(5))/2
    tau = 1-1/alpha
    epsilon = ds/(sust-salt)
    N = round(-2.078*math.log(epsilon))
    k = 0
    s1 = salt + tau*(sust-salt); f1 = GSf(s1,xk,pk);
    s2 = sust - tau*(sust-salt); f2 = GSf(s2,xk,pk);
    while abs(s1-s2)>ds:
        k += 1
        if f1 > f2:
            salt = 1*s1; s1 = 1*s2; f1 = 1*f2;
            s2 = sust - tau*(sust-salt); f2 = GSf(s2,xk,pk);
        else:
            sust = 1*s2; s2 = 1*s1; f2 = 1*f1;
            s1 = salt + tau*(sust-salt); f1 = GSf(s1,xk,pk);
    s = np.mean([s1,s2])
    return s

MaxIter = 100000
epsilon1 = 1e-9
epsilon2 = 1e-9
epsilon3 = 1e-9

x1 = [0]
x2 = [0]
xk = np.array([x1[0],x2[0]])
k = 0 
C1 = True; C2 = True; C3 = True; C4 = True;
while C1 & C2 & C3 & C4:
    k += 1
    pk = -gradient(xk)
    sk = GS(xk,pk)
    # sk = 1
    xk = xk + sk*pk
    x1.append(xk[0])
    x2.append(xk[1])
    print('k:',k,'x1:',format(xk[0],'f'),'x2:',format(xk[1],'f'),'f(x):',format(f(xk),'f'))
    C1 = k < MaxIter
    C2 = epsilon1 < abs(f(xk)-f(xk+sk*pk))
    C3 = epsilon2 < np.linalg.norm(sk*pk)
    C4 = epsilon3 < np.linalg.norm(gradient(xk))

if not C1:
    print('Max iterasyon sayısına ulaşıldı.')
if not C2:
    print('fonksiyonun değeri değişmiyor')
if not C3:
    print('ilerleme yönü bulunamıyor')
if not C4:
    print('gradient sıfıra yakın')

