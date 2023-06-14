import numpy as np
import math

def f(x):
    f = (x1-25)**2 + (x2-36)**2 + 9*x1**2 + 4*x1**2*x2**2
    return f

def g(x):
    g = np.array([8*x1*x2**2+20*x1-50,8*x1**2*x2+2*x2-72])
    return g

def GSf(x,s,p):
    x = x+s*p
    sonuc = f(x)
    return sonuc

def GS(x,p,salt,sust,ds):
    alpha = (1+math.sqrt(5))/2
    tau = 1-1/alpha
    epsilon = ds/(sust-salt)
    N = round(-2.078*math.log(epsilon))
    k = 0
    s1 = salt + tau*(sust-salt); f1 = GSf(s1,x,p);
    s2 = sust - tau*(sust-salt); f2 = GSf(s2,x,p);
    while abs(s1-s2)>ds:  
        if f1 > f2:
            salt = 1*s1; s1 = 1*s2; f1 = 1*f2;
            s2 = sust - tau*(sust-salt); f2 = GSf(s2,x,p);
        else:
            sust = 1*s2; s2 = 1*s1; f2 = 1*f1;
            s1 = salt + tau*(sust-salt); f1 = GSf(s1,x,p); 
    print(k+1,s1,s2,f1,f2)
    k = k+1
    s = np.mean([s1,s2])
    return s


x1 = -4.5
x2 = -3.5

print(f([x1,x2]))
print(g([x1,x2]))

x = np.array([x1,x2])
for i in range(0,500):
    p = -g(x)
    s = GS(x,p,0,1,0.0001)
    x = x+s*p
    print(i,x[0],x[1],f(x)) 