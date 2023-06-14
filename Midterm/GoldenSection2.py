import math
import numpy as np

def f(x,p,s):
    x = x+s*p
    f = 3 + (x[0]-1.5*x[1])**2 + (x[0]-2)**2
    return f

salt = 0
sust = 4
ds = 0.0001
alpha = (1+math.sqrt(5))/2
tau = 1-1/alpha
epsilon = ds/(sust-salt)
N = round(-2.078*math.log(epsilon))
k = 0
s1 = salt + tau*(sust-salt); f1 = f(s1);
s2 = sust - tau*(sust-salt); f2 = f(s2);

print(0,s1,s2,f1,f2)

while abs(s1-s2)>ds:  
    if f1 > f2:
        salt = 1*s1; s1 = 1*s2; f1 = 1*f2;
        s2 = sust - tau*(sust-salt); f2 = f(s2);
    else:
        sust = 1*s2; s2 = 1*s1; f2 = 1*f1;
        s1 = salt + tau*(sust-salt); f1 = f(s1); 
    print(k+1,s1,s2,f1,f2)
    k = k+1

s = np.mean([s1,s2])
print("x = ",s)