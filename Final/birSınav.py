import numpy as np

t = np.array([-2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6])
y = np.array([-1.5, 1.30, 2.64, 2.94, 2.62, 2.10, 1.80, 2.15, 3.56, 6.47])

A = np.vstack([t*3, t*2, t, np.ones(len(t))]).T
a3,a2,a1,a0 = np.linalg.lstsq(A,y,rcond=None)[0]

print(f"Üçüncü-dereceden polinom: {a3:.4f}x^3 + {a2:.4f}x^2 + {a1:.4f}x + {a0:.4f}")