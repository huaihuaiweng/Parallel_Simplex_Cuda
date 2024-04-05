import numpy as np
from scipy.sparse import random, diags

def genrand(m, n, d):
    c = np.zeros(n)
    A = np.array([])
    b = np.array([])

    pl = lambda x: (np.abs(x) + x) / 2

    while np.any(c >= 0):
        A = random(m, n, density=d, format='csr', random_state=42)
        A.data *= 50
        x = 10 * pl(np.random.rand(n))
        u = diags((np.sign(pl(np.random.rand(m) - np.random.rand(m)))), 0) @ (np.random.rand(m) - np.random.rand(m))
        b = A @ x
        c = A.transpose() @ u + diags((np.ones((n)) - np.sign(pl(x))), 0) @ (10 * np.ones((n)))

    f = c.transpose()
    A = A.toarray()
    b = np.transpose(b)
    ppl = np.block([[A, np.eye(m), b.reshape(-1,1)], [f, np.zeros((1, m+1))]])

    return ppl, f, A, b
