# import numpy as np
# from scipy.sparse import random, diags

# def genrand(m, n, d):
#     c = np.zeros(n)
#     A = np.array([])
#     b = np.array([])

#     pl = lambda x: (np.abs(x) + x) / 2

#     while np.any(c >= 0):
#         A = random(m, n, density=d, format='csr', random_state=42)
#         A.data *= 50
#         x = 10 * pl(np.random.rand(n))
#         u = diags((np.sign(pl(np.random.rand(m) - np.random.rand(m)))), 0) @ (np.random.rand(m) - np.random.rand(m))
#         b = A @ x
#         c = A.transpose() @ u + diags((np.ones((n)) - np.sign(pl(x))), 0) @ (10 * np.ones((n)))

#     f = c.transpose()
#     A = A.toarray()
#     b = np.transpose(b)
#     ppl = np.block([[A, np.eye(m), b.reshape(-1,1)], [f, np.zeros((1, m+1))]])

#     return ppl, f, A, b


import numpy as np
from scipy.sparse import random as sprand
from scipy.sparse import csr_matrix

def genrand(m, n, d):
    # Generate a sparse matrix A with density d
    A = sprand(m, n, density=d, format='csr') * 50  # Scale to adjust range of values

    # Generate x with non-negative elements
    x = csr_matrix(np.maximum(0, 10 * np.random.rand(n, 1)))

    # Generate b as a product of A and x
    b = A.dot(x)

    # Ensure b is non-negative by adjusting negative elements
    b = b.multiply(b > 0) + b.multiply(b < 0).multiply(-1)

    # Generate a random vector u for cost vector calculation
    u = np.random.rand(m, 1) * 2 - 1  # Uniform distribution between -1 and 1

    # Generate cost vector c ensuring it's non-zero
    c = A.transpose().dot(u)
    if np.all(c == 0):
        c[np.random.randint(c.size)] = np.random.rand()  # Ensure at least one non-zero element

    # Combine into the simplex tableau format
    f = -c.transpose()  # Objective function (-c input for maximization problem)
    ppl = np.vstack([np.hstack([A.toarray(), np.eye(m), b.toarray()]), np.hstack([f, np.zeros((1, m + 1))])])
    return ppl, f, A, b