import numpy as np
from typing import List, Tuple

# Constants
MAX_VERTICES = 20
MAX_LENGTH = 20
MAX_EDGES = 20
MAX_SYMBOLS = 20
MAX_DEPTH = 5
MAX_STATES = MAX_DEPTH + 1
MIN_DOUBLE = 1e-80
MIN_FLOAT = 1e-10
EPSILON = 1

class HMM:
    def __init__(self, N: int = 0, M: int = 0):
        self.N = N  # Number of states
        self.M = M  # Number of symbols
        self.A = np.zeros((MAX_STATES, MAX_STATES))  # Transition matrix
        self.B = np.zeros((MAX_STATES, MAX_SYMBOLS))  # Emission matrix
        self.Pi = np.zeros(MAX_STATES)  # Initial state distribution
        self.Phi = np.zeros(MAX_STATES)  # Stationary distribution

def adjust_hmm(lambda_hmm: HMM) -> None:
    for i in range(lambda_hmm.N):
        for j in range(lambda_hmm.N):
            if lambda_hmm.A[i, j] < MIN_FLOAT:
                lambda_hmm.A[i, j] = 0.0
        for j in range(lambda_hmm.M):
            if lambda_hmm.B[i, j] < MIN_FLOAT:
                lambda_hmm.B[i, j] = 0.0
        if lambda_hmm.Pi[i] < MIN_FLOAT:
            lambda_hmm.Pi[i] = 0.0
        if lambda_hmm.Phi[i] < MIN_FLOAT:
            lambda_hmm.Phi[i] = 0.0

def copy_hmm(lambda1: HMM, lambda2: HMM) -> None:
    lambda1.N = lambda2.N
    lambda1.M = lambda2.M
    lambda1.Pi[:lambda2.N] = lambda2.Pi[:lambda2.N]
    lambda1.Phi[:lambda2.N] = lambda2.Phi[:lambda2.N]
    lambda1.A[:lambda2.N, :lambda2.N] = lambda2.A[:lambda2.N, :lambda2.N]
    lambda1.B[:lambda2.N, :lambda2.M] = lambda2.B[:lambda2.N, :lambda2.M]

def save_hmm_txt(lambda_hmm: HMM, output_txt_file: str) -> None:
    with open(output_txt_file, 'w') as f:
        f.write(f"\nHMM parameters: {lambda_hmm.N} states and {lambda_hmm.M} symbols\n\n")
        f.write("Matrix A :\n\n")
        for i in range(lambda_hmm.N):
            f.write("\t".join(f"{x:.6g}" for x in lambda_hmm.A[i, :lambda_hmm.N]) + "\n")
        f.write("\n\nMatrix B :\n\n")
        for i in range(lambda_hmm.N):
            f.write("\t".join(f"{x:.6g}" for x in lambda_hmm.B[i, :lambda_hmm.M]) + "\n")
        f.write("\n\nVector Pi :\n\n")
        f.write("\t".join(f"{x:.6g}" for x in lambda_hmm.Pi[:lambda_hmm.N]) + "\n")
        f.write("\n\nVector Phi :\n\n")
        f.write("\t".join(f"{x:.6g}" for x in lambda_hmm.Phi[:lambda_hmm.N]) + "\n")

def copy_matrix(m1: np.ndarray, m2: np.ndarray, nb_row: int, nb_col: int) -> None:
    m1[:nb_row, :nb_col] = m2[:nb_row, :nb_col]

def stationary_distribution(lambda_hmm: HMM) -> None:
    n = lambda_hmm.N
    a_tem = lambda_hmm.A[:n, :n].copy()
    a_product = np.zeros((n, n))
    for _ in range(101):  # Iterate up to 100 times
        a_product.fill(0)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    a_product[i, j] += a_tem[i, k] * lambda_hmm.A[k, j]
        a_tem = a_product.copy()
    lambda_hmm.Phi[:n] = a_product[0, :n]

def forward(lambda_hmm: HMM, o: List[int], t_bar: int) -> np.ndarray:
    alpha = np.zeros((MAX_LENGTH, MAX_STATES), dtype=np.longdouble)
    for j in range(lambda_hmm.N):
        alpha[0, j] = lambda_hmm.B[j, o[0]] * lambda_hmm.Pi[j]
    for t in range(t_bar - 1):
        for j in range(lambda_hmm.N):
            sum_alpha = sum(alpha[t, i] * lambda_hmm.A[i, j] for i in range(lambda_hmm.N))
            alpha[t + 1, j] = sum_alpha * lambda_hmm.B[j, o[t + 1]]
    return alpha

def backward(lambda_hmm: HMM, o: List[int], t_bar: int, T: int) -> np.ndarray:
    beta = np.zeros((MAX_LENGTH, MAX_STATES), dtype=np.longdouble)
    beta[T - 1, :lambda_hmm.N] = 1
    for t in range(T - 1, t_bar - 1, -1):
        for i in range(lambda_hmm.N):
            sum_beta = sum(
                lambda_hmm.A[i, j] * lambda_hmm.B[j, o[t]] * beta[t, j]
                for j in range(lambda_hmm.N)
            )
            beta[t - 1, i] = sum_beta
    return beta

def forward_backward(lambda_hmm: HMM, o: List[int], t_bar: int, T: int) -> float:
    alpha = forward(lambda_hmm, o, T)
    beta = backward(lambda_hmm, o, 1, T)
    proba_observ = sum(alpha[t_bar, j] * beta[t_bar, j] for j in range(lambda_hmm.N))
    return float(proba_observ)

def calcul_xi(lambda_hmm: HMM, o: List[int], T: int, alpha: np.ndarray, beta: np.ndarray, proba_observ: float) -> np.ndarray:
    xi = np.zeros((MAX_LENGTH, MAX_STATES, MAX_STATES), dtype=np.longdouble)
    for t in range(T - 1):
        for i in range(lambda_hmm.N):
            for j in range(lambda_hmm.N):
                xi[t, i, j] = (
                    alpha[t, i] * lambda_hmm.A[i, j] * lambda_hmm.B[j, o[t + 1]] * beta[t + 1, j]
                ) / (proba_observ + MIN_DOUBLE)
    return xi

def calcul_gamma(lambda_hmm: HMM, o: List[int], T: int, alpha: np.ndarray, beta: np.ndarray, proba_observ: float) -> np.ndarray:
    gamma = np.zeros((MAX_LENGTH, MAX_STATES), dtype=np.longdouble)
    for t in range(T):
        for j in range(lambda_hmm.N):
            gamma[t, j] = (alpha[t, j] * beta[t, j]) / (proba_observ + MIN_DOUBLE)
    return gamma

def baum_welch(
    lambda_init: HMM,
    O: List[List[int]],
    T: List[int],
    epsilon: float,
    max_iterations: int,
    nb_mcs: int,
    threshold: float
) -> Tuple[HMM, int]:
    lambda_old = HMM()
    lambda_new = HMM()
    copy_hmm(lambda_old, lambda_init)

    alpha = np.zeros((MAX_VERTICES, MAX_LENGTH, MAX_STATES), dtype=np.longdouble)
    beta = np.zeros((MAX_VERTICES, MAX_LENGTH, MAX_STATES), dtype=np.longdouble)
    for k in range(nb_mcs):
        alpha[k] = forward(lambda_old, O[k], T[k])
        beta[k] = backward(lambda_old, O[k], 1, T[k])

    iterations = 0
    while True:
        iterations += 1
        proba_old = np.zeros(MAX_VERTICES)
        gamma = np.zeros((MAX_VERTICES, MAX_LENGTH, MAX_STATES), dtype=np.longdouble)
        xi = np.zeros((MAX_VERTICES, MAX_LENGTH, MAX_STATES, MAX_STATES), dtype=np.longdouble)
        for k in range(nb_mcs):
            t_bar = np.random.randint(0, T[k])
            proba_old[k] = sum(alpha[k, t_bar, j] * beta[k, t_bar, j] for j in range(lambda_old.N))
            gamma[k] = calcul_gamma(lambda_old, O[k], T[k], alpha[k], beta[k], proba_old[k])
            xi[k] = calcul_xi(lambda_old, O[k], T[k], alpha[k], beta[k], proba_old[k])

        lambda_new.N = lambda_old.N
        lambda_new.M = lambda_old.M
        lambda_new.Pi[:lambda_old.N] = np.mean(gamma[:nb_mcs, 0, :lambda_old.N], axis=0)

        for i in range(lambda_old.N):
            for j in range(lambda_old.N):
                sum_xi = sum(xi[k, t, i, j] for k in range(nb_mcs) for t in range(T[k] - 1))
                sum_gamma = sum(gamma[k, t, i] for k in range(nb_mcs) for t in range(T[k] - 1))
                lambda_new.A[i, j] = sum_xi / (sum_gamma + MIN_DOUBLE)

        for j in range(lambda_old.N):
            for l in range(lambda_old.M):
                s = sum(
                    gamma[k, t, j] for k in range(nb_mcs) for t in range(T[k]) if O[k][t] == l
                )
                sum_gamma = sum(gamma[k, t, j] for k in range(nb_mcs) for t in range(T[k]))
                lambda_new.B[j, l] = s / (sum_gamma + MIN_DOUBLE)

        alpha_bar = np.zeros((MAX_VERTICES, MAX_LENGTH, MAX_STATES), dtype=np.longdouble)
        beta_bar = np.zeros((MAX_VERTICES, MAX_LENGTH, MAX_STATES), dtype=np.longdouble)
        for k in range(nb_mcs):
            alpha_bar[k] = forward(lambda_new, O[k], T[k])
            beta_bar[k] = backward(lambda_new, O[k], 1, T[k])

        rate = 0.0
        proba_new = np.zeros(MAX_VERTICES)
        for k in range(nb_mcs):
            proba_new[k] = sum(alpha_bar[k, t_bar, j] * beta_bar[k, t_bar, j] for j in range(lambda_new.N))
            if abs(proba_new[k] - proba_old[k]) <= epsilon:
                rate += 1.0
        rate = (rate / nb_mcs) * 100

        if rate < threshold:
            copy_hmm(lambda_old, lambda_new)
            alpha = alpha_bar.copy()
            beta = beta_bar.copy()
        else:
            break

        if iterations > max_iterations:
            break

    stationary_distribution(lambda_new)
    adjust_hmm(lambda_new)
    return lambda_new, iterations

def proba_symbol_long_period(k: int, lambda_hmm: HMM) -> float:
    return sum(lambda_hmm.Phi[i] * lambda_hmm.B[i, k] for i in range(lambda_hmm.N))

def vector_hmm(lambda_hmm: HMM) -> np.ndarray:
    v = np.zeros(MAX_SYMBOLS)
    for k in range(lambda_hmm.M):
        v[k] = proba_symbol_long_period(k, lambda_hmm)
    return v