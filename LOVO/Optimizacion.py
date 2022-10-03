import numpy as np
import Function as TestFunction
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt


def check_tolerance(value, value_new, epislon=1e-4):
    """
    Evaluación en los criterios de paro por distancias relativas si está bien definido el cociente.
    Evaluación en los criterios de paro por distancias absolutas, en otro caso.

    :param value: valor anterior.
    :param value_new: valor nuevo.
    :param epislon: tolerancia.
    :return: Verdadero si la evalución es menor a la tolerancia. Falso en otro caso.
    """
    # Evaluación para escalares
    if isinstance(value, (float, np.float)) and isinstance(value_new, (float, np.float)):
        return np.abs(value - value_new) / max(1., np.abs(value)) < epislon

    # Evaluación para vectores
    else:
        return np.linalg.norm(value - value_new) / max(1., np.linalg.norm(value)) < epislon


def compute_direction(J: np.ndarray, gamma: float, g: np.ndarray):
    A = np.matmul(J.T, J) + np.diag([gamma] * J.shape[1])
    try:
        # print("gradient: ", g)
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        print('\'A\' cholesky failed')
        return -g, 1
    else:
        y = solve_triangular(L, -g, lower=True)
        d = solve_triangular(L.T, y, lower=False)
        return d, 0


def lm_lovo(x_0: np.ndarray, lmbda_min: float, epsilon: float, lmbda_0: float,
            lmbda_hat: float, test_function: TestFunction, max_iter: int = 400):
    assert (lmbda_0 > 0. and lmbda_hat > 1.)

    # Initializaton
    lmbda = lmbda_0
    x = x_0

    for k in range(max_iter):
        # get I_min (combination where first p values of Ri are the lowest)
        test_function.getR(x)
        g = test_function.gradient(x)

        # Stoppage criteria
        g_norm = test_function.norm_g_k[-1]

        # Report
        f = test_function.S(x)
        if k % 4000 == 0:
            print(" ")
            print("Iteration n.", k, "results:")
            print("|g(x)| = ", g_norm)
            print("f(x) = ", f)

        if check_tolerance(g_norm, 0., epsilon):
            break

        J = test_function.Jacobian(x)
        while True:
            # Calculate direction
            gamma = lmbda * g_norm * g_norm
            d, reset = compute_direction(J, gamma, g)

            # Cholesky failed
            if reset:
                print("Error in iteration:", k)
                np.random.seed(k)
                x = np.random.normal(size=x.shape[0])
                lmbda = 1.
                break

            # Simple decrease test: (trust-region simplification)
            if f > test_function.S(x + d, False) or check_tolerance(np.linalg.norm(d), 0.):
                break
            else:
                lmbda = lmbda * lmbda_hat

        # Actualization
        lmbda = lmbda/lmbda_hat # max(lmbda_min, lmbda / lmbda_hat)  # TODO: in [max(lmbda_min, lmbda/np.sqrt(lmbda)), lmbda]
        x = x + d

    return x, g_norm


def buildSimilarityMatrix(solutions, rem_indexes):
    n = len(solutions)
    M = np.zeros((n, n))

    # build similarity matrix taking advantage of its symmetry
    for i in range(n):

        # remove element
        if i in rem_indexes:
            M[i, :] = np.inf
            M[:, i] = np.inf
            continue

        for j in range(i):
            x_p, x_q = solutions[i][0], solutions[j][0]
            M[i][j] = np.linalg.norm(x_p - x_q)
            M[j][i] = M[i][j]

    return M


# S and abs_diff are sorted from min_p to max_p
def preprocess(solutions, S, abs_diff, r, tau):
    indexes = []
    p_min = 0
    S_min = S[0]

    # add indexes that should be eliminated
    for i in range(len(S)):

        # check if lovo did not converge
        if solutions[i][1] > tau:
            indexes.append(i)
            continue

        for j in range(i):
            if S[j] > S[i]:
                indexes.append(j)

        # retrieve model such that we have minimum Sp(xp^*)
        if S[i] < S_min and i not in indexes:
            S_min = S[i]
            p_min = i

    # get model with lower Sp and model from p_max
    f_mins = abs_diff[p_min]
    f_maxp = abs_diff[-1]

    # retrieve difference for each point in the dataset between y_i predicted and real y_i for f_mins model and f_maxp model
    diff_mins_maxp = f_mins - f_maxp

    # see how many of previous differences are lower than 0
    k = np.sum(diff_mins_maxp < 0)

    # if k > r/2, then max_p model should be eliminated
    if k > r / 2:
        indexes.append(len(solutions) - 1)

    return indexes


def RAFF(x0: np.ndarray, f_model: TestFunction, pmin, pmax, epsilon=-1, lambda_min=0.1, lambda_0=1, lambda_hat=2,
         tau=1e-4, max_iter=400, real_trust=None):
    assert (1 <= pmin < pmax <= len(f_model.dataset))

    S = []  # vector of Sp
    abs_diff = []  # vector of abs diff between yi and model(x, ti)
    solutions = []  # vector of tuples (x^*, ||grad(x^*)||) for each pmin <= p <= pmax
    trusted_indexes = []  # vector of trusted indexes for each p

    # compute set of solutions for p in range(pmin, pmax)
    for p in range(pmin, pmax):
        print('-' * 90)
        print("RAFF p = ", p)
        f_model.p = p

        # find optimum
        x_p, g_norm = lm_lovo(x0, lambda_min, tau, lambda_0, lambda_hat, f_model, max_iter=max_iter)
        solutions.append((x_p, g_norm))
        S.append(f_model.S(x_p))
        abs_diff.append(f_model.abs_diff(x_p))
        classified_as_trust = f_model.R_index[:p, 1].astype(int)
        trusted_indexes.append(classified_as_trust)
        # plot_trusted(real_trust, classified_as_trust, f_model.r, f_model.dataset, x_p)

        print("x_", p, ":", x_p)

    # solutions = np.array(solutions)

    # preprocess solutions
    rem_indexes = preprocess(solutions, S, abs_diff, r=len(f_model.dataset), tau=tau)

    # build similarity matrix
    M = buildSimilarityMatrix(solutions, rem_indexes)
    C = np.zeros(len(M))

    # calc epsilon if it is not given
    M_aux = M[np.isfinite(M)]

    if epsilon == -1:
        epsilon = np.min(M_aux) + np.mean(M_aux) / (1 + np.sqrt(pmax)) if len(M_aux) > 0 else 0

    for i in range(len(M)):
        k = 0
        for j in range(len(M)):
            if M[i][j] < epsilon:
                k += 1
        C[i] = k

    # see position of solutions that have more votes and return that x^*
    max_k, max_p = 0, 0
    for p in range(len(C)):
        if max_k <= C[p]:
            max_k = C[p]
            max_p = p

    # return x^* and maxp number of trusted points
    return solutions[max_p][0], trusted_indexes[max_p]

def plot_trusted(real_trust, classified_as_trust, r, dataset, xopt):
    all_idx = np.arange(r)
    real_outliers = [idx for idx in all_idx if idx not in real_trust]
    classified_as_outliers = [idx for idx in all_idx if idx not in classified_as_trust]

    real_trust_classified_as_trust = np.intersect1d(real_trust, classified_as_trust)
    real_outliers_classified_as_outliers = np.intersect1d(real_outliers, classified_as_outliers)
    real_trust_classified_as_outliers = [idx for idx in real_trust if idx not in classified_as_trust]
    real_outliers_classified_as_trust = [idx for idx in real_outliers if idx in classified_as_trust]

    mask_idx = np.ones(r)
    mask_idx[real_outliers] = 0
    markers = ['o' if idx in real_trust else '^' for idx in all_idx]
    colors = ['g' if idx in real_outliers_classified_as_outliers or
                     idx in real_trust_classified_as_trust else 'b' for idx in all_idx]

    for m, c, x, y in zip(markers, colors, dataset[:, 0], dataset[:, -1]):
        plt.scatter(x, y, marker=m, c=c)

    # new_dataset = model.dataset[classified_as_trust, :]
    # x_model, y_model = new_dataset[:, 0], gen_model(new_dataset[:, 0].reshape((len(classified_as_trust), 1)), xopt)
    #
    # # sort model_data by x coordinate
    # model_data = np.column_stack((x_model, y_model))
    # model_data = model_data[np.argsort(model_data[:, 0])]
    #
    # plt.plot(model_data[:, 0], model_data[:, 1], color='blue')
    # plt.title("Proposed model based in trusted points")
    plt.show()
