import numpy as np


class TestFunction(object):
    """
    Se recomienda ampliamente que _todo se trabaje de forma vectorizada.
    Trate de evitar lo más que se pueda los búcles cuando se trata de
    evaluar función objetivo, gradiente o hessinano.
    """

    def __init__(self, n: int):
        self.n = n
        self.f_k = []
        self.norm_g_k = []

    def function(self, x: np.ndarray, keep_record=True):
        raise NotImplementedError

    def gradient(self, x: np.ndarray, keep_record=True):
        raise NotImplementedError

    def hessian(self, x: np.ndarray):
        raise NotImplementedError


class Quadratic(TestFunction):

    def __init__(self, n: int, n_cond: float, m: int = 3):

        super().__init__(n)

        # Construcción de Q:
        self.Q = constructQ(n, n_cond, m)

        # Construcción de b:
        self.x_star = np.random.uniform(-1., 1., size=(n,))
        self.b = self.Q @ self.x_star

    def function(self, x: np.ndarray, keep_record=True):

        f_k = 0.5 * np.dot(x, self.Q @ x) - np.dot(self.b, x)

        if keep_record:
            self.f_k.append(float(f_k))

        return f_k

    def gradient(self, x: np.ndarray, keep_record=True):

        g_k = self.Q @ x - self.b

        if keep_record:
            norm_gk = np.linalg.norm(g_k)
            self.norm_g_k.append(norm_gk)

        return g_k

    def hessian(self, x: np.ndarray):

        return self.Q @ x



class SoftNoise(TestFunction):

    def __init__(self, g: np.ndarray, lmbda: float, mu: float = 0.01):
        g_vec = g.flatten()
        n = g_vec.shape[0]
        super().__init__(n)
        self.nrows, self.ncols = g.shape[0], g.shape[1]
        self.g_vec = g_vec
        self.g = g
        self.lmbda = lmbda
        self.mu = mu

    def function(self, x: np.ndarray, keep_record=True):

        f_k = np.sum((x - self.g_vec) ** 2.)  # Primera suma
        f_k += self.lmbda * np.sum(np.sqrt((x[1:] - x[:-1]) ** 2. + self.mu))  # Vecinos derecha
        f_k += self.lmbda * np.sum(np.sqrt((x[:-1] - x[1:]) ** 2. + self.mu))  # Vecinos izquierda
        f_k += self.lmbda * np.sum(np.sqrt((x[self.ncols:] - x[:-self.ncols]) ** 2. + self.mu))  # Vecinos arriba
        f_k += self.lmbda * np.sum(np.sqrt((x[:-self.ncols] - x[self.ncols:]) ** 2. + self.mu))  # Vecinos abajo

        if keep_record:
            self.f_k.append(float(f_k))

        return f_k

    def gradient(self, x: np.ndarray, keep_record=True):

        x_mat = x.reshape(self.nrows, self.ncols)
        x_mat_expand = np.pad(x_mat, 1, mode='constant')
        second_term = np.zeros_like(x)

        n, m = x_mat_expand.shape[0], x_mat_expand.shape[1]
        k = 0

        for i in range(n):

            for j in range(m):

                if 0 < j < m - 1 and 0 < i < n - 1:
                    x_ij = x_mat_expand[i, j]

                    dif = x_mat_expand[i + 1, j] - x_ij
                    second_term[k] += dif / np.sqrt(dif**2 + self.mu)

                    dif = x_mat_expand[i - 1, j] - x_ij
                    second_term[k] += dif / np.sqrt(dif**2 + self.mu)

                    dif = x_mat_expand[i, j + 1] - x_ij
                    second_term[k] += dif / np.sqrt(dif** 2 + self.mu)

                    dif = x_mat_expand[i, j - 1] - x_ij
                    second_term[k] += dif / np.sqrt(dif**2 + self.mu)

                    k += 1

        g_k = 2. * (x - self.g_vec) + 2. * self.lmbda * second_term

        if keep_record:
            norm_gk = np.linalg.norm(g_k)
            self.norm_g_k.append(norm_gk)

        return g_k

