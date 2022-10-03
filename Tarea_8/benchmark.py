import numpy as np
from data import Data


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

    def error(self, beta_star):
        raise NotImplementedError


def sgm(beta_beta_0: np.ndarray, x_1: np.ndarray):
    """
    :param x_1: matriz con observaciones y con 1's en la última columna
    :param beta_beta_0: beta con beta_0 concatenado al final
    :return: su evaluación en la función sigmoide

    """
    """
    if -value > np.log(np.finfo(type(value)).max):
        return 0.0    
    a = np.exp(-value)
    return 1.0/ (1.0 + a)
    """

    m = x_1.dot(beta_beta_0)
    sigmoid = np.zeros_like(m)

    to_eval = -m <= np.log(np.finfo(m.dtype).max)
    sigmoid[to_eval] = 1. / (1. + np.exp(-m[to_eval]))

    return sigmoid


class LogLikeLR(TestFunction):

    def __init__(self, n: int, data: Data):
        super().__init__(n)
        self.data = data

        # etiquetas del entrenamiento
        self.y = data.y_tr

        # A cada observación se le concatena un 1 al final, para faciltar los cálculos en la función sigmoide
        ones = np.ones(shape=(data.nrow_tr, 1), dtype=self.data.x_tr.dtype)

        self.x_1 = np.hstack((data.x_tr, ones))

        # transpone la matriz anterior para facilitar el cálculo del gradiente
        self.x_1_t = np.transpose(self.x_1)

    def function(self, x: np.ndarray, keep_record=True):
        """
        :param x: es beta concatenado con beta_0
        :param keep_record: para guardar todas las evaluaciones
        :return: evaluación en la función de log-likelihood de la regresión logística
        para clasificación de dos clases

        """
        pi = sgm(x, self.x_1)

        log_pi = np.log(pi + 10 ** -10)
        not_inf = np.logical_not(np.isinf(log_pi))
        f_k_pi = np.sum(self.y[not_inf] * log_pi[not_inf])  # Segundos términos de la suma

        log_1_pi = np.log(1. - pi + 10 ** -10)
        not_inf = np.logical_not(np.isinf(log_1_pi))
        f_k_1_pi = np.sum((1. - self.y[not_inf]) * log_1_pi[not_inf])  # Segundos términos de la suma
        f_k = -(f_k_pi + f_k_1_pi) / pi.size
        f_k = f_k / self.y.size

        acc = np.equal(np.around(pi), np.squeeze(self.y)).mean()
        print('f(x)', f_k, 'acc:', acc)
        # Registro de evolución de valores
        if keep_record:
            self.f_k.append(float(f_k))

        return f_k

    def gradient(self, x: np.ndarray, keep_record=True):
        """
        :param x: es beta concatenado con beta_0
        :param keep_record: para guardar todas las evaluaciones
        :return: evaluación en el gradiente de la función de log-likelihood
        de la regresión logística para clasificación de dos clases

        """

        pi_beta = np.array(sgm(x, self.x_1)).reshape(-1, 1)

        dif = pi_beta - self.y
        g_k = np.matmul(self.x_1_t, dif) / self.y.size

        if keep_record:
            norm_gk = np.linalg.norm(g_k)
            # print(norm_gk)
            self.norm_g_k.append(norm_gk)

        return g_k.reshape(-1, )

    def error(self, beta_star):

        ones = np.ones(shape=(self.data.nrow_ts, 1))
        x_ts_1 = np.hstack((self.data.x_ts, ones))

        pi = sgm(beta_star, x_ts_1)

        pred = np.array([1. if y_i > 0.5 else 0. for y_i in pi])

        true = self.data.y_ts

        n = true.size

        err = 1. / n * np.sum(pred - true)

        return err


class ModelWeightsGaussianMix(TestFunction):

    def __init__(self, n: int, data: Data, sigma, means: np.ndarray):
        super().__init__(n)
        self.ngauss = n
        self.n = n
        self.dev_std = sigma ** 2
        self.rescale = -(1. / (2. * self.dev_std))
        self.mus = means
        self.bins = data.bins
        self.hist = data.hist
        self.cs = data.c

    def function(self, x: np.ndarray, keep_record=False):
        """
        :param x: coeficientes de mezclas/pesos (alphas)
        :param keep_record: para guardar todas las evaluaciones
        :return: evaluación en la función de pérdida de ajuste de histrogramas
        por bases radiales

        """
        f = 0.

        for c in self.cs:

            c_centered_norm = np.linalg.norm(c - self.mus.reshape(self.ngauss, -1), axis=1)

            c_centered_sqr_norm_std = c_centered_norm ** 2 / (2. * self.dev_std)

            exp_neg_std_norms = np.exp(-c_centered_sqr_norm_std)

            weightened_sum = np.sum(x * exp_neg_std_norms)

            f += (self.hist[c] - weightened_sum) ** 2

        # Registro de evolución de valores
        if keep_record:
            self.f_k.append(float(f))

        return f

    def gradient(self, x: np.ndarray, keep_record=False):
        """
        :param x: de mezclas/pesos (alphas)
        :param keep_record: para guardar todas las evaluaciones
        :return: evaluación en el gradiente de la función de pérdida de ajuste de histrogramas
        por bases radiales

        """
        g = np.zeros_like(x)

        for c in self.cs:
            c_centered_norm = np.linalg.norm(c - self.mus.reshape(self.ngauss, -1), axis=1)

            c_centered_norm_std = c_centered_norm ** 2 / (2. * self.dev_std)

            exp_neg_std_norms = np.exp(-c_centered_norm_std)

            weightened_sum = np.sum(x * exp_neg_std_norms)

            first_factor = self.hist[c] - weightened_sum

            g += first_factor * exp_neg_std_norms

        g = -2. * g

        if keep_record:
            norm_gk = np.linalg.norm(g)
            self.norm_g_k.append(norm_gk)

        return g.reshape(-1, )


class ModelMeansGaussianMix(TestFunction):

    def __init__(self, n: int, data: Data, sigma, weights: np.ndarray):
        super().__init__(n)
        self.ngauss = n
        self.n = 3*n
        self.dev_std = sigma ** 2
        self.rescale = -(1. / (2. * self.dev_std))
        self.alphas = weights
        self.bins = data.bins
        self.hist = data.hist
        self.cs = data.c

    def function(self, x: np.ndarray, keep_record=False):
        """
        :param x: arreglo de promedios, uno por cada gaussiana
        :param keep_record: para guardar todas las evaluaciones
        :return: evaluación en la función de pérdida de ajuste de histrogramas
        por bases radiales

        """
        f = 0.

        for c in self.cs:
            c_centered_norm = np.linalg.norm(c - x.reshape(self.ngauss, -1), axis=1)

            c_centered_norm_std = c_centered_norm ** 2 / (2. * self.dev_std)

            exp_neg_std_norms = np.exp(-c_centered_norm_std)

            weightened_sum = np.sum(self.alphas * exp_neg_std_norms)

            f += (self.hist[c] - weightened_sum) ** 2

        # Registro de evolución de valores
        if keep_record:
            self.f_k.append(float(f))

        return f

    def gradient(self, x: np.ndarray, keep_record=False):
        """
        :param x: arreglo de promedios, uno por cada gaussiana
        :param keep_record: para guardar todas las evaluaciones
        :return: evaluación en el gradiente de la función de pérdida de ajuste de histrogramas
        por bases radiales

        """
        g = []

        # Cálculo de gradientes respecto a cada mu_k, k = 1, . . .,n
        # (El gradiente buscado es igual a la traspuesta de respecto a cada mu_k, k = 1, . . .,n)

        # Factores en común de los sumandos de todos los g_mu_k

        first_factor_c = []

        for c in self.cs:
            c_centered_norm = np.linalg.norm(c - x.reshape(self.ngauss, -1), axis=1)

            c_centered_sqr_norm_std = c_centered_norm ** 2 / (2. * self.dev_std)

            exp_neg_std_norms = np.exp(-c_centered_sqr_norm_std)

            weightened_sum = np.sum(self.alphas * exp_neg_std_norms)

            first_factor_c.append(self.hist[c] - weightened_sum)

        first_factor_c = -2. * np.array(first_factor_c)

        for k, mu in enumerate(x.reshape(-1, 3)):

            sqr_norm_dif = (np.linalg.norm(self.cs - mu, axis=1)) ** 2

            coef = self.alphas[k] * np.exp(self.rescale * sqr_norm_dif) * (1. / self.dev_std)

            second_factor_c = coef[:, np.newaxis] * (self.cs - mu)

            grad_mu_k = np.sum(first_factor_c[:, np.newaxis] * second_factor_c, axis=0)

            g.append(grad_mu_k)

        g = np.array(g).flatten()

        if keep_record:
            norm_gk = np.linalg.norm(g)
            self.norm_g_k.append(norm_gk)

        return g.reshape(-1, )


class Rastrigin(TestFunction):

    def __init__(self, n: int):
        super().__init__(n)

    def function(self, x: np.ndarray, keep_record=True):
        f_k = 10 * self.n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
        if keep_record:
            self.f_k.append(float(f_k))

        return f_k

    def gradient(self, x: np.ndarray, keep_record=True):
        g_k = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
        if keep_record:
            norm_gk = np.linalg.norm(g_k)
            self.norm_g_k.append(norm_gk)

        return g_k

    def hessian(self, x: np.ndarray):
        diag = 2 + 40 * np.pi ** 2 * np.cos(2 * np.pi * x)
        h_k = np.diag(diag, k=0)

        return h_k


class Wood(TestFunction):

    def __init__(self, n: int):
        super().__init__(n)

    def function(self, x: np.ndarray, keep_record=True):

        x_1, x_2, x_3, x_4 = x[0], x[1], x[2], x[3]
        f_x = 100. * (x_1 ** 2. - x_2) ** 2. + \
              (x_1 - 1.) ** 2. + \
              (x_3 - 1.) ** 2. + 90. * (x_3 ** 2. - x_4) ** 2. + \
              10.1 * ((x_2 - 1.) ** 2. + (x_4 - 1.) ** 2.) + 19.8 * (x_2 - 1) * (x_4 - 1.)

        # Registro de evolución de valores
        if keep_record:
            self.f_k.append(float(f_x))

        return f_x

    def gradient(self, x: np.ndarray, keep_record=True):

        x_1, x_2, x_3, x_4 = x[0], x[1], x[2], x[3]

        g_x = np.array([400. * x_1 * (x_1 ** 2 - x_2) + 2. * (x_1 - 1.),
                        -200. * (x_1 ** 2 - x_2) + 20.2 * (x_2 - 1.) + 19.8 * (x_4 - 1),
                        2. * (x_3 - 1) + 360. * x_3 * (x_3 ** 2 - x_4),
                        -180. * (x_3 ** 2 - x_4) + 20.2 * (x_4 - 1.) + 19.8 * (x_2 - 1.)])

        if keep_record:
            norm_g_x = np.linalg.norm(g_x)
            self.norm_g_k.append(norm_g_x)

        return g_x


class Tangled(TestFunction):

    def __init__(self, n: int, lmbda: float, sigma: float, sigma_y: float):
        assert (lmbda > 0)
        assert (sigma > 0)
        super().__init__(n)
        self.eta = np.random.normal(0, sigma, size=(n,))
        self.lmbda = lmbda
        self.sigma = sigma
        self.t = np.array([(2. * i / (n - 1.)) - 1. for i in range(n)])
        self.y = self.t ** 2 + self.eta

    def function(self, x: np.ndarray, keep_record=True):

        f_k = np.sum((x - self.y) ** 2)  # Primera suma
        f_k += self.lmbda * np.sum((x[1:] - x[:-1]) ** 2)  # Segunda suma

        # Registro de evolución de valores
        if keep_record:
            self.f_k.append(float(f_k))

        return f_k

    def gradient(self, x: np.ndarray, keep_record=True):

        # Primera coordenada
        g_k = np.array(2. * (x[0] - self.y[0]) + 2. * self.lmbda * (x[0] - x[1]))

        # Entradas distintas de la primera y última coordenada
        x_i_m_1 = x[0:-2]
        x_i = x[1:-1]
        y_i = self.y[1:-1]
        x_i_p_1 = x[2:]
        middle = 2. * (x_i - y_i) + 2. * self.lmbda * (x_i - x_i_m_1) + 2. * self.lmbda * (x_i - x_i_p_1)
        g_k = np.concatenate((g_k, middle), axis=None)

        # Última coordenada
        last = np.array([2. * (x[-1] - self.y[-1]) + 2. * self.lmbda * (x[-1] - x[-2])])
        g_k = np.concatenate((g_k, last), axis=None)

        if keep_record:
            norm_gk = np.linalg.norm(g_k)
            self.norm_g_k.append(norm_gk)

        return g_k
