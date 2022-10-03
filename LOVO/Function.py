import numpy as np


class TestFunction(object):
    def __init__(self, n: int, p: int, dataset: np.ndarray):
        self.n = n # dimension of the space of observations
        self.p = p  # number of normal observations (r-p=number of outliers)
        self.r = len(dataset)  # total number of observations
        self.f_k = []
        self.norm_g_k = []
        self.dataset = dataset
        self.R_index = np.zeros((self.r, 2))  # stores Ri  (1/2 * (model - yi)^2)  and index of point in the dataset

    # returns vector of model function applied to each point in dataset
    def function(self, x: np.ndarray):
        raise NotImplementedError

    # gradient of objective function
    def gradient(self, x: np.ndarray, keep_record=True):
        # get indexes of first-p Ri (lower values)
        dataset_index = self.R_index[:self.p, 1].astype(int)
        y_model_diff = self.dataset[:, -1] - self.function(x)

        J = self.Jacobian(x)

        grad = np.matmul(J.T, y_model_diff[dataset_index])

        # record grad norm
        if keep_record:
            norm_gk = np.linalg.norm(grad)
            self.norm_g_k.append(norm_gk)

        return grad

    def Jacobian(self, x: np.ndarray):
        raise NotImplementedError

    # returns vector of abs difference between model function applied to each point and the real value y_i (last column in dataset)
    def abs_diff(self, x: np.ndarray):
        return np.abs(self.function(x) - self.dataset[:, -1])

    # returns sum of first p values of R_Index
    # basically this is the objective function
    def S(self, x: np.ndarray, keep_record=True):

        # sort Ri's
        self.getR(x)
        # indexes = self.R_index[:self.p, 1].astype(int)
        # R = 1 / 2 * (self.function(x)[indexes] - self.dataset[indexes, -1]) ** 2
        sum_ = np.sum(self.R_index[:self.p, 0])

        if keep_record:
            self.f_k.append(sum_)
        return sum_

    # calculates each Ri and sorts it with its index
    def getR(self, x: np.ndarray):
        R = 1 / 2 * (self.function(x) - self.dataset[:, -1]) ** 2

        # build vector of R value and its index
        for i in range(len(R)):
            self.R_index[i] = (R[i], i)

        # sort R_index by first attribute, ie, R value
        self.R_index = self.R_index[np.argsort(self.R_index[:, 0])]


class Linear_Model(TestFunction):
    def __init__(self, n, p, dataset: np.ndarray):
        super().__init__(n, p, dataset)

    # model function to adjust
    def function(self, x: np.ndarray, keep_record=True):
        return np.matmul(self.dataset[:, :-1], x[:-1]) + x[-1]

    def Jacobian(self, x: np.ndarray):
        J = np.zeros((self.p, self.n))

        # get indexes of first-p Ri (lower values)
        dataset_index = self.R_index[:self.p, 1].astype(int)

        # Set each Jacobian Entry
        J[:, :-1] = -self.dataset[dataset_index, :-1]
        J[:, -1] = -1

        return J


class Polinomial_Model(TestFunction):

    def __init__(self, n, p, dataset: np.ndarray):
        super().__init__(n, p, dataset)
        degree = self.n-1
        powe = np.arange(degree) + 1
        powe = -np.sort(-powe)  # powers are in descendent order
        t_rep = np.column_stack((self.dataset[:, 0].reshape(-1,),)*degree)
        t_rep = t_rep ** powe  # raise each column

        ones = np.ones(shape=(self.r, 1))
        self.t_c1 = np.hstack((t_rep, ones))  # concatanates a ones column

    # model function to adjust
    def function(self, x: np.ndarray, keep_record=True):

        """
        :param keep_record: guardar la evaluación en el objeto

        :return: su evaluación en el modelo polinomial

        """
        return np.matmul(self.t_c1, x).reshape(-1,)

    def Jacobian(self, x: np.ndarray):

        # get indexes of first-p Ri (lower values)
        dataset_index = self.R_index[:self.p, 1].astype(int)

        return -self.t_c1[dataset_index]

class Exponential_Model(TestFunction):

    def __init__(self, n, p, dataset: np.ndarray):
        super().__init__(n, p, dataset)

    # model function to adjust
    def function(self, x: np.ndarray, keep_record=True):
        """
        :param keep_record: guardar la evaluación en el objeto

        :return: su evaluación en el modelo polinomial

        """
        m = np.matmul(self.dataset[:, :-1], x[2:])

        return x[0] + x[1] * np.exp(-m)

    def Jacobian(self, x: np.ndarray):
        J = np.zeros((self.p, self.n))

        # get indexes of first-p Ri (lower values)
        dataset_index = self.R_index[:self.p, 1].astype(int)

        # common terms
        t_k = self.dataset[dataset_index, :-1]
        m = np.matmul(t_k, x[2:])
        exp_m = np.exp(-m)

        # Set each Jacobian Entry
        J[:, 0] = - 1.

        J[:, 1] = - exp_m

        obs = self.dataset[dataset_index, :-1].reshape(-1,)

        m = np.multiply(obs, exp_m)

        col = (x[1] * m).reshape(-1,)

        J[:, 2] = col

        return J

class Logistic_Model(TestFunction):

    def __init__(self, n, p, dataset: np.ndarray):
        super().__init__(n, p, dataset)

        # A cada observación se le concatena un 1 al final, para faciltar los
        # cálculos en la función sigmoide
        ones = -np.ones(shape=(self.r, 1))
        self.x_1 = np.hstack((self.dataset[:, :-1], ones))

    # model function to adjust
    def function(self, x: np.ndarray, keep_record=True):
        """
        :param keep_record: guardar la evaluación en el objeto

        :param x: x1 y x2 son las primeras dos entradas, la última es x4
        y el resto son x3

        :return: su evaluación en el modelo logístico

        """
        m = np.matmul(self.x_1, x[2:])

        result = x[0] + x[1] / (1. + np.exp(-m))

        return result.reshape(-1, )

    def Jacobian(self, x: np.ndarray):

        J = np.zeros((self.p, self.n))

        # get indexes of first-p Ri (lower values)
        dataset_index = self.R_index[:self.p, 1].astype(int)

        # common terms
        t_k_1 = self.x_1[dataset_index, :]
        m = np.matmul(t_k_1, x[2:])
        exp_m = np.exp(-m)

        # Set each Jacobian Entry
        J[:, 0] = -1.

        J[:, 1] = -1. / (1. + exp_m)

        factor = exp_m / (1. + exp_m) ** 2

        obs = self.dataset[dataset_index, :-1].reshape(-1,)

        J_i = np.multiply(obs, factor)

        col = - x[1] * J_i # .reshape(-1, 1)

        J[:, 2] = col

        J[:, 3] = x[1] * exp_m / (1. + exp_m) ** 2

        return J



