import numpy as np
import csv
import matplotlib.pyplot as plt


def generateLinearData(a, b, n=100, out_ratio=0.1):
    n_out = int(n * out_ratio)

    x = np.arange(n)
    x_out = np.random.uniform(0, n, n_out)

    y_out = a * x_out + b + np.random.uniform(-100, 100, n_out)
    y = a * x + b + np.random.uniform(0, 5, n)

    x_data = np.append(x, x_out)
    y_data = np.append(y, y_out)

    return np.column_stack((x_data, y_data))


def saveData(filename, data:np.ndarray):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            strrow = [x for x in row]
            writer.writerow(strrow)


def readData(filename):
    data = []
    with open(filename, 'r') as file:
        rows = file.readlines()
        for row in rows:
            data.append([float(x) for x in str.split(row, ',')])

    return np.array(data)


""" The following function generates data from a model and perturbs the output with a gaussian distribution
    - f_model is the function model to generate the data. It must be a callable function. The function must ensure that it returns 
      a vector, not a matrix of (n, 1) because in that case the outliers wont be generated. The function must receive first the 
      features matrix (x) and then the other parameters (*args).

    - *args are the additional arguments that the function receives. It can receive none

    - x is the matrix of features to produce the output based in the f_model. If x is none, then x will be a matrix generated
      from a uniform distribution of size n * m and with U(a, b) ~ U(min_x, max_x). If m = 1 then x will be transformed into a
      vector rather than a matrix of shape (n, 1)

    - out_ratio is the percentage of outliers in the dataset

    - mean_noise is the mean of the gaussian for perturbing the output of f_model
    - std_noise is the std of the gaussian for perturbing the output of f_model

    There are  n_out = out_ratio * n outliers generated. For that we select n_out index from y = f_model(x, args) and 

    we add values from a Uniform distribution ~ U(min_out, max_out). min_out and max_out should be big values in relation with
    the max and min output of the f_model so these points really become outliers. 
"""


def generateRandomData(f_model, *args, x=None, n=100, m=1, min_x=0, max_x=100, out_ratio=0.1, mean_noise=0, std_noise=1,
                       min_out=-1000, max_out=1000):
    assert (out_ratio < 1)
    n_out = int(out_ratio * n)

    # generate x matrix (features data)
    if x is None:
        x = np.random.uniform(min_x, max_x, (n, m))
        #x = np.linspace(min_x, max_x, n)

        # if data has one column, reshape it as a vector
        if m == 1:
            x = x.reshape(-1,)

    # chose index of elements that will be trusted (not outliers)
    trust_index = np.random.choice(n, n-n_out, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[trust_index] = False

    # generate output base on model with
    y = f_model(x, *args)

    # add noise from normal distribution to regular dots
    y[trust_index] = y[trust_index] + np.random.normal(mean_noise, std_noise, n - n_out)

    # make elements outliers
    s = np.random.choice([-1., 1.])
    out_noise = 7. * s * np.abs(np.random.normal(mean_noise, std_noise, n_out)) * np.random.uniform(min_out, max_out, n_out)
    y[mask] = y[mask] + out_noise

    # return features and output data in single matrix
    return np.column_stack((x, y)), trust_index


""" The following function generates data from a model and perturbs the output with a gaussian distribution
    - f_model is the function model to generate the data. It must be a callable function. The function must ensure that it returns 
      a vector, not a matrix of (n, 1) because in that case the outliers wont be generated. The function must receive first the 
      features matrix (x) and then the other parameters (*args).
    - *args are the additional arguments that the function receives. It can receive none
    - x is the matrix of features to produce the output based in the f_model. If x is none, then x will be a matrix generated
      from a uniform distribution of size n * m and with U(a, b) ~ U(min_x, max_x). If m = 1 then x will be transformed into a
      vector rather than a matrix of shape (n, 1)
    - out_ratio is the percentage of outliers in the dataset
    - mean_noise is the mean of the gaussian for perturbing the output of f_model
    - std_noise is the std of the gaussian for perturbing the output of f_model
    There are  n_out = out_ratio * n outliers generated. For that we select n_out index from y = f_model(x, args) and 
    we add values from a Uniform distribution ~ U(min_out, max_out). min_out and max_out should be big values in relation with
    the max and min output of the f_model so these points really become outliers. 
"""


def generateRandomData_2(f_model, *args, x=None, n=100, m=1, min_x=0, max_x=100, out_ratio=0.1, mean_noise=0, std_noise=1,
                         min_out=-1000, max_out=1000):
    assert (out_ratio < 1)
    n_out = int(out_ratio * n)

    # generate x matrix (features data)
    if x is None:
        x = np.random.uniform(min_x, max_x, (n, m))

        # if data has one column, reshape it as a vector
        if m == 1:
            x = x.reshape(n)

    # generate output base on model with noise from normal distribution
    y = f_model(x, *args) + np.random.normal(mean_noise, std_noise, n)

    # chose index of elements that will be outliers
    out_index = np.random.choice(n, n_out, replace=False)

    # made elements outliers
    y[out_index] = y[out_index] + np.random.uniform(min_out, max_out, n_out)

    # return features and output data in single matrix
    return np.column_stack((x, y))

# ------------------------------------------------ MODELS TO GENERATE DATA ------------------------------------------------

def linear_model(x, coeff):

    return np.matmul(x.reshape(-1, 1), coeff[:-1]) + coeff[-1]


def linear_model2D(x, a1, a2, b):
    return a1 * x[:, 0] + a2 * x[:, 1] + b


def cuadratic_model2D(x, a, b, c):
    return a * x ** 2 + b * x + c

def polinomial_model(x, coeffs):

    degree = len(coeffs)-1
    powe = np.arange(degree) + 1
    powe = -np.sort(-powe)  # powers are in descendent order
    x_rep = np.column_stack((x.reshape(-1,),)*degree)
    x_pow = x_rep ** powe  # raise each column

    ones = np.ones(shape=(x.shape[0], 1))
    x_c1 = np.hstack((x_pow, ones))  # concatanates a ones column

    mult = np.matmul(x_c1, coeffs)

    return mult.reshape(-1, )


def exponential_model(x, coeffs):

    m = np.matmul(x.reshape(-1, 1), coeffs[2:])

    result = coeffs[0] + coeffs[1] * np.exp(-m)

    return result.reshape(-1, )

def logistic_model(x, coeffs):

    ones = -np.ones(shape=(x.shape[0], 1))

    x_1 = np.hstack((x.reshape(-1, 1), ones))

    m = np.matmul(x_1, coeffs[2:])

    result = coeffs[0] + coeffs[1] / (1. + np.exp(-m))

    return result.reshape(-1, )

#
