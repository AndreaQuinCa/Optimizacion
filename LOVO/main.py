import matplotlib.pyplot as plt
import numpy as np

import Function
import Optimizacion
import Data
import time


def generateDatasets(model, coeffs, csv_filename, n_points, x_dim, out_ratio):
    dataset, out_index = Data.generateRandomData(model, coeffs, x=None, n=n_points, m=x_dim, min_x=1,
                                                 max_x=30, out_ratio=out_ratio, std_noise=200, min_out=1, max_out=2)

    # np.savetxt(csv_filename, dataset)
    # Data.saveData(csv_filename, dataset)
    # plot dataset
    plt.scatter(dataset[:, 0], dataset[:, -1])
    plt.title("Original data")
    plt.show()

    return dataset, out_index


print("Init")

# Params:

r = 100  # number of observations
m = 1  # dim of observations
model = Function.Polinomial_Model
out_ratio = .10  # outliers proportion of the observations
max_iter = 5000
np.random.seed(1)

if model == Function.Linear_Model:
    coeffs = np.array([-200., -1000])
    model_name = 'linear'
    gen_model = Data.linear_model
elif model == Function.Polinomial_Model:
    coeffs = np.array([0.5, -2., 10., 20])  # np.array([0.5, -2., 10., 20])
    model_name = 'cubic'
    gen_model = Data.polinomial_model
elif model == Function.Exponential_Model:
    coeffs = np.array([5000., 4000., 0.2])
    model_name = 'exp'
    gen_model = Data.exponential_model
else:
    coeffs = np.array([6000, -5000., -0.2, -3.7])
    model_name = 'logistic'
    gen_model = Data.logistic_model

# read dataset
csv_filename = model_name + '_' + str(r) + '_data_' + '.csv'
n = len(coeffs)
dataset, real_trust = generateDatasets(gen_model, coeffs, csv_filename, n_points=r, x_dim=m, out_ratio=out_ratio)

# create model function
p = len(real_trust)
model = model(n, p, dataset)

# initial point
# x0 = np.array([1000., 5000., 0.2, 15]) + np.random.normal(4) * 0.1 # np.zeros_like(coeffs)  # coeffs + np.random.normal(0, 1, coeffs.shape)
x0 = np.zeros_like(coeffs) + np.random.normal(4) * 0.1 # coeffs + np.random.normal(0, 1, coeffs.shape)

# find optimum and number of trusted points
p_min = int(.5 * r)
p_max = r
xopt, classified_as_trust = Optimizacion.RAFF(x0, model, p_min, p_max, max_iter=max_iter, real_trust=real_trust, lambda_0=1e-4, epsilon=100)

# new dataset without outliers
new_dataset = model.dataset[classified_as_trust, :]

# plot result
assert (len(dataset) == r)
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

# TODO: legends


x_model, y_model = new_dataset[:, 0], gen_model(new_dataset[:, 0].reshape((len(classified_as_trust), 1)), xopt)

# sort model_data by x coordinate
model_data = np.column_stack((x_model, y_model))
model_data = model_data[np.argsort(model_data[:, 0])]

plt.plot(model_data[:, 0], model_data[:, 1], color='blue')
plt.title("Proposed model based in trusted points")
plt.show()
print("Relative error", np.linalg.norm(coeffs-xopt)/np.linalg.norm(coeffs))

# dataset = Data.readData('linear_data.csv')
# plt.scatter(dataset[:, 0], dataset[:, -1])
# plt.title("Original data")
# plt.show()
# x0 = np.array([1, 1])
