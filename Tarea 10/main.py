
from benchmark import SoftNoise
from optimization import *
from report import plot_2
import time
import matplotlib.pyplot as plt
import cv2
import pandas as pd


if __name__ == '__main__':
    """

    En esta sección se deben cambiar los parámetros.

    """
    # Parámetros de imagen:
    path = r'lenanoise.png'  # Dirección de la imagen
    img = cv2.imread(path, 0)  # Lectura de imagen en escala de grises
    filename_new_img = 'new_img.png'

    # Prámetros función:
    lmbda = 1.  # Parámetro de regularización
    mu = 0.01
    test_function = SoftNoise(img, lmbda, mu)  # Construcción de función objetivo y gradiente

    # Parámetros punto inicial:
    var = 1.  # Radio ruido
    seed = 101
    np.random.seed(seed)
    eta = np.random.uniform(-var, var, size=(test_function.n,))
    x_target = test_function.g_vec
    x_0 = x_target + eta  # Punto inicial

    # Parámetros algoritmo de optimización
    betas = ['FR']  # ['FRPR', 'PR', 'FR', 'HS']
    c_1, c_2 = 1e-4, 0.9
    update_alpha = bisection_with_wolfe(c_1, c_2, test_function)
    # beta = betas[0]

    # Parámetros criterios de paro:
    tol = 1.e-4  # Tolerancia norma del gradiente
    max_iter = 20  # Número máximo de iteraciones

    # Parámetros estadísticas
    filename_stats = 'estadísticas.csv'
    nruns = 1
    data = []
    index = []

    # Parámetros gráficas
    start_plot = 3

    for idx, beta in enumerate(betas):
        method_name = "beta = " + beta
        update_beta = get_non_linear_beta(beta)

        # Estadísticas
        times = []
        iterations = []
        errors = []

        for run in range(nruns):
            # Optimización
            begin = time.time()
            x, f, g, test_function, k = conjugate_gradient_non_linear(x_0, test_function, tol, update_beta,
                                                                      max_iter, update_alpha)
            time.sleep(1)
            end = time.time()
            times += [end - begin]
            iterations += [k]
            dif = x_target - x
            err = np.linalg.norm(dif)
            errors += [err]

        # Gráficas
        plot_title = 'Gradiente Conjugado, beta = ' + method_name
        y1, y2 = test_function.f_k[start_plot:], test_function.norm_g_k[start_plot:]
        x1 = np.array(list(range(len(y1)))) + start_plot
        x2 = x1
        plot_2(x1, y1, x2, y2, 'f(x_k)', '|g(x_k)|', plot_title)

        # Imagen suavizada
        new_img = x.reshape(img.shape)
        plt.imshow("Image", new_img)
        cv2.waitKey(0)
        plt.show()
        cv2.imwrite(beta+'_'+filename_new_img, new_img)

        # Tabla con reporte de desempeño:

        method_dict = {"Método": beta,
                       "Tiempo promedio (s)": round(np.mean(times), 2),
                       "Iteraciones promedio": round(np.mean(iterations), 2),
                       "Error promedio": round(np.mean(errors), 2)
                       }

        data.append(method_dict)
        index.append(idx)


    stats = pd.DataFrame(data, index=index)
    stats.to_csv(filename_stats, encoding='utf-8', index=False)

