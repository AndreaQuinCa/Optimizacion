import numpy as np
from benchmark import TestFunction, ModelMeansGaussianMix, ModelWeightsGaussianMix



def bisection_with_wolfe(c_1: np.float64, c_2: np.float64, test_function: TestFunction):
    """
    Búsqueda en línea con bisección y condiciones fuertes de Wolfe.

    Se requiere que c_1 < c_2.
    :param c_1: Parámetro de primera condición de Wolfe. Valor típico c_1 = 10e-4.
    :param c_2: Parámetro de segunda condición fuerte de Wolfe. Valor típico c_2 = 0.9,
    :param test_function: Clase con el gradiente y la definición de la función que se quiere
    optimizar
    :return: Una función que calcula el tamaño de paso con las  las condiciones fuertes de Wolfe
     y los parámetros anteriores en memoria.
    """

    assert (c_1 < c_2)
    f = test_function.function
    g = test_function.gradient

    def get_alpha(x_k: np.ndarray, g_k_d_k: np.float64, d_k: np.ndarray, alpha_0: np.float64 = 1.):

        # c_1, c_2 and test_function are already given
        a = 0.
        b = np.inf
        f_k = f(x_k, False)

        assert (0. <= alpha_0)
        alpha_i = alpha_0

        while True:
            x_k_1 = x_k + alpha_i * d_k
            g_k_1_d_k = np.dot(g(x_k_1, False), d_k)

            # No se cumple primera condición
            if not f(x_k_1, False) <= f_k + c_1 * alpha_i * g_k_d_k:

                # Actualización extremos; se acerca al inicial
                b = alpha_i

                # Actualización alpha_i
                alpha_i = 0.5 * (a + b)

            # No se cumple segunda condición
            elif not abs(g_k_1_d_k) <= c_2 * abs(g_k_d_k):

                # Actualización extremos; se aleja del inicial
                a = alpha_i

                # Actualización alpha_i
                if b < np.inf:
                    alpha_i = 0.5 * (a + b)
                else:
                    alpha_i = 2. * a
            # Se cumplen las dos
            else:
                break
            if check_tolerance(a, b, 10e-4):
                break

        return alpha_i

    return get_alpha


def backtracking_with_wolfe(c_1: np.float64, rho: np.float64, test_function: TestFunction):
    """
    Búsqueda en línea con backtracking y condiciones débiles de Wolfe.

    :param c_1: Valor típico 10e-4.
    :param rho: Debe cumplir 0<rho<1.
    :param test_function:
    :return:
    """
    assert (0. < c_1 < 1.)
    assert (0. < rho < 1.)

    f = test_function.function
    # g = test_function.gradient

    def get_alpha(x_k: np.ndarray, g_k_d_k: np.float64, d_k: np.ndarray, alpha_0: np.float64 = 1.):
        """
        Búsqueda en línea con backtracking y condiciones débiles de Wolfe.
        c_1, c_2, rho and test_function are already given
        :param x_k:
        :param g_k_d_k:
        :param d_k:
        :param alpha_0: Debe de ser positivo.
        :return:
        """
        assert (0. <= alpha_0)
        alpha = alpha_0
        f_k = f(x_k, False)
        x_k_1 = x_k + alpha * d_k

        # g_k_1_d_k = np.dot(g(x_k_1), d_k)

        while not (f(x_k_1, False) <= f_k + (alpha * c_1) * g_k_d_k):  # and abs(g_k_1_d_k) <= c_2 * abs(g_k_d_k)
            alpha = rho * alpha
            x_k_1 = x_k + alpha * d_k
            # g_k_1_d_k = np.dot(g(x_k_1), d_k)
            if check_tolerance(alpha, 0, 10e-4):
                break
        return alpha

    return get_alpha


def check_tolerance(value, value_new, epislon=1e-8):
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


def optimize_weights_means_alternate(x_0_alphas, x_0_means, data, c_1, rho, tol, initial_alpha_steepest,
                                     max_iter_outer, max_iter_inner, n, sigma=0.1):
    norm_g_alphas = []
    norm_g_mus = []
    f_k = []

    stop = False
    n_zero_alpha = np.zeros(n)
    n_zero_means = np.zeros(n*3)

    i = 1
    while True:

        # Optimización de pesos alpha, promedios fijos mu:

        print('\n Optimización alphas')

        function_weights = ModelWeightsGaussianMix(n, data, sigma, x_0_means)

        alpha_method = backtracking_with_wolfe(c_1, rho, function_weights)

        x_0_alphas_new, f_alphas_new, g_alphas_new, function_weights_new, k_weights = \
            steepest_descent(x_0_alphas, initial_alpha_steepest, function_weights, tol, tol, tol, max_iter_inner,
                             update_alpha=alpha_method)


        # Optimización de promedios mu, pesos fijos alpha:

        print('\n Optimización mu')

        function_means = ModelMeansGaussianMix(n, data, sigma, x_0_alphas)

        alpha_method = backtracking_with_wolfe(c_1, rho, function_means)

        x_0_means_new, f_means_new, g_means_new, function_means_new, k_means = \
            steepest_descent(x_0_means, initial_alpha_steepest, function_means, tol, tol, tol, max_iter_inner,
                             update_alpha=alpha_method)

        # Reporte
        if i % 1 == 0:
            print(" ")
            print("Iteration n.", i, "results:")
            print("f(x) = ", f_alphas_new)
            print("|g_alpha(x)| = ", np.linalg.norm(g_alphas_new))
            print("|g_means(x)| = ", np.linalg.norm(g_means_new))

        # Criterios de paro
        if i > 1:
            if check_tolerance(g_alphas_new, n_zero_alpha, tol) and check_tolerance(g_means_new, n_zero_means, tol):
                print("\n Convergió por gradiente")
                stop = True
            # if check_tolerance(f_alphas_new, f_alphas, tol) and check_tolerance(f_means_new, f_means, tol):
            #     print("\n Convergió por evaluación en f")
            #     stop = True
            # if check_tolerance(x_0_alphas_new, x_0_alphas, tol) and check_tolerance(x_0_means_new, x_0_means, tol):
            #     print("\n Convergió por cercanía en el dominio")
            #     stop = True
            if i >= max_iter_outer:
                print("Paró por número de iteraciones")
                stop = True


        # Registro
        f_k.append(f_alphas_new)
        norm_g_alphas.append(g_alphas_new)
        norm_g_mus.append(g_means_new)

        if stop:
            break

        # Actualización de puntos
        x_0_alphas, f_alphas, g_alphas = x_0_alphas_new, f_alphas_new, g_alphas_new
        x_0_means, f_means, g_means = x_0_means_new, f_means_new, g_means_new
        i += 1

    return x_0_alphas_new, x_0_means_new, norm_g_alphas, norm_g_mus, f_k


def steepest_descent(x, alpha, test_function: TestFunction, x_tol, f_tol, g_tol, max_iter=1000, update_alpha=None):
    """
    Optimización por máxima dirección de descenso (menos gradiente) con búsqueda en línea.

    :param x: punto inicial.
    :param alpha: tamaño de paso inicial.
    :param test_function: objeto con función de prueba, gradiente de la misma, registros de evaluaciones en la función.
    :param x_tol: para criterio de paro en el dominio.
    :param f_tol: para criterio de paro en las evaluaciones en la función de prueba.
    :param g_tol: para criterio de paro en las evaluaciones en el gradiente.
    :param max_iter: número máximo de interaciones para criterio de paro.
    :param update_alpha: función que se urlizará para búsqueda en línea.
    :return: regresa el último valor del dominio utlilizado, su evaluación en el a función y dentro del objeto
    test_function, el registro de las evaluaciones en f de la sucesión de puntos usados en la búsqueda.
    """
    f = test_function.function(x)
    g = test_function.gradient(x)
    d = -g
    n_zero = np.zeros(test_function.n)

    for i in range(max_iter):
        # Búsqueda en línea
        if update_alpha is not None:
            alpha = update_alpha(x, np.dot(g, d), d, alpha)

        # Criterios de paro
        x_new = x + alpha * d
        f_new = test_function.function(x_new)
        g_new = test_function.gradient(x_new)

        stop = False
        if check_tolerance(g, n_zero, g_tol):
            # print("\n Convergió por gradiente")
            stop = True
        if check_tolerance(f, f_new, f_tol):
            # print("\nConvergió por evaluación en f")
            stop = True
        if check_tolerance(x, x_new, x_tol):
            # print("Convergió por cercanía en el dominio")
            stop = True

        # if i % 1 == 0:
        #     print(" ")
        #     print("Iteration n.", i, "results:")
        #     print("|g(x)| = ", np.linalg.norm(g))
        #     print("f(x) = ", test_function.f_k[i])

        if stop:
            break
        # Actualización de punto del dominio
        x, f, g, d = x_new, f_new, g_new, -g_new

    # if i >= max_iter-1:
    #     # print("Paró por número de iteraciones")

    return x_new, f_new, g_new, test_function, i


