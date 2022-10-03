import cv2
import numpy as np
from data import Data, Segmentation
from optimization import optimize_weights_means_alternate
from report import plot_2

if __name__ == '__main__':
    """
    En esta sección se deben cambiar los parámetros.
     
    """

    # Parámetros de segmentación

    img_path = 'sheep/sheep.bmp'
    path_class_0 = 'sheep/H_0.txt'
    path_class_1 = 'sheep/H_1.txt'

    seed = 10
    np.random.seed(seed)
    n = 8  # Número de gaussianas en la base
    sigma = 1.0
    alphas = np.random.sample(n)  # Distribución uniforme entre el [0,1)
    mus = np.random.sample(n*3)

    title_original = 'Original'
    title_segmentation_F = 'Segmentation with F'
    title_segmentation_hist = 'Segmentation with H'

    # Parámetros criterios de paro:
    tol = 1.e-4
    max_iter_outer = 500
    max_iter_inner = 1000

    # Parámetros backtracking:
    alpha_steepest = 0.5  # Tamaño de paso inicial
    c_1 = 1.e-4  # Debe de estar entre 0 y 1
    rho = 0.9  # Debe de estar entre 0 y 1

    # Parámetros reportes:
    filename_best_alphas_0 = 'filename_best_alphas_0' + '.txt'
    filename_best_means_0 = 'filename_best_means_0' + '.txt'
    filename_best_alphas_1 = 'filename_best_alphas_1' + '.txt'
    filename_best_means_1 = 'filename_best_means_1' + '.txt'

    star_plot = 0  # primera iteración que se muestra en las gráficas de normas de gradientes
    y1_label, y2_label = "|g_alpha|", "|g_mu|"
    title_plot_norms = 'Norma de gradientes'

    # Principal

    # Lectura de datos:
    data_0 = Data(path_class_0)
    data_1 = Data(path_class_1)

    # Optimización clase 0
    print("\n Optimización clase 0")
    alphas_0, means_0, norm_g_alphas_0, norm_g_mus_0, f_k = \
        optimize_weights_means_alternate(alphas, mus, data_0, c_1, rho, tol, alpha_steepest,
                                         max_iter_outer, max_iter_inner, n, sigma)
    # Guarda resultados
    np.savetxt(filename_best_alphas_0, alphas_0)
    np.savetxt(filename_best_means_0, means_0)

    print("\n Optimización clase 1")
    # Optimización clase 1
    alphas_1, means_1, norm_g_alphas_1, norm_g_mus_1, f_k = \
        optimize_weights_means_alternate(alphas, mus, data_1, c_1, rho, tol, alpha_steepest,
                                         max_iter_outer, max_iter_inner, n, sigma)

    # Guarda resultados
    np.savetxt(filename_best_alphas_1, alphas_1)
    np.savetxt(filename_best_means_1, means_1)

    # Segmentación
    segmentation = Segmentation(alphas_0, means_0, alphas_1, means_1, data_0, data_1, sigma)
    img = cv2.imread(img_path)
    segmentation.segmentate(img, title_original, title_segmentation_F)
    segmentation.segmentate(img, title_original, title_segmentation_hist, False)






