
import numpy as np
import matplotlib.pyplot as plt
import cv2


class Data(object):
    """
    Contiene los datos además de funciones para leerlos y para mostrarlos.

    """

    def __init__(self, filename_hist: str):

        self.filename = filename_hist

        # Load the dataset class 0
        with open(self.filename) as f:
            bins = next(f)
            self.bins = [int(x) for x in bins.split()]
            hist_flat = np.loadtxt(f)  # no skiprows here
            self.hist = np.reshape(hist_flat, self.bins)

        c = []
        for i in range(self.bins[0]):
            for j in range(self.bins[1]):
                for k in range(self.bins[2]):
                    c.append((i, j, k))
        self.c = c


def plot_two_img(img1, img2, title_1='Original', title_2='Segmentation'):
    fig = plt.figure(figsize=(7, 7))
    rows = 1
    columns = 2
    ax1 = fig.add_subplot(rows, columns, 1)
    ax1.set_title(title_1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2 = fig.add_subplot(rows, columns, 2)
    ax2.set_title(title_2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.show()
    title = title_1 + " " + title_2
    title = "_".join(title.split())
    plt.savefig('comparison' + '.png')


class Segmentation(object):

    def __init__(self, alpha_1, mu_1, alpha_2, mu_2, data_0: Data, data_1: Data, sigma):

        self.alpha_1 = alpha_1
        self.mu_1 = mu_1

        self.alpha_2 = alpha_2
        self.mu_2 = mu_2

        self.bins = data_0.bins
        self.num_bins = self.bins[0]

        self.hist_1 = data_0.hist
        self.hist_2 = data_1.hist

        self.red = 1
        self.blue = 2

        self.n = alpha_1.shape[0]
        self.dev_std = sigma**2
        self.epsilon = 0.01

    def f(self, c, alpha, mu):

        c_centered_norm = np.linalg.norm(c - mu.reshape(self.n, -1), axis=1)

        c_centered_norm_std = c_centered_norm ** 2 / (2. * self.dev_std)

        exp_neg_std_norms = np.exp(-c_centered_norm_std)

        f = np.sum(alpha * exp_neg_std_norms)

        return f

    def F(self, c, alpha, mu):

        num = self.f(c, alpha, mu) + self.epsilon
        denum = self.f(c, self.alpha_1, self.mu_1) + self.f(c, self.alpha_2, self.mu_2) + 2. * self.epsilon

        return num/denum

    def H(self, c, hist):
        c = tuple(c)
        num = hist[c] + self.epsilon
        denum = self.hist_1[c] + self.hist_2[c] + 2. * self.epsilon

        return num/denum

    def classify(self, c, withF=True):
        if withF:
            if self.F(c, self.alpha_1, self.mu_1) < self.F(c, self.alpha_2, self.mu_2):
                return self.blue
            else:
                return self.red
        else:
            if self.H(c, self.hist_1) < self.H(c, self.hist_2):
                return self.blue
            else:
                return self.red

    def classify_colors(self, withF):

        labels = np.zeros(self.bins, dtype=np.int)

        for i in range(self.bins[0]):
            for j in range(self.bins[1]):
                for k in range(self.bins[2]):

                    c = np.array([i, j, k])

                    labels[i, j, k] = self.classify(c, withF)

        return labels

    def rgb_to_bin(self, rgb):

        x = int(float(rgb[0]) / 256.0 * self.num_bins)
        y = int(float(rgb[1]) / 256.0 * self.num_bins)
        z = int(float(rgb[2]) / 256.0 * self.num_bins)

        return np.array([x, y, z])

    def segmentate(self, img, title1, title2, withF=True):

        labels = self.classify_colors(withF)

        seg_img = img.copy()

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):

                c = self.rgb_to_bin(img[i][j])

                label = labels[c[0], c[1], c[2]]

                seg_img[i][j][0] = 255 if label == self.red else 0
                seg_img[i][j][1] = 0
                seg_img[i][j][2] = 255 if label == self.blue else 0

        plot_two_img(img, seg_img, title1, title2)

        return seg_img


if __name__ == '__main__':

    # Lectura de datos:
    #
    # # data = Data('3bins/H_0.txt', '3bins/H_1.txt')
    #
    # # Cargar imagenes y convertirlas a escala de grises
    # imagen_1 = cv2.imread('3bins/Strokes.png')
    # img_1 = imagen_1.copy()
    # row = img_1.shape[0]
    # col = img_1.shape[1]
    #
    # color = [(255, 255, 255), (0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255)]
    # status_clase = 3
    #
    # x, y = 0, 0
    # cv2.circle(img_1, (x, y), 10, color[status_clase], -1)
    #

    x = np.arange(27)
    x = x.reshape((3, 3, 3))

