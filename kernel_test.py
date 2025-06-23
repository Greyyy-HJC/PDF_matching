# %%
import numpy as np

GEV_FM = 0.1973269631  # 1 = 0.197 GeV . fm
CF = 4 / 3  # color factor
NF = 3  # number of flavors
CA = 3
TF = 1 / 2

lms = 0.24451721864451428  # Lambda_MS
mu = 2  # GeV, for factorization
b0 = 11 - 2 / 3 * NF  # beta0 for QCD
alphas = 2 * np.pi / (b0 * np.log(mu / lms))


def matching_kernel_matrix(quasi_x_ls, lc_y_ls, pz_gev):
    x_ls = quasi_x_ls
    y_ls = lc_y_ls

    delta_y = abs(y_ls[1] - y_ls[0])

    matrix_LO = np.zeros([len(x_ls), len(y_ls)])
    for idx in range(len(x_ls)):
        matrix_LO[idx][idx] = 1

    def H1(x, y):
        return (1 + x - y) / (y - x) * (1 - x) / (1 - y) * np.log((y - x) / (1 - x)) + (
            1 + y - x
        ) / (y - x) * x / y * np.log((y - x) / (-x))

    def H2(x, y):
        return (1 + y - x) / (y - x) * x / y * np.log(
            4 * x * (y - x) * pz_gev**2 / mu**2
        ) + (1 + x - y) / (y - x) * (
            (1 - x) / (1 - y) * np.log((y - x) / (1 - x)) - x / y
        )

    matrix_NLO = np.zeros([len(x_ls), len(y_ls)])
    for idx1 in range(len(x_ls)):
        for idx2 in range(len(y_ls)):
            x = x_ls[idx1]
            y = y_ls[idx2]
            if abs(x - y) > 0.0001:
                if x < 0 and y > 0 and y < 1:
                    matrix_NLO[idx1][idx2] = H1(x, y)
                elif x > 0 and y > x and y < 1:
                    matrix_NLO[idx1][idx2] = H2(x, y)
                elif y > 0 and y < x and x < 1:
                    matrix_NLO[idx1][idx2] = H2(1 - x, 1 - y)
                elif y > 0 and y < 1 and x > 1:
                    matrix_NLO[idx1][idx2] = H1(1 - x, 1 - y)

    matrix_NLO = matrix_NLO * alphas * CF / (2 * np.pi)

    for idx in range(len(x_ls)):  # diagnoal input
        if matrix_NLO[idx][idx] != 0:
            print("matrix diagnoal error")
        matrix_NLO[idx][idx] = -np.sum([matrix_NLO[i][idx] for i in range(len(x_ls))])

    matrix = matrix_LO + matrix_NLO * delta_y

    return matrix_NLO
