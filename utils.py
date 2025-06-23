# %%
import numpy as np

GEV_FM = 0.1973269631  # 1 = 0.197 GeV . fm
CF = 4 / 3  # color factor
NF = 3  # number of flavors
CA = 3
TF = 1 / 2

lms = 0.2445  # Lambda_MS
b0 = 11 - 2 / 3 * NF  # beta0 for QCD



def matching_kernel_NLO(lc_x_ls, quasi_y_ls, pz_gev, mu):
    x_ls = lc_x_ls
    y_ls = quasi_y_ls
    
    alphas = 2 * np.pi / (b0 * np.log(mu / lms))
    
    dx = abs(x_ls[1] - x_ls[0])
    
    # for x > y
    def H1(x, y): 
        xi = x / y
        
        val = ( 1 + xi**2 ) / (1 - xi) * np.log( xi / (xi - 1) ) + 1 + 3 / (2 * xi)
        
        return val

    # for 0 < x < y
    def H2(x, y):
        xi = x / y
        temp = - np.log( mu**2 / (4 * x**2 * pz_gev**2) ) + np.log( (1 - xi) / xi )
        
        val = ( 1 + xi**2 ) / (1 - xi) * temp - xi * (1 + xi) / (1 - xi)
        
        return val

    x_grid, y_grid = np.meshgrid(x_ls, y_ls, indexing='ij')
    diff = x_grid - y_grid
    xi_grid = x_grid / y_grid
    
    matrix_NLO = np.zeros((len(x_ls), len(y_ls)))
    # x > y: H1
    mask1 = diff > (dx / 10)
    matrix_NLO[mask1] = H1(x_grid[mask1], y_grid[mask1])
    # x < y: H2
    mask2 = diff < -(dx / 10)
    matrix_NLO[mask2] = H2(x_grid[mask2], y_grid[mask2])
    
    # diagonal input
    for idx in range(len(x_ls)):
        if matrix_NLO[idx, idx] != 0:
            print("matrix diagnoal error")
        matrix_NLO[idx, idx] = -np.sum(matrix_NLO[:, idx])
    
    # extra term for x > y
    matrix_NLO[mask1] += 3 / (2 * xi_grid[mask1])
    
    matrix_NLO = matrix_NLO * alphas * CF / (2 * np.pi)
    return matrix_NLO

def matching_fixed_order(quasi_da, lc_x_ls, quasi_y_ls, pz_gev, mu):
    dy = abs(quasi_y_ls[1] - quasi_y_ls[0])
    
    matching_matrix_NLO = matching_kernel_NLO(lc_x_ls, quasi_y_ls, pz_gev, mu)
    
    y_grid = np.broadcast_to(quasi_y_ls, matching_matrix_NLO.shape)
    matching_matrix_complete = matching_matrix_NLO * dy / np.abs(y_grid)
    lc_da_fixed_order = quasi_da - np.dot(matching_matrix_complete, quasi_da)
    return lc_da_fixed_order


def DGLAP_kernel(x_ls, v_ls, mu):
    dx = abs(x_ls[1] - x_ls[0])
    
    alphas = 2 * np.pi / (b0 * np.log(mu / lms))
    x_grid, v_grid = np.meshgrid(x_ls, v_ls, indexing='ij')
    w_grid = x_grid / v_grid
    
    matrix_DGLAP = np.zeros((len(x_ls), len(v_ls)))
    mask = (v_grid - x_grid) > (dx / 10)
    matrix_DGLAP[mask] = 2 / (1 - w_grid[mask]) - 1 - w_grid[mask]
    
    # diagonal input
    for idx in range(len(x_ls)):
        if matrix_DGLAP[idx, idx] != 0:
            print("matrix diagnoal error")
        matrix_DGLAP[idx, idx] = -np.sum(matrix_DGLAP[:, idx])
    matrix_DGLAP = matrix_DGLAP * alphas * CF / (2 * np.pi)
        
    return matrix_DGLAP


def DGLAP_evolution(lc_x_ls, lc_da_mu_i, mu_i, mu_f):
    N_steps = 100
    dmu = (mu_f - mu_i) / N_steps
    
    x_ls = lc_x_ls
    v_ls = lc_x_ls
    dv = abs(v_ls[1] - v_ls[0])
    
    
    lc_da_loop = lc_da_mu_i
    for step in range(N_steps + 1):
        mu = mu_i + step * dmu
        
        matrix_DGLAP = DGLAP_kernel(x_ls, v_ls, mu)
        
        matrix_complete = np.zeros_like(matrix_DGLAP)
        x_grid, v_grid = np.meshgrid(x_ls, v_ls, indexing='ij')
        mask = (v_grid - x_grid) > (dv / 10)
        matrix_complete[mask] = matrix_DGLAP[mask] * dv / np.abs(v_grid[mask])
        
        g_mu = np.dot(matrix_complete, lc_da_loop)
        
        lc_da_loop += g_mu * ( np.log( mu + dmu ) - np.log( mu ) )
    
    assert abs(mu - mu_f) < 0.001, f"Evolution loop ended at mu={mu}, but target was mu_f={mu_f}"
    lc_da_mu_f = lc_da_loop
        
    return lc_da_mu_f

# %%