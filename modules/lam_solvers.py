from modules import *

def build_gamma_matrix(path_list, network):
    """Construit la matrice gamma (m x p) reliant OD aux chemins."""
    m = network.m   # total ODs
    p = sum(len(k) for k in path_list) # total paths

    gamma = np.zeros((m, p))
    col_start = np.cumsum([0] + [len(k) for k in path_list[:-1]])
    for i, start in enumerate(col_start):
        gamma[i, start:start+len(path_list[i])] = 1
    return gamma


def build_delta_matrix(path_list, network, G):
    """Construit la matrice delta (n x p) reliant liens aux chemins."""
    m = network.m   # total ODs
    n = network.n   # total links
    p = sum(len(k) for k in path_list) # total paths

    delta = np.zeros((n, p))
    nn = 0
    for i in range(m):
        for j, path in enumerate(path_list[i]):
            for k in range(len(path) - 1):
                u, v = path[k], path[k+1]
                arc_index = G[u][v]['index']
                delta[arc_index, nn + j] = 1
        nn += len(path_list[i])
    return delta


def build_T_matrix(path_list, network, G):
    """Construit la matrice T (p-m x n) pour les différences entre chemins."""
    m = network.m   # total ODs
    n = network.n   # total links
    p = sum(len(k) for k in path_list) # total paths
    
    T = np.zeros((p - m, n))
    nn = 0
    for i in range(m):
        for j in range(len(path_list[i]) - 1):
            path1 = path_list[i][j]
            path2 = path_list[i][j + 1]

            for k in range(len(path1) - 1):
                u, v = path1[k], path1[k + 1]
                link_idx = G[u][v]['index']
                T[nn + j, link_idx] += 1

            for k in range(len(path2) - 1):
                u, v = path2[k], path2[k + 1]
                link_idx = G[u][v]['index']
                T[nn + j, link_idx] -= 1

        nn += len(path_list[i]) - 1
    return T


def extract_multiple_indices(path_list, delta):
    """    Retourne les indices associés aux OD multipath.    """
    m = len(path_list)
    nn = 0
    multiple_od_indices = []
    multiple_path_indices = []
    
    for i, paths in enumerate(path_list):
        if len(paths) > 1:
            multiple_od_indices.append(i)
            multiple_path_indices.extend(range(nn, nn + len(paths)))
        nn += len(paths)
    
    multiple_od_indices = np.array(multiple_od_indices)
    multiple_path_indices = np.array(multiple_path_indices)
    multiple_link_indices = np.where(np.sum(delta[:, multiple_path_indices], axis=1) > 0)[0]
    
    return {
        "od_m": multiple_od_indices,
        "paths_m": multiple_path_indices,
        "links_m": multiple_link_indices,
        "m": len(multiple_od_indices),
        "p": len(multiple_path_indices),
        "n": len(multiple_link_indices)
    }


def build_gamma_m_matrix(gamma, indices):
    """ Construit la matrice gamma réduite aux ODs et chemins multiples. """
    gamma_m = np.zeros((indices['m'], indices['p']))
    for i_local, i_global in enumerate(indices['od_m']):
        for j_local, j_global in enumerate(indices['paths_m']):
            gamma_m[i_local, j_local] = gamma[i_global, j_global]
    return gamma_m


def build_delta_m_matrix(delta, indices):
    delta_m = np.zeros((indices['n'], indices['p']))
    for i_local, i_global in enumerate(indices['links_m']):
        for j_local, j_global in enumerate(indices['paths_m']):
            delta_m[i_local, j_local] = delta[i_global, j_global]
    return delta_m
    

def build_delta_m_u_matrix(delta, gamma_m, delta_m, indices):
    p = delta.shape[1]
    m_multiple = gamma_m.shape[0]
    
    delta_m2 = []
    delta_m_copy = delta_m.copy()
    multiple_link_indices_copy = indices['links_m'].copy()
    
    i = 0
    while i < len(multiple_link_indices_copy):
        A = np.vstack([delta_m_copy[i, :], gamma_m])
        
        if np.linalg.matrix_rank(A) < m_multiple + 1:
            delta_m2.append(delta_m_copy[i, :].copy())
            delta_m_copy = np.delete(delta_m_copy, i, axis=0)
            multiple_link_indices_copy = np.delete(multiple_link_indices_copy, i)
        else:
            i += 1
    
    # Indices des colonnes uniques (complémentaires aux chemins multiples)
    cols_u = sorted(set(range(p)) - set(indices['paths_m']))
    delta_m_u = delta[np.ix_(multiple_link_indices_copy, cols_u)]
    
    # Mise à jour du dictionnaire d'indices avec les liens finaux
    indices_updated = indices.copy()
    indices_updated['links_m'] = multiple_link_indices_copy
    indices_updated['n'] = len(multiple_link_indices_copy)
    
    return delta_m_u, delta_m2, indices_updated


def build_delta_u_m_matrix(path_list, network, G):

    # Construction des matrices de base
    gamma = build_gamma_matrix(path_list, network)
    delta = build_delta_matrix(path_list, network, G)
    
    # Identification des indices multiples
    indices = extract_multiple_indices(path_list, delta)
    
    # Construction des matrices réduites
    gamma_m = build_gamma_m_matrix(gamma, indices)
    delta_m = build_delta_m_matrix(delta, indices)
    
    # Construction de delta_m_u et récupération des indices finaux
    delta_m_u, delta_m2, indices_final = build_delta_m_u_matrix(
        delta, gamma_m, delta_m, indices
    )
    
    # Construction de la matrice finale
    n = network.n
    n_remaining = n - indices_final['n']
    p_multiple = indices_final['p']
    
    delta_u_m = np.zeros((n_remaining, p_multiple))
    
    if len(delta_m2) > 0:
        delta_m2 = np.array(delta_m2)
        delta_u_m_top = (pinv(gamma_m.T) @ delta_m2.T).T
        delta_u_m = np.vstack([delta_u_m_top, delta_u_m])
    
    return delta_u_m, indices_final

# ---------- UTILITAIRES ----------
def simplify_binary(mat, tol=1e-8):
    binary_mat = np.zeros_like(mat)
    binary_mat[np.abs(mat) > tol] = 1
    return binary_mat
# ---------- ----- ----------

from numpy.linalg import matrix_rank
from scipy.linalg import null_space, pinv
from sympy import Matrix

def compute_Rr_and_r0(delta_m, gamma_m, indices_final, network):
    """ Construit les matrices Rr et r0 pour la réduction dimensionnelle. """
    # Réduction en forme échelonnée
    delta_m_sym = Matrix(delta_m)
    delta_m_rref, _ = delta_m_sym.rref()
    delta_m_tilde = np.array(delta_m_rref, dtype=np.float64)

    rank_delta_m = matrix_rank(delta_m)
    delta_m_tilde = delta_m_tilde[:rank_delta_m, :]

    # Nullspace binaire
    M_for_null = (delta_m_tilde - pinv(gamma_m @ pinv(delta_m_tilde)) @ gamma_m).T
    ns = null_space(M_for_null)

    if ns.size == 0:
        A = np.zeros((0, M_for_null.shape[0]))
    else:
        A = simplify_binary(ns).T

    # Matrice Rr
    Rr = A @ pinv(gamma_m @ pinv(delta_m_tilde))

    # Extraction de q_od restreint (OD multiples uniquement)
    q_od_r = np.array(network.q_od)[indices_final['od_m']].reshape(-1, 1)
    r0 = Rr @ q_od_r

    return {
        "delta_r_prime": delta_m_tilde,
        "rank_delta_r": rank_delta_m,
        "A_r": A,
        "Rr": Rr,
        "r0": r0
    }
