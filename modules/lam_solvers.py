from modules import *
import modules.cost_functions as cf

def build_gamma_matrix(path_list, network):
    """Construit la matrice gamma (m x p) reliant OD aux chemins."""
    n = len(network.sn) # total links
    m = len(network.on) # total ODs
    p = sum(len(k) for k in path_list) # total paths

    gamma = np.zeros((m, p))
    col_start = np.cumsum([0] + [len(k) for k in path_list[:-1]])
    for i, start in enumerate(col_start):
        gamma[i, start:start+len(path_list[i])] = 1
    return gamma


def build_delta_matrix(path_list, network, G):
    """Construit la matrice delta (n x p) reliant liens aux chemins."""
    n = len(network.sn) # total links
    m = len(network.on) # total ODs
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
    n = len(network.sn) # total links
    m = len(network.on) # total ODs
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
        "m_m": len(multiple_od_indices),
        "p_m": len(multiple_path_indices),
        "n_m": len(multiple_link_indices)
    }


def build_gamma_m_matrix(gamma, indices):
    """ Construit la matrice gamma réduite aux ODs et chemins multiples. """
    gamma_m = np.zeros((indices['m_m'], indices['p_m']))
    for i_local, i_global in enumerate(indices['od_m']):
        for j_local, j_global in enumerate(indices['paths_m']):
            gamma_m[i_local, j_local] = gamma[i_global, j_global]
    return gamma_m


def build_delta_m_matrix(delta, indices):
    delta_m = np.zeros((indices['n_m'], indices['p_m']))
    for i_local, i_global in enumerate(indices['links_m']):
        for j_local, j_global in enumerate(indices['paths_m']):
            delta_m[i_local, j_local] = delta[i_global, j_global]
    return delta_m
    

def build_delta_m_delta_mu_delta_um(delta, delta_m,gamma_m, indices, network, path_list):

    n = len(network.sn)
    p = sum(len(k) for k in path_list)
    n_m = indices["n_m"]
    m_m = indices["m_m"]
    p_m = indices["p_m"]
    links_m = indices["links_m"]
    paths_m = indices["paths_m"]

    delta_u_m = np.zeros((n-n_m,m_m))
    delta_uu = []
    i = 0     

    while i < len(links_m):

            A = np.vstack([delta_m[i, :], gamma_m])
                
            if np.linalg.matrix_rank(A) < m_m + 1:
                    delta_uu.append(delta_m[i, :].copy())
                    delta_m = np.delete(delta_m, i, axis=0)     
                    links_m = np.delete(links_m, i)             
            else:
                    i += 1

    final_indices = indices
    final_indices["links_m"] = links_m
    final_indices["m_m"] = len(links_m)


    cols_u = list(set(range(p)) - set(paths_m))
    cols_u.sort() 
    delta_m_u = delta[np.ix_(links_m, cols_u)] 

    delta_uu = np.array(delta_uu)
    delta_u_m_top = (pinv(gamma_m.T) @ delta_uu.T).T
    delta_u_m = np.vstack([delta_u_m_top, delta_u_m])

    return delta_m, delta_m_u, delta_u_m, final_indices 


# ---------- UTILITAIRES ----------
def simplify_binary(mat, tol=1e-8):
    binary_mat = np.zeros_like(mat)
    binary_mat[np.abs(mat) > tol] = 1
    return binary_mat
# ---------- ----- ----------


def rref(M):
    M_sym = Matrix(M)
    M1_sym, _ = M_sym.rref()
    M_rref = np.array(M1_sym, dtype=np.float64)
    rank_M = np.linalg.matrix_rank(M)
    M_rref = M_rref[:rank_M, :]
    return M_rref, rank_M

def build_A_Rr_and_r0(delta_m, gamma_m, final_indices, network):
    """ Construit les matrices Rr et r0 pour la réduction dimensionnelle. """

    delta_m_tilde, rank_delta_m = rref(delta_m)

    # Nullspace binaire
    M_for_null = (delta_m_tilde - pinv(gamma_m @ pinv(delta_m_tilde)) @ gamma_m).T
    ns = null_space(M_for_null)
    A = simplify_binary(ns).T

    # Matrice Rr
    Rr = A @ pinv(gamma_m @ pinv(delta_m_tilde))

    # Extraction de q_od restreint (OD multiples uniquement)
    q_od_m = np.array(network.q_od)[final_indices['od_m']].reshape(-1, 1)
    r0 = Rr @ q_od_m

    return A, Rr, r0, q_od_m

def build_B_q0(delta_m,delta_m_tilde, delta_m_u, network, final_indices):
    m = len(network.q_od)
    q_od = network.q_od
    od_m = final_indices["od_m"]
    q_od_u = np.array(q_od)[list(set(range(m)) - set(od_m))]
    q_od_u = q_od_u.reshape(-1,1)

    B = delta_m @ pinv(delta_m_tilde)
    q0 = delta_m_u @ q_od_u
    return B, q0, q_od_u

def build_T_m(T, final_indices):
    links_m = final_indices["links_m"]
    T = T[:,links_m]
    T_m, rank_T_m = rref(T)
    return T_m

def extract_dimensions(delta_m, delta_m_tilde, A, T_m):
    n1 = delta_m.shape[0]
    r1 = delta_m_tilde.shape[0]
    s1 = A.shape[0]
    u1 = T_m.shape[0]

    if u1 + s1 != r1:
        raise ValueError(
            f"Erreur : u1 + s1 ≠ r1 (u1={u1}, r1={r1}, s1={s1})"
        )
    return {
        "n1": n1,   
        "r1": r1,   
        "s1": s1,   
        "u1": u1   
    }

def _reconstruct_full_solution(q_m, t_m, network, delta, final_indices, t0_lin, K, delta_u_m, q_od_u, q_od_m):
    
    n = len(network.sn)
    p = delta.shape[1]
    
    links_m = final_indices["links_m"]
    paths_m = final_indices["paths_m"]
    
    links_u = sorted(list(set(range(n)) - set(links_m)))
    paths_u = sorted(list(set(range(p)) - set(paths_m)))
    delta_u = delta[np.ix_(links_u, paths_u)]
    
    q_u = delta_u @ q_od_u + delta_u_m @ q_od_m
    t_u = t0_lin[links_u].reshape(-1, 1) + K[np.ix_(links_u, links_u)] @ q_u
    
    lam_flows = np.zeros((n, 1))
    lam_flows[links_m] = q_m
    lam_flows[links_u] = q_u
    
    lam_times = np.zeros((n, 1))
    lam_times[links_m] = t_m
    lam_times[links_u] = t_u
    
    return lam_flows.flatten(), lam_times.flatten()
    
def lam_solver_linear_system(network, final_indices, dimensions, alpha, beta, eps, A, T_m, B, r0, q0, delta, delta_u_m, q_od_u, q_od_m):

    t0, C = network.t0, network.C
    t0_lin, K = cf.linearised_bpr_matrices(t0, C, alpha, beta, eps)
    links_m = final_indices["links_m"]
    n1, r1, s1, u1 = dimensions["n1"], dimensions["r1"], dimensions["s1"], dimensions["u1"]
    K_m = K[np.ix_(links_m, links_m)]
    t0_lin_m = t0_lin[links_m].reshape(-1, 1)

    M = np.block([
    [A,                np.zeros((s1, 2*n1))],
    [np.zeros((u1, r1+n1)), T_m],
    [-B,               np.eye(n1), np.zeros((n1, n1))],
    [np.zeros((n1, r1)), -K_m,       np.eye(n1)]
    ])

    y = np.vstack([r0,np.zeros((u1, 1)), q0, t0_lin_m])

    x = solve(M,y)

    r = x[:r1]
    q_m = x[r1:r1+n1]
    t_m= x[r1+n1:]

    return _reconstruct_full_solution(q_m, t_m, network, delta, final_indices, t0_lin, K, delta_u_m, q_od_u, q_od_m)

def lam_solver_qp(network, final_indices, dimensions, alpha, beta, eps, A, B, q0, r0, delta, delta_u_m, q_od_u, q_od_m):

    t0, C = network.t0, network.C
    t0_lin, K = cf.linearised_bpr_matrices(t0, C, alpha, beta, eps)
    n1, r1, s1, u1 = dimensions["n1"], dimensions["r1"], dimensions["s1"], dimensions["u1"]
    links_m = final_indices["links_m"]
    t0_lin_m = t0_lin[links_m].reshape(-1, 1)
    K_m = K[np.ix_(links_m, links_m)]

    M_top = np.hstack([B.T @ K_m @ B, A.T])
    M_bottom = np.hstack([A, np.zeros((s1, s1))])
    M = np.vstack([M_top, M_bottom])  

    y_top = - B.T @ K_m @ q0 - B.T @ t0_lin_m
    y_bottom = r0
    y = np.vstack([y_top, y_bottom])  

    x = solve(M, y)

    r = x[:r1]
    q_m = B @ r + q0
    t_m = t0_lin_m + K_m @ q_m

    return _reconstruct_full_solution(q_m, t_m, network, delta, final_indices, t0_lin, K, delta_u_m, q_od_u, q_od_m)
