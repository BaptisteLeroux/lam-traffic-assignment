import numpy as np
import warnings
from scipy.linalg import null_space
from sympy import Matrix

import modules.cost_functions as cf
from modules.robust_functions import (
    clean_matrix, 
    RobustnessConfig, 
    robust_pinv, 
    robust_solve
)


# ========== MATRICES DE BASE ==========

def build_gamma_matrix(path_list, network):
    """Construit la matrice γ (m x p) reliant OD aux chemins."""
    m = len(network.on)
    p = sum(len(k) for k in path_list)

    gamma = np.zeros((m, p))
    col_start = np.cumsum([0] + [len(k) for k in path_list[:-1]])
    for i, start in enumerate(col_start):
        gamma[i, start:start+len(path_list[i])] = 1
    
    print("Gamma :", gamma)
    return gamma


def build_delta_matrix(path_list, network, G):
    """Construit la matrice δ (n x p) reliant liens aux chemins."""
    n = len(network.sn)
    m = len(network.on)
    p = sum(len(k) for k in path_list)

    delta = np.zeros((n, p))
    nn = 0
    for i in range(m):
        for j, path in enumerate(path_list[i]):
            for k in range(len(path) - 1):
                u, v = path[k], path[k+1]
                arc_index = G[u][v]['index']
                delta[arc_index, nn + j] = 1
        nn += len(path_list[i])
    
    print("Delta :", delta)
    return delta


def build_T_matrix(path_list, network, G):
    """Construit la matrice T (p-m x n) pour les différences entre chemins."""
    n = len(network.sn)
    m = len(network.on)
    p = sum(len(k) for k in path_list)
    
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


# ========== EXTRACTION ET RÉDUCTION ==========

def extract_multiple_indices(path_list, delta):
    """Retourne les indices associés aux OD multipath."""
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
    multiple_link_indices = np.where(
        np.sum(delta[:, multiple_path_indices], axis=1) > 0
    )[0]
    
    return {
        "od_m": multiple_od_indices,
        "paths_m": multiple_path_indices,
        "links_m": multiple_link_indices,
        "m_m": len(multiple_od_indices),
        "p_m": len(multiple_path_indices),
        "n_m": len(multiple_link_indices)
    }


def build_gamma_m_matrix(gamma, indices):
    """Construit la matrice γ_m réduite aux ODs et chemins multiples."""
    gamma_m = np.zeros((indices['m_m'], indices['p_m']))
    for i_local, i_global in enumerate(indices['od_m']):
        for j_local, j_global in enumerate(indices['paths_m']):
            gamma_m[i_local, j_local] = gamma[i_global, j_global]
    return gamma_m


def build_delta_m_matrix(delta, indices):
    """Construit la matrice δ_m réduite aux liens et chemins multiples."""
    delta_m = np.zeros((indices['n_m'], indices['p_m']))
    for i_local, i_global in enumerate(indices['links_m']):
        for j_local, j_global in enumerate(indices['paths_m']):
            delta_m[i_local, j_local] = delta[i_global, j_global]
    return delta_m


def build_delta_m_delta_mu_delta_um(delta, delta_m, gamma_m, indices, network, path_list):
    """
    Construit les matrices δ réduites et élimine les lignes linéairement dépendantes.
    """
    n = len(network.sn)
    p = sum(len(k) for k in path_list)
    
    n_m = indices["n_m"]
    m_m = indices["m_m"]
    links_m = indices["links_m"].copy()
    paths_m = indices["paths_m"]
    
    delta_uu_rows = []
    eliminated_links_indices = []
    
    i = 0
    while i < len(links_m):
        A = np.vstack([delta_m[i, :], gamma_m])
        rank_test = np.linalg.matrix_rank(A, tol=RobustnessConfig.THRESHOLD_RANK)
        
        if rank_test < m_m + 1:
            if RobustnessConfig.VERBOSE:
                print(f"Ligne dépendante détectée: link {links_m[i]}, rank={rank_test} < {m_m + 1}")
            
            delta_uu_rows.append(delta_m[i, :].copy())
            eliminated_links_indices.append(links_m[i])
            delta_m = np.delete(delta_m, i, axis=0)
            links_m = np.delete(links_m, i)
        else:
            i += 1

    delta_m = clean_matrix(delta_m)
    
    final_indices = indices.copy()
    final_indices["links_m"] = links_m
    final_indices["n_m"] = len(links_m)

    # Construire δ_m^u
    cols_u = sorted(list(set(range(p)) - set(paths_m)))
    delta_m_u = delta[np.ix_(links_m, cols_u)]
    delta_m_u = clean_matrix(delta_m_u)

    # Reconstruction de delta_u_m
    all_links_indices = set(range(n))
    links_m_set = set(links_m)
    links_u = sorted(list(all_links_indices - links_m_set))
    final_indices["links_u"] = links_u

    if len(delta_uu_rows) > 0:
        delta_uu_rows = np.array(delta_uu_rows)
        delta_u_m_eliminated = (robust_pinv(gamma_m.T) @ delta_uu_rows.T).T
        delta_u_m_eliminated = clean_matrix(delta_u_m_eliminated)
    else:
        delta_u_m_eliminated = np.zeros((0, m_m))

    final_delta_u_m = np.zeros((len(links_u), m_m))
    
    for i, link_idx in enumerate(eliminated_links_indices):
        row_in_final = links_u.index(link_idx)
        final_delta_u_m[row_in_final, :] = final_delta_u_m[row_in_final, :] + delta_u_m_eliminated[i, :]

    return delta_m, delta_m_u, final_delta_u_m, final_indices


def rref(M):
    """Calcule la forme échelonnée réduite d'une matrice."""
    try:
        M_sym = Matrix(M)
        M1_sym, _ = M_sym.rref()
        M_rref = np.array(M1_sym.tolist(), dtype=np.float64)
    except Exception as e:
        print("rref problem:", e)
        raise e

    rank_M = np.linalg.matrix_rank(M_rref)
    M_rref = M_rref[:rank_M, :]
    
    return M_rref, rank_M


# ========== CONSTRUCTION DES MATRICES FINALES ==========

def build_A_Rr_and_r0(delta_m, gamma_m, final_indices, network):
    """Construit les matrices A (contraintes structurelles), R_r et r_0."""
    delta_m_tilde, rank_delta_m = rref(delta_m)

    gamma_pinv_delta_tilde = np.linalg.pinv(
        (gamma_m @ np.linalg.pinv(delta_m_tilde)).astype(float)
    )
    D = delta_m_tilde - gamma_pinv_delta_tilde @ gamma_m
    ns = null_space(D.T)
    A = ns.T

    R_r = A @ gamma_pinv_delta_tilde

    q_od_m = np.array(network.q_od)[final_indices['od_m']].reshape(-1, 1)
    r_0 = R_r @ q_od_m

    return A, R_r, r_0, q_od_m


def build_B_q0(delta_m, delta_m_tilde, delta_m_u, network, final_indices):
    """Construit la matrice B et le vecteur q_0."""
    m = len(network.q_od)
    q_od = network.q_od
    od_m = final_indices["od_m"]
    q_od_u = np.array(q_od)[list(set(range(m)) - set(od_m))]
    q_od_u = q_od_u.reshape(-1, 1)

    B = delta_m @ np.linalg.pinv(delta_m_tilde)
    q_0 = delta_m_u @ q_od_u
    
    return B, q_0, q_od_u


def build_T_m(T, final_indices):
    """Construit la matrice T réduite."""
    links_m = final_indices["links_m"]
    T = T[:, links_m]
    T_m, rank_T_m = rref(T)
    return T_m


def extract_dimensions(delta_m, delta_m_tilde, A, T_m):
    """Extrait les dimensions des matrices réduites."""
    n1 = delta_m.shape[0]
    r1 = delta_m_tilde.shape[0]
    s1 = A.shape[0]
    u1 = T_m.shape[0]

    if (u1 + s1 != r1) and (s1 != 0):
        warnings.warn(
            f"Dimensions non cohérentes: u1 + s1 ≠ r1 (u1={u1}, r1={r1}, s1={s1})"
        )
    
    return {"n1": n1, "r1": r1, "s1": s1, "u1": u1}


# ========== RECONSTRUCTION ==========

def _reconstruct_full_solution(q_m, t_m, network, delta, final_indices, 
                               t0_lin, K, delta_u_m, q_od_u, q_od_m):
    """Reconstruit la solution complète à partir des variables réduites."""
    n = len(network.sn)
    p = delta.shape[1]
    
    links_m = final_indices["links_m"]
    paths_m = final_indices["paths_m"]
    
    if "links_u" in final_indices:
        links_u = final_indices["links_u"]
    else:
        links_u = sorted(list(set(range(n)) - set(links_m)))
        
    paths_u = sorted(list(set(range(p)) - set(paths_m)))
    
    delta_u = delta[np.ix_(links_u, paths_u)]
    
    q_u = delta_u @ q_od_u + delta_u_m @ q_od_m
    q_u = clean_matrix(q_u)
    
    K_uu = K[np.ix_(links_u, links_u)]
    t_u = t0_lin[links_u].reshape(-1, 1) + K_uu @ q_u
    t_u = clean_matrix(t_u)
    
    lam_flows_full = np.zeros(n)
    lam_times_full = np.zeros(n)
    
    for i, link_idx in enumerate(links_m):
        lam_flows_full[link_idx] = q_m[i]
        lam_times_full[link_idx] = t_m[i]
        
    for i, link_idx in enumerate(links_u):
        lam_flows_full[link_idx] = q_u.flatten()[i]
        lam_times_full[link_idx] = t_u.flatten()[i]
    
    print("links_u, paths_u, links_m, paths_m :", links_u, paths_u, links_m, paths_m)
    
    lam_flows = clean_matrix(lam_flows_full.reshape(-1, 1))
    lam_times = clean_matrix(lam_times_full.reshape(-1, 1))
    
    return lam_flows.flatten(), lam_times.flatten()


# ========== SOLVEURS ==========

def lam_solver_linear_system(network, final_indices, dimensions, alpha, beta, eps_num, 
                             A, T_m, B, r_0, q_0, delta, delta_u_m, q_od_u, q_od_m):
    """Résout le système linéarisé via système linéaire (méthode 1)."""
    t0, C = network.t0, network.C
    t0_lin, K = cf.linearised_bpr_matrices(t0, C, alpha, beta, eps_num)
    links_m = final_indices["links_m"]
    n1, r1, s1, u1 = dimensions["n1"], dimensions["r1"], dimensions["s1"], dimensions["u1"]
    K_m = K[np.ix_(links_m, links_m)]
    t0_lin_m = t0_lin[links_m].reshape(-1, 1)

    M = np.block([
        [A, np.zeros((s1, 2*n1))],
        [np.zeros((u1, r1+n1)), T_m],
        [-B, np.eye(n1), np.zeros((n1, n1))],
        [np.zeros((n1, r1)), -K_m, np.eye(n1)]
    ])

    y = np.vstack([r_0, np.zeros((u1, 1)), q_0, t0_lin_m])

    x = np.linalg.solve(M, y)

    r = x[:r1]
    q_m = x[r1:r1+n1]
    t_m = x[r1+n1:]

    print('dimensions r1, u1, s1 :', r1, u1, s1)
    print('dimensions M, y, x :', M.shape, y.shape, x.shape)

    return _reconstruct_full_solution(
        q_m, t_m, network, delta, final_indices, 
        t0_lin, K, delta_u_m, q_od_u, q_od_m
    )


def lam_solver_qp(network, final_indices, dimensions, alpha, beta, eps_num, 
                 A, B, q0, r0, delta, delta_u_m, q_od_u, q_od_m):
    """Résout via formulation quadratique (méthode 2)."""
    t0, C = network.t0, network.C
    t0_lin, K = cf.linearised_bpr_matrices(t0, C, alpha, beta, eps_num)
    n1, r1, s1, u1 = dimensions["n1"], dimensions["r1"], dimensions["s1"], dimensions["u1"]
    links_m = final_indices["links_m"]
    t0_lin_m = t0_lin[links_m].reshape(-1, 1)
    K_m = K[np.ix_(links_m, links_m)]

    M_top = np.hstack([B.T @ K_m @ B, A.T])
    M_bottom = np.hstack([A, np.zeros((s1, s1))])
    M = np.vstack([M_top, M_bottom])

    y_top = -B.T @ K_m @ q0 - B.T @ t0_lin_m
    y_bottom = r0
    y = np.vstack([y_top, y_bottom])

    x = np.linalg.solve(M, y)

    r = x[:r1]
    q_m = B @ r + q0
    t_m = t0_lin_m + K_m @ q_m

    return _reconstruct_full_solution(
        q_m, t_m, network, delta, final_indices, 
        t0_lin, K, delta_u_m, q_od_u, q_od_m
    )


def lam_solver_qp_analytical(network, final_indices, dimensions, alpha, beta, eps_num, 
                             A, B, Rr, q0, r0, delta, delta_u_m, q_od_u, q_od_m):
    """Résout via formulation analytique optimisée (méthode 3)."""
    t0, C = network.t0, network.C
    t0_lin, K = cf.linearised_bpr_matrices(t0, C, alpha, beta, eps_num)
    n1, r1, s1, u1 = dimensions["n1"], dimensions["r1"], dimensions["s1"], dimensions["u1"]
    links_m = final_indices["links_m"]
    t0_lin_m = t0_lin[links_m].reshape(-1, 1)
    K_m = K[np.ix_(links_m, links_m)]

    M_top = np.hstack([B.T @ K_m @ B, A.T])
    M_bottom = np.hstack([A, np.zeros((s1, s1))])
    M = np.vstack([M_top, M_bottom])
    M = clean_matrix(M)

    M_inv = robust_solve(M, np.eye(M.shape[0]), method="auto")
    M_inv = clean_matrix(M_inv)

    bkb = np.linalg.inv(B.T @ K_m @ B)
    abkba = np.linalg.inv(A @ bkb @ A.T)
    Mrr = bkb - bkb @ A.T @ abkba @ A @ bkb
    Mrl = bkb @ A.T @ abkba    

    Rt = -B @ Mrr @ B.T
    Rq = B @ Mrl @ Rr
    q_m = Rt @ (t0_lin_m + K_m @ q0) + Rq @ q_od_m + q0
    t_m = t0_lin_m + K_m @ q_m

    return _reconstruct_full_solution(
        q_m, t_m, network, delta, final_indices, 
        t0_lin, K, delta_u_m, q_od_u, q_od_m
    )


# ========== FONCTION PRINCIPALE ==========

def compute_lam_solution(network, path_list, G, eps_num, method, alpha, beta):
    """
    Calcule la solution LAM complète.
    
    Args:
        network: Objet Network
        path_list: Liste des chemins trouvés par MSA
        G: Graphe NetworkX
        eps_num: Niveau de congestion de référence
        method: Méthode de résolution ('qp', 'linear_system', 'qp_analytical')
        alpha, beta: Paramètres BPR
    
    Returns:
        lam_flows, lam_times: Solutions LAM
    """
    # 1. Construction des matrices de base
    gamma = build_gamma_matrix(path_list, network)
    delta = build_delta_matrix(path_list, network, G)
    T = build_T_matrix(path_list, network, G)
    
    # 2. Extraction des indices multiples
    indices = extract_multiple_indices(path_list, delta)
    gamma_m = build_gamma_m_matrix(gamma, indices)
    delta_m = build_delta_m_matrix(delta, indices)
    
    # 3. Réduction dimensionnelle
    delta_m, delta_m_u, delta_u_m, final_indices = build_delta_m_delta_mu_delta_um(
        delta, delta_m, gamma_m, indices, network, path_list
    )
    
    delta_m_tilde, _ = rref(delta_m)
    
    # 4. Construction des matrices finales
    A, Rr, r0, q_od_m = build_A_Rr_and_r0(delta_m, gamma_m, final_indices, network)
    B, q0, q_od_u = build_B_q0(delta_m, delta_m_tilde, delta_m_u, network, final_indices)
    T_m = build_T_m(T, final_indices)
    dimensions = extract_dimensions(delta_m, delta_m_tilde, A, T_m)
    
    # 5. Résolution selon la méthode choisie
    if method == 'qp':
        lam_flows, lam_times = lam_solver_qp(
            network, final_indices, dimensions, alpha, beta, eps_num,
            A, B, q0, r0, delta, delta_u_m, q_od_u, q_od_m
        )
    elif method == 'linear_system':
        lam_flows, lam_times = lam_solver_linear_system(
            network, final_indices, dimensions, alpha, beta, eps_num,
            A, T_m, B, r0, q0, delta, delta_u_m, q_od_u, q_od_m
        )
    elif method == 'qp_analytical':
        lam_flows, lam_times = lam_solver_qp_analytical(
            network, final_indices, dimensions, alpha, beta, eps_num, 
            A, B, Rr, q0, r0, delta, delta_u_m, q_od_u, q_od_m
        )
    else:
        raise ValueError(
            f"Méthode inconnue: {method}. "
            "Utilisez 'qp', 'linear_system' ou 'qp_analytical'."
        )
    
    _analyze_and_print_path_costs(lam_times, path_list, network, G)
    
    return lam_flows, lam_times


# ========== ANALYSEUR DE CHEMINS ==========

def _analyze_and_print_path_costs(lam_times, path_list, network, G):
    """Analyse et affiche les coûts des chemins."""
    print("\n" + "="*80)
    print("ANALYSE DES COÛTS DE CHEMINS (BASÉE SUR LES TEMPS LAM FINAUX)")
    print("="*80)
    
    m = len(network.on)
    
    for k in range(m):
        origin = network.on[k]
        dest = network.dn[k]
        demand = network.q_od[k]
        paths_for_od = path_list[k]
        
        print(f"\n--- Paire OD {k+1}: {origin} -> {dest} (Demande = {demand:.2f}) ---")
        
        if not paths_for_od:
            print("    Aucun chemin trouvé pour cette OD.")
            continue
            
        path_costs = []
        for path in paths_for_od:
            cost = 0.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                try:
                    link_idx = G[u][v]['index']
                    cost += lam_times[link_idx]
                except KeyError:
                    print(f"    ERREUR: Arc {u}->{v} non trouvé dans le graphe G.")
            path_costs.append((cost, path))
            
        if not path_costs:
            print("    Aucun coût n'a pu être calculé.")
            continue

        min_cost = min(c for c, p in path_costs)
        
        print(f"    Coût minimum (Équilibre): {min_cost:.6f}")
        
        print("    Chemins Actifs (se partagent la demande) :")
        count_active = 0
        for cost, path in path_costs:
            if np.isclose(cost, min_cost, atol=1e-6):
                path_str = " -> ".join(map(str, path))
                print(f"      - [Coût: {cost:.6f}] {path_str}")
                count_active += 1
        
        if count_active == 0:
             print("      (Aucun chemin actif trouvé - étrange)")
             
        inactive_paths = [(c, p) for c, p in path_costs 
                          if not np.isclose(c, min_cost, atol=1e-6)]
        if inactive_paths:
            print("\n    Chemins Inactifs (plus chers) :")
            for cost, path in inactive_paths:
                path_str = " -> ".join(map(str, path))
                print(f"      - [Coût: {cost:.6f}] {path_str}")
                
    print("="*80 + "\n")