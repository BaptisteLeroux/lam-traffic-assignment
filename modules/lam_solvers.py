from scipy.linalg import pinv as scipy_pinv
from numpy.linalg import pinv as numpy_pinv
import numpy as np
import warnings
from sympy import Matrix
import modules.cost_functions as cf
from modules.robust_functions import clean_matrix, diagnose_matrix, RobustnessConfig, robust_pinv, robust_null_space, robust_solve


# ========== MATRICES DE BASE ==========

def build_gamma_matrix(path_list, network):
    """Construit la matrice γ (m x p) reliant OD aux chemins."""
    n = len(network.sn)
    m = len(network.on)
    p = sum(len(k) for k in path_list)

    gamma = np.zeros((m, p))
    col_start = np.cumsum([0] + [len(k) for k in path_list[:-1]])
    for i, start in enumerate(col_start):
        gamma[i, start:start+len(path_list[i])] = 1
    
    return clean_matrix(gamma)


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
    
    return clean_matrix(delta)


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
    
    return clean_matrix(T)


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
    """Construit la matrice γ_m réduite aux ODs et chemins multiples."""
    gamma_m = np.zeros((indices['m_m'], indices['p_m']))
    for i_local, i_global in enumerate(indices['od_m']):
        for j_local, j_global in enumerate(indices['paths_m']):
            gamma_m[i_local, j_local] = gamma[i_global, j_global]
    return clean_matrix(gamma_m)


def build_delta_m_matrix(delta, indices):
    """Construit la matrice δ_m réduite aux liens et chemins multiples."""
    delta_m = np.zeros((indices['n_m'], indices['p_m']))
    for i_local, i_global in enumerate(indices['links_m']):
        for j_local, j_global in enumerate(indices['paths_m']):
            delta_m[i_local, j_local] = delta[i_global, j_global]
    return clean_matrix(delta_m)
    

def build_delta_m_delta_mu_delta_um(delta, delta_m, gamma_m, indices, network, path_list):
    """
    Construit les matrices δ réduites et élimine les lignes linéairement dépendantes.
    
    Cette fonction implémente la logique MATLAB (linaff.m lignes 64-74) pour détecter
    et supprimer les lignes de δ_m qui sont linéairement dépendantes de γ_m.
    
    C'est CRITIQUE pour la robustesse du système.
    """
    n = len(network.sn)
    p = sum(len(k) for k in path_list)
    n_m = indices["n_m"]
    m_m = indices["m_m"]
    p_m = indices["p_m"]
    links_m = indices["links_m"].copy()
    paths_m = indices["paths_m"]

    delta_u_m = np.zeros((n-n_m, m_m))
    delta_uu = []
    i = 0

    # Élimination des dépendances linéaires (comme MATLAB)
    while i < len(links_m):
        # Former la matrice augmentée [δ_m[i,:]; γ_m]
        A = np.vstack([delta_m[i, :], gamma_m])
        
        # Tester le rang avec tolérance robuste
        rank_test = np.linalg.matrix_rank(A, tol=RobustnessConfig.THRESHOLD_RANK)
        
        if rank_test < m_m + 1:
            # Ligne dépendante détectée
            if RobustnessConfig.VERBOSE:
                print(f"Ligne dépendante détectée: link {links_m[i]}, rank={rank_test} < {m_m + 1}")
            
            delta_uu.append(delta_m[i, :].copy())
            delta_m = np.delete(delta_m, i, axis=0)
            links_m = np.delete(links_m, i)
        else:
            i += 1

    # Nettoyer delta_m
    delta_m = clean_matrix(delta_m)
    
    final_indices = indices.copy()
    final_indices["links_m"] = links_m
    final_indices["n_m"] = len(links_m)

    # Construire δ_m^u (liens multiples, chemins uniques)
    cols_u = list(set(range(p)) - set(paths_m))
    cols_u.sort()
    delta_m_u = delta[np.ix_(links_m, cols_u)]
    delta_m_u = clean_matrix(delta_m_u)

    # Construire δ_u^m (liens uniques, chemins multiples)
    if len(delta_uu) > 0:
        delta_uu = np.array(delta_uu)
        # Utiliser robust_pinv au lieu de pinv standard
        delta_u_m_top = (robust_pinv(gamma_m.T) @ delta_uu.T).T
        delta_u_m_top = clean_matrix(delta_u_m_top)
    else:
        delta_uu = np.zeros((0, delta_m.shape[1]))
        delta_u_m_top = np.zeros((0, m_m))

    delta_u_m = np.vstack([delta_u_m_top, delta_u_m])
    delta_u_m = clean_matrix(delta_u_m)

    return delta_m, delta_m_u, delta_u_m, final_indices


# ========== UTILITAIRES ==========

def simplify_binary(mat, tol=None):
    """Convertit une matrice en matrice binaire."""
    if tol is None:
        tol = RobustnessConfig.THRESHOLD_CLEAN
    
    binary_mat = np.zeros_like(mat)
    binary_mat[np.abs(mat) > tol] = 1
    return binary_mat


def rref(M):
    """
    Calcule la forme échelonnée réduite d'une matrice (RREF) de manière robuste.
    
    Équivalent MATLAB: [M_rref, ~] = rref(M)
    
    Args:
        M: Matrice d'entrée
    
    Returns:
        M_rref: Forme échelonnée réduite
        rank_M: Rang de la matrice
    """
    M = clean_matrix(M)
    
    # Méthode 1: Via sympy (plus lente mais exacte)
    try:
        M_sym = Matrix(M)
        M1_sym, _ = M_sym.rref()
        M_rref = np.array(M1_sym, dtype=np.float64)
    except:
        # Fallback: implémentation manuelle
        M_rref = _rref_manual(M)
    
    # Nettoyer et extraire les lignes non nulles
    M_rref = clean_matrix(M_rref, threshold=RobustnessConfig.THRESHOLD_RREF)
    rank_M = np.linalg.matrix_rank(M_rref, tol=RobustnessConfig.THRESHOLD_RANK)
    M_rref = M_rref[:rank_M, :]
    
    return M_rref, rank_M


def _rref_manual(A):
    """Implémentation manuelle robuste de RREF (fallback)."""
    A = A.astype(float).copy()
    rows, cols = A.shape
    r = 0  # Current row
    
    for c in range(cols):
        if r >= rows:
            break
        
        # Trouver le pivot
        pivot_row = np.argmax(np.abs(A[r:rows, c])) + r
        
        if np.abs(A[pivot_row, c]) < RobustnessConfig.THRESHOLD_RREF:
            continue
        
        # Échanger les lignes
        A[[r, pivot_row]] = A[[pivot_row, r]]
        
        # Normaliser la ligne du pivot
        A[r] = A[r] / A[r, c]
        
        # Éliminer
        for i in range(rows):
            if i != r and np.abs(A[i, c]) > RobustnessConfig.THRESHOLD_RREF:
                A[i] -= A[i, c] * A[r]
        
        r += 1
    
    # Nettoyer
    A = clean_matrix(A, threshold=RobustnessConfig.THRESHOLD_RREF)
    
    return A


# ========== CONSTRUCTION DES MATRICES FINALES ==========

def build_A_Rr_and_r0(delta_m, gamma_m, final_indices, network):
    """
    Construit les matrices A (contraintes structurelles), R_r et r_0.
    
    Suit la logique MATLAB pour calculer le null space et les contraintes.
    """
    # RREF robuste
    delta_m_tilde, rank_delta_m = rref(delta_m)
    delta_m_tilde = clean_matrix(delta_m_tilde)
    
    if RobustnessConfig.VERBOSE:
        print(f"δ_m: shape={delta_m.shape}, rank={rank_delta_m}")
        diagnose_matrix(delta_m_tilde, "δ_m_tilde")

    # Calcul de D = δ̃_m - pinv(γ_m @ pinv(δ̃_m)) @ γ_m
    # Équivalent MATLAB: D = Q1 - pinv(P*pinv(Q1))*P
    gamma_pinv_delta_tilde = robust_pinv(gamma_m @ robust_pinv(delta_m_tilde))
    D = delta_m_tilde - gamma_pinv_delta_tilde @ gamma_m
    D = clean_matrix(D, threshold=RobustnessConfig.THRESHOLD_CLEAN)
    
    if RobustnessConfig.VERBOSE:
        diagnose_matrix(D, "D")

    # Null space robuste de D.T (équivalent MATLAB: null(D', 'rational'))
    ns = robust_null_space(D)
    A = simplify_binary(ns).T
    A = clean_matrix(A)
    
    if RobustnessConfig.VERBOSE:
        print(f"A: shape={A.shape}")

    # R_r = A @ pinv(γ_m @ pinv(δ̃_m))
    R_r = A @ gamma_pinv_delta_tilde
    R_r = clean_matrix(R_r)

    # r_0 = R_r @ q^{od}_m
    q_od_m = np.array(network.q_od)[final_indices['od_m']].reshape(-1, 1)
    r_0 = R_r @ q_od_m
    r_0 = clean_matrix(r_0)

    return A, R_r, r_0, q_od_m


def build_B_q0(delta_m, delta_m_tilde, delta_m_u, network, final_indices):
    """Construit la matrice B et le vecteur q_0."""
    m = len(network.q_od)
    q_od = network.q_od
    od_m = final_indices["od_m"]
    q_od_u = np.array(q_od)[list(set(range(m)) - set(od_m))]
    q_od_u = q_od_u.reshape(-1, 1)

    # B = δ_m @ pinv(δ̃_m)
    B = delta_m @ robust_pinv(delta_m_tilde)
    B = clean_matrix(B)
    
    # q_0 = δ_m^u @ q^{od}_u
    q_0 = delta_m_u @ q_od_u
    q_0 = clean_matrix(q_0)
    
    return B, q_0, q_od_u


def build_T_m(T, final_indices):
    """Construit la matrice T réduite."""
    links_m = final_indices["links_m"]
    T = T[:, links_m]
    T_m, rank_T_m = rref(T)
    T_m = clean_matrix(T_m)
    return T_m


def extract_dimensions(delta_m, delta_m_tilde, A, T_m):
    """Extrait les dimensions des matrices réduites."""
    n1 = delta_m.shape[0]
    r1 = delta_m_tilde.shape[0]
    s1 = A.shape[0]
    u1 = T_m.shape[0]

    if (u1 + s1 != r1) and (s1 != 0):
        warnings.warn(f"Dimensions non cohérentes: u1 + s1 ≠ r1 (u1={u1}, r1={r1}, s1={s1})")
        # Ne pas lever d'erreur, continuer avec un avertissement
    
    return {"n1": n1, "r1": r1, "s1": s1, "u1": u1}


# ========== RECONSTRUCTION ==========

def _reconstruct_full_solution(q_m, t_m, network, delta, final_indices, t0_lin, K, delta_u_m, q_od_u, q_od_m):
    """Reconstruit la solution complète à partir des variables réduites."""
    n = len(network.sn)
    p = delta.shape[1]
    
    links_m = final_indices["links_m"]
    paths_m = final_indices["paths_m"]
    
    links_u = sorted(list(set(range(n)) - set(links_m)))
    paths_u = sorted(list(set(range(p)) - set(paths_m)))
    delta_u = delta[np.ix_(links_u, paths_u)]
    
    # Flux sur les liens uniques
    q_u = delta_u @ q_od_u + delta_u_m @ q_od_m
    q_u = clean_matrix(q_u)
    
    # Temps sur les liens uniques
    K_uu = K[np.ix_(links_u, links_u)]
    t_u = t0_lin[links_u].reshape(-1, 1) + K_uu @ q_u
    t_u = clean_matrix(t_u)
    
    # Reconstruction complète
    lam_flows = np.zeros((n, 1))
    lam_flows[links_m] = q_m
    lam_flows[links_u] = q_u
    
    lam_times = np.zeros((n, 1))
    lam_times[links_m] = t_m
    lam_times[links_u] = t_u
    
    # Nettoyer et retourner
    lam_flows = clean_matrix(lam_flows)
    lam_times = clean_matrix(lam_times)
    
    return lam_flows.flatten(), lam_times.flatten()


# ========== SOLVEURS ==========

def lam_solver_linear_system(network, final_indices, dimensions, alpha, beta, eps_num, 
                             A, T_m, B, r_0, q_0, delta, delta_u_m, q_od_u, q_od_m):
    """
    Résout le système linéarisé via système linéaire (méthode 1).
    
    Système:
    [ A    0    0    0  ] [ r  ]   [ r_0        ]
    [ 0    0    0   T_m ] [ q_m] = [ 0          ]
    [-B   I_n   0    0  ] [ t_m]   [ q_0        ]
    [ 0    0  -K_m  I_n ] [λ_m ]   [ t_{0,lin}^m]
    """
    t0, C = network.t0, network.C
    t0_lin, K = cf.linearised_bpr_matrices(t0, C, alpha, beta, eps_num)
    links_m = final_indices["links_m"]
    n1, r1, s1, u1 = dimensions["n1"], dimensions["r1"], dimensions["s1"], dimensions["u1"]
    K_m = K[np.ix_(links_m, links_m)]
    t0_lin_m = t0_lin[links_m].reshape(-1, 1)

    # Construction du système
    M = np.block([
        [A, np.zeros((s1, 2*n1))],
        [np.zeros((u1, r1+n1)), T_m],
        [-B, np.eye(n1), np.zeros((n1, n1))],
        [np.zeros((n1, r1)), -K_m, np.eye(n1)]
    ])
    M = clean_matrix(M)

    y = np.vstack([r_0, np.zeros((u1, 1)), q_0, t0_lin_m])
    y = clean_matrix(y)

    # Diagnostic
    if RobustnessConfig.DIAGNOSTICS:
        diagnose_matrix(M, "M (linear_system)")

    # Résolution robuste
    x = robust_solve(M, y, method='auto')

    # Extraction
    r = x[:r1]
    q_m = x[r1:r1+n1]
    t_m = x[r1+n1:]

    print('dimensions r1, u1, s1 : ',r1, u1, s1)
    print('dimensions m , y , x : ', M.shape, y.shape, x.shape)

    return _reconstruct_full_solution(q_m, t_m, network, delta, final_indices, t0_lin, K, delta_u_m, q_od_u, q_od_m)

"""
def lam_solver_qp(network, final_indices, dimensions, alpha, beta, eps_num, 
                 A, B, q0, r0, delta, delta_u_m, q_od_u, q_od_m):

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

    x = robust_solve(M, y)

    r = x[:r1]
    q_m = B @ r + q0
    t_m = t0_lin_m + K_m @ q_m

    return _reconstruct_full_solution(q_m, t_m, network, delta, final_indices, t0_lin, K, delta_u_m, q_od_u, q_od_m)

"""
def lam_solver_qp(network, final_indices, dimensions, alpha, beta, eps_num, 
                 A, B, q0, r0, delta, delta_u_m, q_od_u, q_od_m):
    """
    Résout le système linéarisé via problème quadratique (version robuste).
    
    Système équivalent à résoudre:
    min_{r} (1/2) r^T (B^T K_m B) r + (B^T K_m q_0 + B^T t_{0,lin}^m)^T r
    s.c.   A r = r_0
    
    Conditions KKT:
    [ B^T K_m B    A^T ] [ r  ]   [ -B^T K_m q_0 - B^T t_{0,lin}^m ]
    [    A          0  ] [ λ  ] = [          r_0                    ]
    """
    # Extraction des données
    t0, C = network.t0, network.C
    t0_lin, K = cf.linearised_bpr_matrices(t0, C, alpha, beta, eps_num)
    n1, r1, s1, u1 = dimensions["n1"], dimensions["r1"], dimensions["s1"], dimensions["u1"]
    links_m = final_indices["links_m"]
    
    # Préparation des matrices locales
    t0_lin_m = t0_lin[links_m].reshape(-1, 1)
    K_m = K[np.ix_(links_m, links_m)]
    
    # Nettoyage préalable
    K_m = clean_matrix(K_m)
    B = clean_matrix(B)
    A = clean_matrix(A)
    q0 = clean_matrix(q0)
    r0 = clean_matrix(r0)
    t0_lin_m = clean_matrix(t0_lin_m)
    
    # Construction de la matrice hessienne H = B^T K_m B
    # Avec régularisation si nécessaire
    H = B.T @ K_m @ B
    H = clean_matrix(H)
    
    # Vérifier le conditionnement de H
    if RobustnessConfig.DIAGNOSTICS:
        cond_H = np.linalg.cond(H)
        if cond_H > RobustnessConfig.MAX_CONDITION:
            warnings.warn(f"Matrice H mal conditionnée (cond={cond_H:.2e}), ajout de régularisation.")
            # Régularisation de Tikhonov adaptative
            lambda_reg = RobustnessConfig.THRESHOLD_CLEAN * np.trace(H) / H.shape[0]
            H += lambda_reg * np.eye(H.shape[0])
            H = clean_matrix(H)
    
    # Construction de la matrice KKT
    if s1 > 0:  # Si contraintes structurelles présentes
        M_top = np.hstack([H, A.T])
        M_bottom = np.hstack([A, np.zeros((s1, s1))])
        M = np.vstack([M_top, M_bottom])
        
        # Vecteur second membre
        y_top = -B.T @ K_m @ q0 - B.T @ t0_lin_m
        y_bottom = r0
        y = np.vstack([y_top, y_bottom])
    else:  # Cas s1 = 0: pas de contraintes structurelles
        M = H
        y = -B.T @ K_m @ q0 - B.T @ t0_lin_m
    
    # Nettoyage final
    M = clean_matrix(M)
    y = clean_matrix(y)
    
    # Diagnostic détaillé
    if RobustnessConfig.DIAGNOSTICS:
        diagnose_matrix(M, "M (QP)")
        diagnose_matrix(H, "H (Hessian)")
        if s1 > 0:
            diagnose_matrix(A, "A (constraints)")
    
    # Résolution robuste avec détection automatique de méthode
    x = robust_solve(M, y, method='auto')
    
    # Extraction de la solution
    if s1 > 0:
        r = x[:r1]
        # λ = x[r1:r1+s1]  # Multiplicateurs de Lagrange (optionnel)
    else:
        r = x
    
    # Reconstruction des flux et temps
    q_m = B @ r + q0
    q_m = clean_matrix(q_m)
    
    t_m = t0_lin_m + K_m @ q_m
    t_m = clean_matrix(t_m)
    
    # Vérification de la satisfaction des contraintes (si présentes)
    if RobustnessConfig.VERBOSE and s1 > 0:
        constraint_residual = np.linalg.norm(A @ r - r0)
        print(f"Résidu des contraintes: {constraint_residual:.2e}")
        if constraint_residual > RobustnessConfig.THRESHOLD_CLEAN * 10:
            warnings.warn(f"Contraintes mal satisfaites: résidu = {constraint_residual:.2e}")
    
    if RobustnessConfig.VERBOSE:
        print(f'Dimensions QP: r1={r1}, s1={s1}, n1={n1}')
        print(f'Shape M: {M.shape}, y: {y.shape}, x: {x.shape}')
    
    # Reconstruction de la solution complète
    return _reconstruct_full_solution(
        q_m, t_m, network, delta, final_indices, 
        t0_lin, K, delta_u_m, q_od_u, q_od_m
    )

# ========== FONCTION PRINCIPALE ==========

def compute_lam_solution(network, path_list, G, eps_num, method='qp', alpha=0.15, beta=4):
    """
    Calcule la solution analytique LAM.
    
    Args:
        network: Objet Network
        path_list: Liste des chemins trouvÃ©s par MSA
        G: Graphe NetworkX
        eps_num: Niveau de congestion moyen (flow/capacity)
        method: 'qp' ou 'linear_system'
        alpha, beta: ParamÃ¨tres BPR
    
    Returns:
        lam_flows, lam_times: Flux et temps de parcours analytiques
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

    """     if A.shape[0] == 0:  # s1 = 0 case
        print("Note: s1 = 0 detected, using direct path formulation")
        return lam_solver_direct_path(
            network, final_indices, dimensions, alpha, beta, eps_num,
            gamma_m, delta_m, T_m, q0, q_od_m, delta, delta_u_m, q_od_u
        ) """
    
    # 5. Résolution selon la mÃ©thode choisie
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
    else:
        raise ValueError(f"Méthode inconnue: {method}. Utilisez 'qp' ou 'linear_system'.")
    
    return lam_flows, lam_times



def analyze_conditioning(M, tol=1e-12):
    """Analyse le conditionnement de la matrice M et retourne le mode de rÃ©solution appropriÃ©."""
    U, s, Vt = np.linalg.svd(M)
    sigma_min = np.min(s)
    sigma_max = np.max(s)
    cond = sigma_max / max(sigma_min, tol)
    
    if cond < 1e3:
        method = 'inv'       # inversion directe
    elif cond < 1e6:
        method = 'pinv'      # pseudo-inverse stable
    elif cond < 1e10:
        method = 'lstsq'     # rÃ©solution moindres carrÃ©s
    else:
        method = 'tikhonov'  # rÃ©gularisation
    
    return cond, method