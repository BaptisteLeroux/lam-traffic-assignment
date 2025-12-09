import numpy as np
import warnings
from scipy.linalg import null_space
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp

import modules.cost_functions as cf
from modules.robust_functions import (
    clean_matrix, 
    RobustnessConfig, 
    robust_pinv, 
    robust_solve
)


# ========== OPTIMISATION 1: RREF RAPIDE SANS SYMPY ==========

def rref_fast(M, tol=1e-10):
    """
    RREF optimisé sans SymPy - beaucoup plus rapide.
    Utilise l'élimination de Gauss numérique directe.
    """
    M = np.array(M, dtype=np.float64)
    rows, cols = M.shape
    
    # Copie de travail
    A = M.copy()
    
    pivot_row = 0
    for col in range(cols):
        if pivot_row >= rows:
            break
            
        # Trouver le pivot
        max_row = np.argmax(np.abs(A[pivot_row:, col])) + pivot_row
        
        if abs(A[max_row, col]) < tol:
            continue
            
        # Échanger les lignes
        if max_row != pivot_row:
            A[[pivot_row, max_row]] = A[[max_row, pivot_row]]
        
        # Normaliser la ligne pivot
        A[pivot_row] = A[pivot_row] / A[pivot_row, col]
        
        # Éliminer les autres lignes
        for i in range(rows):
            if i != pivot_row and abs(A[i, col]) > tol:
                A[i] -= A[i, col] * A[pivot_row]
        
        pivot_row += 1
    
    # Supprimer les lignes nulles
    rank = 0
    for i in range(rows):
        if np.any(np.abs(A[i]) > tol):
            rank += 1
        else:
            break
    
    return A[:rank], rank


# ========== OPTIMISATION 2: MATRICES CREUSES ==========

def build_gamma_matrix_sparse(path_list, network):
    """Version creuse de build_gamma_matrix."""
    m = len(network.on)
    p = sum(len(k) for k in path_list)
    
    row_indices = []
    col_indices = []
    data = []
    
    col_start = 0
    for i in range(m):
        n_paths = len(path_list[i])
        for j in range(n_paths):
            row_indices.append(i)
            col_indices.append(col_start + j)
            data.append(1.0)
        col_start += n_paths
    
    gamma = csr_matrix((data, (row_indices, col_indices)), shape=(m, p))
    return gamma


def build_delta_matrix_sparse(path_list, network, G):
    """Version creuse de build_delta_matrix."""
    n = len(network.sn)
    m = len(network.on)
    p = sum(len(k) for k in path_list)
    
    row_indices = []
    col_indices = []
    data = []
    
    col_idx = 0
    for i in range(m):
        for path in path_list[i]:
            for k in range(len(path) - 1):
                u, v = path[k], path[k+1]
                arc_index = G[u][v]['index']
                row_indices.append(arc_index)
                col_indices.append(col_idx)
                data.append(1.0)
            col_idx += 1
    
    delta = csr_matrix((data, (row_indices, col_indices)), shape=(n, p))
    return delta


# ========== OPTIMISATION 3: CACHE DES INVERSIONS ==========

class MatrixCache:
    """Cache pour les inversions matricielles coûteuses."""
    def __init__(self):
        self.cache = {}
    
    def get_pinv(self, key, matrix, sparse=False):
        """Récupère ou calcule la pseudo-inverse."""
        if key not in self.cache:
            if sparse and issparse(matrix):
                # Pour matrices creuses, utiliser solve plutôt que pinv
                self.cache[key] = None  # Pas d'inverse stockée, on utilisera spsolve
            else:
                self.cache[key] = robust_pinv(matrix.toarray() if issparse(matrix) else matrix)
        return self.cache[key]
    
    def get_inv(self, key, matrix):
        """Récupère ou calcule l'inverse."""
        if key not in self.cache:
            if issparse(matrix):
                matrix = matrix.toarray()
            self.cache[key] = np.linalg.inv(matrix)
        return self.cache[key]


# ========== OPTIMISATION 4: SOLVEUR QP ANALYTIQUE OPTIMISÉ ==========

def lam_solver_qp_analytical_fast(network, final_indices, dimensions, alpha, beta, eps_num, 
                                   A, B, Rr, q0, r0, delta, delta_u_m, q_od_u, q_od_m,
                                   gamma, delta_m_u, gamma_m, cache=None):
    """
    Version optimisée du solveur QP analytique avec:
    - Cache des inversions matricielles
    - Évitement des calculs redondants
    - Gestion optimale de la mémoire
    """
    if cache is None:
        cache = MatrixCache()
    
    t0, C = network.t0, network.C
    t0_lin, K = cf.linearised_bpr_matrices(t0, C, alpha, beta, eps_num)
    
    n1, r1, s1 = dimensions["n1"], dimensions["r1"], dimensions["s1"]
    links_m = final_indices["links_m"]
    t0_lin_m = t0_lin[links_m].reshape(-1, 1)
    K_m = K[np.ix_(links_m, links_m)]
    
    # Utiliser le cache pour les inversions
    BtKB = B.T @ K_m @ B
    
    # Vérifier si déjà en cache
    bkb_inv = cache.get_inv('bkb', BtKB)
    
    ABKBAt = A @ bkb_inv @ A.T
    abkba_inv = cache.get_inv('abkba', ABKBAt)
    
    # Matrices de réponse (optimisé)
    Mrr = bkb_inv - bkb_inv @ A.T @ abkba_inv @ A @ bkb_inv
    Mrl = bkb_inv @ A.T @ abkba_inv
    
    Rt = -B @ Mrr @ B.T
    Rq = B @ Mrl @ Rr
    
    # Calcul des flux/temps
    q_m = Rt @ (t0_lin_m + K_m @ q0) + Rq @ q_od_m + q0
    t_m = t0_lin_m + K_m @ q_m
    
    return _reconstruct_full_solution_fast(
        q_m, t_m, network, delta, final_indices, 
        t0_lin, K, delta_u_m, q_od_u, q_od_m,
        gamma, Rt, Rq, delta_m_u, gamma_m
    )


def _reconstruct_full_solution_fast(q_m, t_m, network, delta, final_indices, 
                                     t0_lin, K, delta_u_m, q_od_u, q_od_m,
                                     gamma, R_t, R_q, delta_m_u, gamma_m):
    """Version optimisée de la reconstruction."""
    n = len(network.sn)
    m = len(network.on)
    
    # Convertir delta en array dense si nécessaire
    if issparse(delta):
        delta = delta.toarray()
    
    links_m = final_indices["links_m"]
    paths_m = final_indices["paths_m"]
    
    if "links_u" in final_indices:
        links_u = final_indices["links_u"]
    else:
        links_u = sorted(list(set(range(n)) - set(links_m)))
    
    # Identifier les chemins uniques (non multiples)
    p = delta.shape[1]
    paths_u = sorted(list(set(range(p)) - set(paths_m)))
    
    # Extraire delta_u correctement : liens_u × chemins_u
    delta_u = delta[np.ix_(links_u, paths_u)]
    
    # Vérifier les dimensions
    expected_dim_u = len(paths_u)
    if q_od_u.shape[0] != expected_dim_u:
        print(f"WARNING: q_od_u dimension mismatch: {q_od_u.shape[0]} vs expected {expected_dim_u}")
        # Correction si nécessaire
        if q_od_u.shape[0] < expected_dim_u:
            q_od_u_corrected = np.zeros((expected_dim_u, 1))
            q_od_u_corrected[:q_od_u.shape[0]] = q_od_u
            q_od_u = q_od_u_corrected
    
    # Reconstruction vectorisée
    lam_flows_full = np.zeros(n)
    lam_times_full = np.zeros(n)
    
    lam_flows_full[links_m] = q_m.flatten()
    lam_times_full[links_m] = t_m.flatten()
    
    # Calcul des flux/temps pour liens uniques
    if len(links_u) > 0 and len(paths_u) > 0:
        # q_u = delta_u @ q_od_u + delta_u_m @ q_od_m
        q_u = delta_u @ q_od_u
        if delta_u_m.shape[0] > 0 and delta_u_m.shape[1] > 0:
            q_u += delta_u_m @ q_od_m
        
        q_u = clean_matrix(q_u)
        
        K_uu = K[np.ix_(links_u, links_u)]
        t_u = t0_lin[links_u].reshape(-1, 1) + K_uu @ q_u
        t_u = clean_matrix(t_u)
        
        lam_flows_full[links_u] = q_u.flatten()
        lam_times_full[links_u] = t_u.flatten()
    
    # Calcul des temps OD
    lam_flows_full = clean_matrix(lam_flows_full.reshape(-1, 1)).flatten()
    lam_times_full = clean_matrix(lam_times_full.reshape(-1, 1)).flatten()
    
    t_OD = compute_od_times_fast(lam_flows_full, lam_times_full, t0_lin, gamma, delta, network)
    
    print("\n=== TEMPS OD CALCULÉS ===")
    for i, (origin, dest) in enumerate(zip(network.on, network.dn)):
        if i < len(t_OD):
            print(f"OD {i+1}: {origin} -> {dest}: t_OD = {t_OD[i]:.4f}")
    
    return lam_flows_full, lam_times_full, t_OD


def compute_od_times_fast(lam_flows, lam_times, t0, gamma, delta, network):
    """Version optimisée du calcul des temps OD."""
    if issparse(delta):
        t_paths = delta.T @ lam_times.reshape(-1, 1)
    else:
        t_paths = delta.T @ lam_times.reshape(-1, 1)
    
    if issparse(gamma):
        gamma_pinv = robust_pinv(gamma.toarray())
    else:
        gamma_pinv = robust_pinv(gamma)
    
    t_OD = (gamma_pinv.T @ t_paths).flatten()
    return clean_matrix(t_OD)


# ========== OPTIMISATION 5: FONCTION PRINCIPALE AVEC PROFILAGE ==========

def compute_lam_solution_optimized(network, path_list, G, eps_num, method, alpha, beta, 
                                   use_sparse=True, verbose=True):
    """
    Version optimisée de compute_lam_solution avec:
    - Matrices creuses quand possible
    - Cache des inversions
    - RREF rapide
    - Profilage optionnel
    """
    import time
    
    cache = MatrixCache()
    timings = {}
    
    # 1. Construction des matrices (creuses si demandé)
    t0 = time.time()
    if use_sparse and sum(len(p) for p in path_list) > 1000:
        # Utiliser creuses seulement pour grands réseaux
        if verbose:
            print("→ Utilisation de matrices creuses (grand réseau)")
        gamma = build_gamma_matrix_sparse(path_list, network)
        delta = build_delta_matrix_sparse(path_list, network, G)
        # Convertir en dense pour les calculs (plus stable)
        gamma = gamma.toarray()
        delta = delta.toarray()
    else:
        if verbose:
            print("→ Utilisation de matrices denses")
        from modules.lam_solvers import build_gamma_matrix, build_delta_matrix
        gamma = build_gamma_matrix(path_list, network)
        delta = build_delta_matrix(path_list, network, G)
    
    T = build_T_matrix(path_list, network, G)
    timings['matrix_construction'] = time.time() - t0
    
    if verbose:
        print(f"✓ Construction matrices: {timings['matrix_construction']:.2f}s")
        print(f"  Gamma: {gamma.shape}")
        print(f"  Delta: {delta.shape}")
    
    # 2. Extraction indices
    t0 = time.time()
    from modules.lam_solvers import extract_multiple_indices, build_gamma_m_matrix, build_delta_m_matrix
    
    indices = extract_multiple_indices(path_list, delta)
    gamma_m = build_gamma_m_matrix(gamma, indices)
    delta_m = build_delta_m_matrix(delta, indices)
    timings['extraction'] = time.time() - t0
    
    if verbose:
        print(f"✓ Extraction indices: {timings['extraction']:.2f}s")
        print(f"  OD multiples: {indices['m_m']}, Chemins multiples: {indices['p_m']}")
    
    # 3. Réduction dimensionnelle (AVEC RREF RAPIDE)
    t0 = time.time()
    from modules.lam_solvers import build_delta_m_delta_mu_delta_um
    
    delta_m, delta_m_u, delta_u_m, final_indices = build_delta_m_delta_mu_delta_um(
        delta, delta_m, gamma_m, indices, network, path_list
    )
    
    # UTILISER RREF RAPIDE AU LIEU DE SYMPY
    if verbose:
        print(f"  → RREF rapide sur matrice {delta_m.shape}...")
    delta_m_tilde, rank_delta_m = rref_fast(delta_m)
    
    timings['reduction'] = time.time() - t0
    
    if verbose:
        print(f"✓ Réduction dimensionnelle: {timings['reduction']:.2f}s")
        print(f"  Rang delta_m: {rank_delta_m}")
    
    # 4. Construction matrices finales
    t0 = time.time()
    from modules.lam_solvers import (
        build_A_Rr_and_r0, build_T_m, extract_dimensions
    )
    
    A, Rr, r0, q_od_m = build_A_Rr_and_r0(delta_m, gamma_m, final_indices, network)
    
    # Fix pour build_B_q0 - calculer manuellement
    m = len(network.q_od)
    od_m = final_indices["od_m"]
    od_u_indices = sorted(list(set(range(m)) - set(od_m)))
    q_od_u = np.array(network.q_od)[od_u_indices].reshape(-1, 1)
    
    B = delta_m @ np.linalg.pinv(delta_m_tilde)
    q0 = delta_m_u @ q_od_u
    
    T_m = build_T_m(T, final_indices)
    dimensions = extract_dimensions(delta_m, delta_m_tilde, A, T_m)
    timings['final_matrices'] = time.time() - t0
    
    if verbose:
        print(f"✓ Matrices finales: {timings['final_matrices']:.2f}s")
        print(f"  Dimensions: n1={dimensions['n1']}, r1={dimensions['r1']}, s1={dimensions['s1']}, u1={dimensions['u1']}")
    
    # 5. Résolution (AVEC CACHE)
    t0 = time.time()
    if method == 'qp_analytical':
        lam_flows, lam_times, t_OD = lam_solver_qp_analytical_fast(
            network, final_indices, dimensions, alpha, beta, eps_num, 
            A, B, Rr, q0, r0, delta, delta_u_m, q_od_u, q_od_m,
            gamma, delta_m_u, gamma_m, cache=cache
        )
    else:
        raise ValueError(f"Méthode {method} non supportée dans la version optimisée. Utilisez 'qp_analytical'")
    
    timings['solve'] = time.time() - t0
    
    if verbose:
        print(f"✓ Résolution: {timings['solve']:.2f}s")
        print(f"\n{'='*60}")
        print(f"  TEMPS TOTAL: {sum(timings.values()):.2f}s")
        print(f"{'='*60}")
        print("\nDétail des temps:")
        for key, val in sorted(timings.items(), key=lambda x: -x[1]):
            pct = val/sum(timings.values())*100
            bar = '█' * int(pct/2)
            print(f"  {key:25s}: {val:6.2f}s  {pct:5.1f}% {bar}")
    
    return lam_flows, lam_times, t_OD


def build_T_matrix(path_list, network, G):
    """Construction de la matrice T (inchangée)."""
    from modules.lam_solvers import build_T_matrix as build_T_orig
    return build_T_orig(path_list, network, G)