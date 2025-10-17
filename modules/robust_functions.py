from scipy.linalg import pinv as scipy_pinv
from numpy.linalg import pinv as numpy_pinv
import numpy as np
import warnings
from sympy import Matrix

# ========== FONCTIONS ROBUSTES NUMÉRIQUES ==========

def robust_pinv(A, rcond=None):
    """
    Pseudo-inverse robuste et compatible SciPy / NumPy.
    
    - Utilise scipy.linalg.pinv si disponible (meilleure stabilité et tolérance contrôlable)
    - Fallback automatique sur numpy.linalg.pinv si scipy indisponible
    - Nettoyage des petites valeurs pour limiter les erreurs numériques
    """
    if rcond is None:
        rcond = RobustnessConfig.RCOND_PINV

    A_clean = clean_matrix(A)

    try:
        # SciPy est plus stable et accepte explicitement rcond
        A_pinv = scipy_pinv(A_clean, cond=rcond)
    except TypeError:
        # Fallback sur NumPy (certaines versions n’acceptent pas rcond)
        try:
            A_pinv = numpy_pinv(A_clean, rcond=rcond)
        except TypeError:
            A_pinv = numpy_pinv(A_clean)

    return clean_matrix(A_pinv)


def robust_null_space(A, rcond=None):
    """
    Calcul robuste du noyau (null space) — équivalent MATLAB: null(A', 'rational').
    
    - Basé sur SVD
    - Tolérance adaptative selon la première valeur singulière
    - Nettoyage final pour supprimer le bruit numérique
    """
    if rcond is None:
        rcond = RobustnessConfig.RCOND_SVD

    A_clean = clean_matrix(A)
    U, s, Vt = np.linalg.svd(A_clean.T, full_matrices=True)
    
    if len(s) == 0:
        return np.zeros((A.shape[1], 0))

    tol = rcond * s[0]
    rank = np.sum(s > tol)
    ns = Vt[rank:].T  # vecteurs à droite associés aux petites valeurs singulières

    ns_clean = clean_matrix(ns)
    return ns_clean


def robust_solve(A, b, method='auto'):
    """
    Résolution robuste du système linéaire Ax = b.
    Choisit automatiquement la meilleure méthode selon le conditionnement.
    """
    A_clean = clean_matrix(A)
    b_clean = clean_matrix(b)

    # Choix automatique de la méthode
    if method == 'auto':
        try:
            cond = np.linalg.cond(A_clean)
            if cond < 1e3:
                method = 'direct'
            elif cond < 1e6:
                method = 'lstsq'
            elif cond < 1e10:
                method = 'pinv'
            else:
                method = 'tikhonov'
                if RobustnessConfig.VERBOSE:
                    warnings.warn(f"[robust_solve] Matrice très mal conditionnée (cond={cond:.2e}), régularisation.")
        except Exception:
            method = 'lstsq'

    # Résolution selon la méthode
    try:
        if method == 'direct':
            x = np.linalg.solve(A_clean, b_clean)
        elif method == 'lstsq':
            x, *_ = np.linalg.lstsq(A_clean, b_clean, rcond=RobustnessConfig.RCOND_PINV)
        elif method == 'pinv':
            x = robust_pinv(A_clean) @ b_clean
        elif method == 'tikhonov':
            lambda_reg = 1e-8 * np.max(np.diag(A_clean.T @ A_clean))
            x = np.linalg.solve(
                A_clean.T @ A_clean + lambda_reg * np.eye(A_clean.shape[1]),
                A_clean.T @ b_clean
            )
        else:
            raise ValueError(f"Méthode inconnue: {method}")
    except np.linalg.LinAlgError:
        if RobustnessConfig.VERBOSE:
            warnings.warn(f"[robust_solve] Échec de {method}, fallback sur lstsq.")
        x, *_ = np.linalg.lstsq(A_clean, b_clean, rcond=RobustnessConfig.RCOND_PINV)

    return clean_matrix(x)



from modules import *
import modules.cost_functions as cf
import warnings

# ========== CONFIGURATION GLOBALE DE ROBUSTESSE ==========

class RobustnessConfig:
    """Configuration centralisée des paramètres de robustesse numérique."""
    RCOND_PINV = 1e-5          # Tolérance pseudo-inverse (vs 1e-15 défaut Python)
    RCOND_SVD = 1e-5           # Tolérance SVD
    THRESHOLD_CLEAN = 1e-5     # Seuil de nettoyage (comme Dr(abs(Dr)<1e-3)=0 MATLAB)
    THRESHOLD_RREF = 1e-5      # Tolérance pour RREF
    THRESHOLD_RANK = 1e-5      # Tolérance pour le calcul de rang
    MAX_CONDITION = 1e12        # Numéro de condition maximal acceptable
    
    # Paramètres de diagnostic
    VERBOSE = False
    DIAGNOSTICS = True


def clean_matrix(matrix, threshold=None):
    """
    Nettoie les petites valeurs d'une matrice (équivalent MATLAB: Dr(abs(Dr)<1e-3)=0).
    
    Args:
        matrix: Matrice à nettoyer
        threshold: Seuil de nettoyage (défaut: RobustnessConfig.THRESHOLD_CLEAN)
    
    Returns:
        Matrice nettoyée
    """
    if threshold is None:
        threshold = RobustnessConfig.THRESHOLD_CLEAN
    
    cleaned = matrix.copy()
    cleaned[np.abs(cleaned) < threshold] = 0
    return cleaned



def diagnose_matrix(A, name="Matrix"):
    """
    Diagnostique les problèmes numériques d'une matrice.
    
    Args:
        A: Matrice à diagnostiquer
        name: Nom pour l'affichage
    
    Returns:
        Dictionnaire avec les diagnostics
    """
    if not RobustnessConfig.DIAGNOSTICS:
        return {}
    
    diagnostics = {
        'shape': A.shape,
        'rank': np.linalg.matrix_rank(A, tol=RobustnessConfig.THRESHOLD_RANK),
        'condition': np.linalg.cond(A) if A.shape[0] == A.shape[1] else None
    }
    
    # Valeurs singulières
    U, s, Vt = np.linalg.svd(A)
    diagnostics['singular_values'] = s
    diagnostics['min_sv'] = s[-1] if len(s) > 0 else 0
    diagnostics['max_sv'] = s[0] if len(s) > 0 else 0
    diagnostics['num_small_sv'] = np.sum(s < RobustnessConfig.THRESHOLD_CLEAN)
    
    if RobustnessConfig.VERBOSE:
        print(f"\n=== Diagnostics: {name} ===")
        print(f"Shape: {diagnostics['shape']}")
        print(f"Rank: {diagnostics['rank']}")
        if diagnostics['condition'] is not None:
            print(f"Condition: {diagnostics['condition']:.2e}")
        print(f"Singular values: [{diagnostics['min_sv']:.2e}, {diagnostics['max_sv']:.2e}]")
        print(f"Small singular values (<{RobustnessConfig.THRESHOLD_CLEAN}): {diagnostics['num_small_sv']}")
        
        if diagnostics['condition'] and diagnostics['condition'] > RobustnessConfig.MAX_CONDITION:
            print(f"⚠️  WARNING: Matrice mal conditionnée!")
    
    return diagnostics
