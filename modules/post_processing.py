"""
Module de post-traitement pour l'analyse et la comparaison des solutions.
"""
import numpy as np
from modules.numerical_solvers import MSA_solver
from modules.lam_solvers import compute_lam_solution
from modules.plot_utils import plot_comparison, evaluate_solution
import matplotlib.pyplot as plt


def _calculate_all_path_costs(path_list, link_times, G):
    """
    Calcule les coûts de tous les chemins basés sur les temps de liens donnés.
    Retourne une structure de coûts miroir de path_list.
    """
    all_costs = []
    for od_paths in path_list:
        od_costs = []
        for path in od_paths:
            cost = 0.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                try:
                    link_idx = G[u][v]['index']
                    cost += link_times[link_idx]
                except KeyError:
                    cost += np.inf
            od_costs.append(cost)
        all_costs.append(od_costs)
    return all_costs


def compare_solutions(sol_num, sol_lam, network, tolerance=1e-6):
    """
    Compare les solutions d'équilibre de MSA (numérique) et LAM (analytique).
    
    Args:
        sol_num (dict): Résultats de MSA. Doit contenir:
                        {'path_list': list, 'flows': np.array, 
                         'times': np.array, 'G': nx.DiGraph}
        sol_lam (dict): Résultats de LAM. Doit contenir:
                        {'flows': np.array, 'times': np.array}
        network: L'objet Network (pour les infos OD)
        tolerance (float): Tolérance pour comparer les coûts de chemins.
    """
    
    print("\n" + "="*80)
    print("COMPARAISON DES SOLUTIONS MSA (Numérique) vs LAM (Analytique)")
    print("="*80)
    
    # --- 1. Comparaison des Flux de Liens ---
    print("\n### 1. Comparaison des Flux de Liens ###")
    flows_msa = sol_num['flows']
    flows_lam = sol_lam['flows']
    
    flow_diff = flows_lam - flows_msa
    norm_diff = np.linalg.norm(flow_diff)
    norm_msa = np.linalg.norm(flows_msa)
    
    if norm_msa > 0:
        relative_diff = (norm_diff / norm_msa) * 100
        print(f"  Norme L2 du flux MSA (V):           {norm_msa:.4f}")
        print(f"  Norme L2 de la différence (LAM-MSA): {norm_diff:.4f}")
        print(f"  Différence relative:                 {relative_diff:.2f}%")
    else:
        print("  Flux MSA nuls, comparaison de flux de liens ignorée.")
        
    print("\n--- Liens avec les plus grandes différences (LAM - MSA) ---")
    top_5_diff_indices = np.argsort(np.abs(flow_diff))[-5:][::-1]
    for idx in top_5_diff_indices:
        print(f"  Lien {idx}: Différence = {flow_diff[idx]:+.4f} "
              f"(MSA={flows_msa[idx]:.2f}, LAM={flows_lam[idx]:.2f})")

    """
    # --- 2. Comparaison des Chemins Actifs par OD ---
    print("\n\n### 2. Comparaison des Chemins Actifs (par Coût) ###")
    
    path_list = sol_num['path_list']
    G = sol_num['G']
    
    costs_msa = _calculate_all_path_costs(path_list, sol_num['times'], G)
    costs_lam = _calculate_all_path_costs(path_list, sol_lam['times'], G)
    
    diff_count = 0
    
    for k in range(len(network.on)):
        origin = network.on[k]
        dest = network.dn[k]
        paths_k = path_list[k]
        
        if len(paths_k) <= 1:
            continue
            
        costs_k_msa = costs_msa[k]
        costs_k_lam = costs_lam[k]
        
        min_cost_msa = min(costs_k_msa)
        min_cost_lam = min(costs_k_lam)
        
        active_set_msa = {i for i, c in enumerate(costs_k_msa) 
                          if np.isclose(c, min_cost_msa, atol=tolerance)}
        
        active_set_lam = {i for i, c in enumerate(costs_k_lam) 
                          if np.isclose(c, min_cost_lam, atol=tolerance)}
        
        if active_set_msa != active_set_lam:
            diff_count += 1
            print(f"\n--- OD {k}: {origin} -> {dest} [DIFFÉRENCE D'ÉQUILIBRE TROUVÉE] ---")
            
            print("  Solution MSA (Coûts):")
            for i, (path, cost) in enumerate(zip(paths_k, costs_k_msa)):
                actif = "ACTIF" if i in active_set_msa else ""
                path_str = " -> ".join(map(str, path))
                print(f"    Chemin {i} [Cost: {cost:.6f}] {actif} \n      ({path_str})")
                
            print("\n  Solution LAM (Coûts):")
            for i, (path, cost) in enumerate(zip(paths_k, costs_k_lam)):
                actif = "ACTIF" if i in active_set_lam else ""
                path_str = " -> ".join(map(str, path))
                print(f"    Chemin {i} [Cost: {cost:.6f}] {actif} \n      ({path_str})")

    if diff_count == 0:
        print("\n  -> Tous les chemins actifs sont identiques entre les deux solutions.")
    else:
        print(f"\n  -> {diff_count} paires OD montrent des ensembles de chemins actifs différents.")
        
    print("="*80)
    """

def robustness_test(network, path_list, G, eps_num, config, metrics_original):
    """
    Teste la robustesse de la méthode LAM avec du bruit sur les capacités.
    
    Args:
        network: Objet Network
        path_list: Liste des chemins trouvés par MSA
        G: Graphe NetworkX
        eps_num: Niveau de congestion de référence
        config: Dictionnaire de configuration avec les paramètres
        metrics_original: Métriques de la solution sans bruit
    """
    print(f"\n{'='*60}")
    print(f"  TEST DE ROBUSTESSE LAM AVEC BRUIT SUR CAPACITÉS")
    print(f"{'='*60}\n")
    
    # Sauvegarde des capacités originales
    C_original = network.C.copy()
    results_noise = []
    
    for noise_level in config['noise_levels']:
        print(f"\n--- Test avec bruit de {noise_level*100:.0f}% ---")
        
        # Ajout de bruit aléatoire sur les capacités
        np.random.seed(42)
        noise = np.random.uniform(-noise_level, noise_level, size=len(C_original))
        network.C = C_original * (1 + noise)
        
        # Calcul MSA avec nouvelles capacités
        print(f"  Recalcul MSA avec capacités bruitées...")
        _, msa_flows_noise, msa_times_noise, _ = MSA_solver(
            network, config['N_iter'], config['tol'], 
            config['alpha'], config['beta'], 
            linearize_bpr=False, eps=None
        )
        
        # Prédiction LAM SANS recalculer path_list et eps
        print(f"  Prédiction LAM avec paramètres originaux...")
        lam_flows_pred, lam_times_pred, t_OD_pred = compute_lam_solution(
            network, path_list, G, eps_num,
            method=config['lam_method'], 
            alpha=config['alpha'], 
            beta=config['beta']
        )
        
        # Évaluation de la prédiction LAM
        metrics_noise = evaluate_solution(
            lam_flows_pred, lam_times_pred, 
            msa_flows_noise, msa_times_noise
        )
        
        results_noise.append({
            'noise_level': noise_level,
            'R2_flows': metrics_noise['flows']['R2'],
            'R2_times': metrics_noise['times']['R2'],
            'MAPE_flows': metrics_noise['flows']['MAPE'],
            'MAPE_times': metrics_noise['times']['MAPE'],
            'RMSE_flows': metrics_noise['flows']['RMSE'],
            'RMSE_times': metrics_noise['times']['RMSE']
        })
        
        print(f"    R² (flows) : {metrics_noise['flows']['R2']:.4f}")
        print(f"    R² (times) : {metrics_noise['times']['R2']:.4f}")
        print(f"    MAPE (flows) : {metrics_noise['flows']['MAPE']:.2f}%")
        print(f"    MAPE (times) : {metrics_noise['times']['MAPE']:.2f}%")

        suffix = f"noise_{int(noise_level*100)}pct"
        print(f"  → Génération du plot pour bruit {noise_level*100:.0f}%")
        
        plot_comparison(
            lam_flows_pred, lam_times_pred, 
            msa_flows_noise, msa_times_noise, 
            network, 
            config['network_name'],
            config['beta'],
            suffix=suffix
        )
    
    # Restauration des capacités originales
    network.C = C_original
    
    # Résumé des résultats
    _print_robustness_summary(results_noise, metrics_original)
    
    # Visualisation du dernier cas avec bruit
    print(f"\nVisualisation du cas avec {config['noise_levels'][-1]*100:.0f}% de bruit:")


def _print_robustness_summary(results_noise, metrics_original):
    """
    Affiche un résumé formaté des résultats du test de robustesse.
    """
    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ TEST DE ROBUSTESSE")
    print(f"{'='*60}")
    print(f"{'Bruit':>8} | {'R² flows':>10} | {'R² times':>10} | {'MAPE flows':>12} | {'MAPE times':>12}")
    print(f"{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")
    print(f"{'0%':>8} | {metrics_original['flows']['R2']:>10.4f} | "
          f"{metrics_original['times']['R2']:>10.4f} | "
          f"{metrics_original['flows']['MAPE']:>11.2f}% | "
          f"{metrics_original['times']['MAPE']:>11.2f}%")
    
    for res in results_noise:
        print(f"{res['noise_level']*100:>7.0f}% | {res['R2_flows']:>10.4f} | "
              f"{res['R2_times']:>10.4f} | {res['MAPE_flows']:>11.2f}% | "
              f"{res['MAPE_times']:>11.2f}%")
    print(f"{'='*60}\n")


# ========== BACKCASTING: Optimisation de Capacités ==========

def backcasting_capacity_optimization(network, path_list, G, t_OD_target, 
                                      eps_num, config, method='qp',
                                      alpha=0.15, beta=4.0, max_iter=50, 
                                      tolerance=1e-4):
    """
    Backcasting : Fixe des temps OD cibles et calcule les variations de capacité
    nécessaires sur les liens pour atteindre ces objectifs.
    
    Approche : Optimisation itérative pour trouver C_new tel que les temps OD
    calculés par LAM correspondent aux cibles.
    
    Args:
        network: Objet Network
        path_list: Liste des chemins
        G: Graphe NetworkX
        t_OD_target: Vecteur des temps OD cibles (m,) - Utiliser np.nan pour les OD sans cible
                     Exemple: np.array([40.0, np.nan, 25.5, 15.])  # Cible seulement pour OD 1,3,4
        eps_num: Niveau de congestion de référence
        config: Configuration (inutilisée mais fournie pour cohérence)
        method: Méthode LAM ('qp', 'linear_system', 'qp_analytical')
        alpha, beta: Paramètres BPR
        max_iter: Nombre d'itérations maximum
        tolerance: Tolérance de convergence sur les erreurs OD
    
    Returns:
        dict: {
            'C_original': Capacités originales,
            'C_optimized': Capacités optimisées,
            'capacity_changes': Variations absolues (C_new - C_orig),
            'capacity_changes_pct': Variations en pourcentage,
            'affected_links': Indices des liens modifiés,
            'targeted_od_indices': Indices des OD avec cibles,
            'convergence': {'iterations': int, 'final_error': float, 'converged': bool},
            'lam_flows': Flux LAM finaux,
            'lam_times': Temps LAM finaux,
            't_OD_achieved': Temps OD finaux atteints,
            't_OD_target': Temps OD cibles (avec np.nan pour les OD sans cible)
        }
    """
    from modules.cost_functions import linearised_bpr_matrices
    
    print("\n" + "="*80)
    print("BACKCASTING : OPTIMISATION DE CAPACITÉS POUR ATTEINDRE DES TEMPS OD CIBLES")
    print("="*80)
    
    n = len(network.sn)
    m = len(network.on)
    
    # Vérifier la cohérence de t_OD_target
    t_OD_target = np.asarray(t_OD_target, dtype=float).flatten()
    if len(t_OD_target) != m:
        raise ValueError(
            f"Erreur: t_OD_target a {len(t_OD_target)} valeurs mais "
            f"le réseau a {m} OD. Les tailles doivent correspondre."
        )
    
    # Identifier les OD avec cibles (non-NaN)
    targeted_mask = ~np.isnan(t_OD_target)
    targeted_od_indices = np.where(targeted_mask)[0]
    num_targeted = len(targeted_od_indices)
    
    if num_targeted == 0:
        raise ValueError("Aucune cible de temps OD spécifiée (tous les éléments sont NaN)")
    
    # Stockage des capacités originales
    C_original = network.C.copy()
    C_current = network.C.copy()
    
    print(f"\nObjectifs de temps OD cibles ({num_targeted}/{m} OD) :")
    for k in targeted_od_indices:
        origin = network.on[k]
        dest = network.dn[k]
        print(f"  OD {k+1}: {origin} -> {dest} : t_OD = {t_OD_target[k]:.4f}")
    
    print(f"\nOD sans cible : {m - num_targeted}")
    
    # Paramètres d'optimisation
    learning_rate = 5.0  # Taux d'apprentissage PLUS AGRESSIF pour les ajustements de capacité
    converged = False
    iteration = 0
    errors_history = []
    capacity_history = []
    
    print(f"\nDémarrage de l'optimisation itérative (max_iter={max_iter})...")
    print(f"{'Iter':<6} {'MAE(t_OD_ciblé)':<18} {'Max_Error_ciblé':<18} {'C_change(%)':<15}")
    print("-" * 70)
    
    while iteration < max_iter:
        # 1. Mise à jour des capacités
        network.C = C_current.copy()
        
        # 2. Calcul de la solution LAM avec capacités courantes
        try:
            lam_flows, lam_times, t_OD_achieved = compute_lam_solution(
                network, path_list, G, eps_num,
                method=method, alpha=alpha, beta=beta
            )
        except Exception as e:
            print(f"  ERREUR lors du calcul LAM à l'itération {iteration}: {e}")
            break
        
        # 3. Calcul de l'erreur SEULEMENT sur les OD ciblés
        t_OD_error = t_OD_achieved - t_OD_target
        t_OD_error_targeted = t_OD_error[targeted_mask]
        mae_od = np.mean(np.abs(t_OD_error_targeted))
        max_error = np.max(np.abs(t_OD_error_targeted))
        errors_history.append(mae_od)
        
        # 4. Déterminer les liens affectés pour chaque OD
        delta = build_delta_matrix(path_list, network, G)
        gamma = build_gamma_matrix(path_list, network)
        
        # 5. Calcul des ajustements de capacité basé sur les erreurs OD
        C_adjustment = np.zeros(n)
        error_magnitude = np.zeros(n)
        num_adjustments = 0
        
        for k_idx, k in enumerate(targeted_od_indices):
            # SEULEMENT ignorer la tolérance pour forcer l'ajustement
            if np.abs(t_OD_error[k]) > 0.001:
                num_adjustments += 1
                error_direction = np.sign(t_OD_error[k])
                error_mag = np.abs(t_OD_error[k]) / (np.abs(t_OD_target[k]) + 1e-6)
                
                if iteration < 3:
                    print(f"    DEBUG Iter {iteration}: OD {k} - Erreur {t_OD_error[k]:.4f}, mag: {error_mag:.4f}")
                
                # Parcourir les chemins de cette OD
                if path_list[k]:
                    for path in path_list[k]:
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i+1]
                            link_idx = G[u][v]['index']
                            
                            if error_direction > 0:
                                C_adjustment[link_idx] += error_mag
                            else:
                                C_adjustment[link_idx] -= error_mag
                            error_magnitude[link_idx] = max(error_magnitude[link_idx], error_mag)
        
        if iteration < 3:
            print(f"    DEBUG: {num_adjustments} OD avec erreur, liens à ajuster: {np.sum(C_adjustment != 0)}")
        
        # 6. Appliquer l'ajustement aux capacités EN CONSERVANT LA SOMME TOTALE
        C_prev = C_current.copy()
        C_current = C_current.astype(float)
        
        # Somme totale à conserver
        total_capacity_orig = np.sum(C_current)
        
        # Calculer les facteurs d'ajustement d'abord
        adjustment_factors = np.ones(n)
        for link_idx in range(n):
            if C_adjustment[link_idx] != 0 and error_magnitude[link_idx] > 0:
                try:
                    normalized_adjustment = np.clip(float(C_adjustment[link_idx]) / max(0.1, float(error_magnitude[link_idx])), -1.0, 1.0)
                    adjustment_factors[link_idx] = 1.0 + 0.5 * normalized_adjustment * 0.1  # Learning rate = 0.5
                    adjustment_factors[link_idx] = np.clip(adjustment_factors[link_idx], 0.5, 2.0)
                except (ValueError, OverflowError):
                    pass
        
        # Appliquer les facteurs
        C_temp = C_current * adjustment_factors
        C_temp = np.maximum(C_temp, 0.1)  # Minimum 0.1 par lien
        
        # NORMALISER pour conserver la somme totale des capacités
        total_capacity_new = np.sum(C_temp)
        if total_capacity_new > 0:
            normalization_factor = total_capacity_orig / total_capacity_new
            C_current = C_temp * normalization_factor
        else:
            C_current = C_prev.copy()  # Fallback
        
        # 7. Vérifier la convergence
        C_change_pct = np.mean(np.abs(C_current - C_prev) / (C_prev + 1e-6)) * 100
        capacity_history.append(C_change_pct)
        
        print(f"{iteration+1:<6} {mae_od:<15.6f} {max_error:<15.6f} {C_change_pct:<15.2f}")

        
        if max_error < tolerance:
            converged = True
            print(f"\n✓ Convergence atteinte à l'itération {iteration+1}")
            break
        
        if mae_od < tolerance and C_change_pct < 1.0:
            converged = True
            print(f"\n✓ Convergence atteinte (erreur stable)")
            break
        
        iteration += 1
    
    # Résultats finaux
    C_optimized = C_current.copy()
    capacity_changes = C_optimized - C_original
    capacity_changes_pct = (capacity_changes / (C_original + 1e-6)) * 100
    
    # Identifier les liens modifiés (variation > 1%)
    affected_links = np.where(np.abs(capacity_changes_pct) > 1.0)[0]
    
    print(f"\n{'='*80}")
    print("RÉSULTATS DE L'OPTIMISATION DE CAPACITÉS")
    print(f"{'='*80}")
    print(f"Itérations : {iteration}")
    print(f"Convergence : {'OUI' if converged else 'NON'}")
    print(f"Erreur finale : {max_error:.6f}")
    print(f"\nLiens affectés (variation > 1%) : {len(affected_links)} liens")
    
    if len(affected_links) > 0:
        print(f"\n{'Lien':<10} {'C_orig':<12} {'C_opt':<12} {'Variation':<15} {'Var %':<10}")
        print("-" * 70)
        for link_idx in affected_links[:20]:  # Afficher les 20 premiers
            orig = C_original[link_idx]
            opt = C_optimized[link_idx]
            print(f"{link_idx:<10} {orig:<12.4f} {opt:<12.4f} {capacity_changes[link_idx]:<15.4f} "
                  f"{capacity_changes_pct[link_idx]:<10.2f}%")
        if len(affected_links) > 20:
            print(f"... et {len(affected_links) - 20} autres liens")
    
    print(f"\nComparaison des temps OD (ciblés) :")
    print(f"{'OD':<6} {'Cible':<12} {'Atteint':<12} {'Erreur':<12} {'Erreur %':<10} {'Ciblé':<8}")
    print("-" * 70)
    for k in range(len(t_OD_target)):
        is_targeted = '✓' if targeted_mask[k] else '-'
        if targeted_mask[k]:
            error_abs = t_OD_achieved[k] - t_OD_target[k]
            error_pct = (error_abs / (t_OD_target[k] + 1e-6)) * 100 if t_OD_target[k] > 0 else 0
            print(f"{k+1:<6} {t_OD_target[k]:<12.4f} {t_OD_achieved[k]:<12.4f} "
                  f"{error_abs:<12.4f} {error_pct:<10.2f}% {is_targeted:<8}")
        else:
            print(f"{k+1:<6} {'N/A':<12} {t_OD_achieved[k]:<12.4f} "
                  f"{'N/A':<12} {'N/A':<10} {is_targeted:<8}")
    
    print(f"{'='*80}\n")
    
    # Restaurer les capacités originales (pour ne pas modifier l'objet network)

    network.C = C_original.copy()
    
    return {
        'C_original': C_original,
        'C_optimized': C_optimized,
        'capacity_changes': capacity_changes,
        'capacity_changes_pct': capacity_changes_pct,
        'affected_links': affected_links,
        'targeted_od_indices': targeted_od_indices,
        'convergence': {
            'iterations': iteration,
            'final_error': max_error,
            'converged': converged,
            'errors_history': errors_history,
            'capacity_history': capacity_history
        },
        'lam_flows': lam_flows,
        'lam_times': lam_times,
        't_OD_achieved': t_OD_achieved,
        't_OD_target': t_OD_target
    }


def build_gamma_matrix(path_list, network):
    """Construit la matrice Γ (m x p) reliant OD aux chemins."""
    m = len(network.on)
    p = sum(len(k) for k in path_list)

    gamma = np.zeros((m, p))
    col_start = np.cumsum([0] + [len(k) for k in path_list[:-1]])
    for i, start in enumerate(col_start):
        gamma[i, start:start+len(path_list[i])] = 1
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
    return delta

def robustness_test_t0(network, path_list, G, eps_num, config, metrics_original, 
                       noise_levels=None, n_tests=10, seed=42):
    """
    Teste la robustesse de LAM avec du bruit sur les temps à vide (t0).
    
    Args:
        network: Objet Network
        path_list: Liste des chemins trouvés par MSA
        G: Graphe NetworkX
        eps_num: Niveau de congestion de référence
        config: Dictionnaire de configuration
        metrics_original: Métriques de la solution sans bruit
        noise_levels: Liste des niveaux de bruit à tester (ex: [0.1, 0.2, 0.3])
        n_tests: Nombre de tests par niveau de bruit
        seed: Graine aléatoire pour reproductibilité
    
    Returns:
        dict: Résultats agrégés par niveau de bruit
    """
    if noise_levels is None:
        noise_levels = [0.1, 0.2, 0.3]
    
    print(f"\n{'='*70}")
    print(f"  TEST DE ROBUSTESSE LAM - VARIATION DES TEMPS À VIDE (t0)")
    print(f"  {n_tests} tests par niveau de bruit")
    print(f"{'='*70}\n")
    
    t0_original = network.t0.copy()
    results_by_noise = {}
    
    for noise_level in noise_levels:
        print(f"\n--- Niveau de bruit: {noise_level*100:.0f}% ({n_tests} tests) ---")
        
        test_results = []
        np.random.seed(seed)
        
        for test_idx in range(n_tests):
            # Bruit aléatoire sur t0
            noise = np.random.uniform(-noise_level, noise_level, size=len(t0_original))
            network.t0 = t0_original * (1 + noise)
            network.t0 = np.maximum(network.t0, 0.1)  # Éviter t0 négatifs
            
            # Recalcul MSA avec nouveaux t0
            try:
                _, msa_flows_noise, msa_times_noise, _ = MSA_solver(
                    network, config['N_iter'], config['tol'], 
                    config['alpha'], config['beta'], 
                    linearize_bpr=False, eps=None
                )
                
                # Prédiction LAM avec paramètres originaux (path_list, eps_num)
                lam_flows_pred, lam_times_pred, _ = compute_lam_solution(
                    network, path_list, G, eps_num,
                    method=config['lam_method'], 
                    alpha=config['alpha'], 
                    beta=config['beta']
                )
                
                metrics = evaluate_solution(
                    lam_flows_pred, lam_times_pred, 
                    msa_flows_noise, msa_times_noise
                )
                
                test_results.append({
                    'R2_flows': metrics['flows']['R2'],
                    'R2_times': metrics['times']['R2'],
                    'MAPE_flows': metrics['flows']['MAPE'],
                    'MAPE_times': metrics['times']['MAPE'],
                    'RMSE_flows': metrics['flows']['RMSE'],
                    'RMSE_times': metrics['times']['RMSE']
                })
            except Exception as e:
                print(f"    Test {test_idx+1} échoué: {e}")
                continue
        
        # Statistiques agrégées
        if test_results:
            results_by_noise[noise_level] = _aggregate_results(test_results)
            _print_aggregated_results(noise_level, results_by_noise[noise_level])
    
    network.t0 = t0_original
    return results_by_noise


def robustness_test_od_demand(network, path_list, G, eps_num, config, metrics_original,
                              noise_levels=None, n_tests=10, seed=42):
    """
    Teste la robustesse de LAM avec du bruit sur les demandes OD (q_od).
    
    Args:
        network: Objet Network
        path_list: Liste des chemins trouvés par MSA
        G: Graphe NetworkX
        eps_num: Niveau de congestion de référence
        config: Dictionnaire de configuration
        metrics_original: Métriques de la solution sans bruit
        noise_levels: Liste des niveaux de bruit à tester
        n_tests: Nombre de tests par niveau de bruit
        seed: Graine aléatoire
    
    Returns:
        dict: Résultats agrégés par niveau de bruit
    """
    if noise_levels is None:
        noise_levels = [0.1, 0.2, 0.3]
    
    print(f"\n{'='*70}")
    print(f"  TEST DE ROBUSTESSE LAM - VARIATION DES DEMANDES OD (q_od)")
    print(f"  {n_tests} tests par niveau de bruit")
    print(f"{'='*70}\n")
    
    q_od_original = np.array(network.q_od).copy()
    results_by_noise = {}
    
    for noise_level in noise_levels:
        print(f"\n--- Niveau de bruit: {noise_level*100:.0f}% ({n_tests} tests) ---")
        
        test_results = []
        np.random.seed(seed)
        
        for test_idx in range(n_tests):
            # Bruit aléatoire sur les demandes OD
            noise = np.random.uniform(-noise_level, noise_level, size=len(q_od_original))
            network.q_od = list(q_od_original * (1 + noise))
            network.q_od = [max(0.1, q) for q in network.q_od]  # Éviter demandes négatives
            
            try:
                _, msa_flows_noise, msa_times_noise, _ = MSA_solver(
                    network, config['N_iter'], config['tol'], 
                    config['alpha'], config['beta'], 
                    linearize_bpr=False, eps=None
                )
                
                lam_flows_pred, lam_times_pred, _ = compute_lam_solution(
                    network, path_list, G, eps_num,
                    method=config['lam_method'], 
                    alpha=config['alpha'], 
                    beta=config['beta']
                )
                
                metrics = evaluate_solution(
                    lam_flows_pred, lam_times_pred, 
                    msa_flows_noise, msa_times_noise
                )
                
                test_results.append({
                    'R2_flows': metrics['flows']['R2'],
                    'R2_times': metrics['times']['R2'],
                    'MAPE_flows': metrics['flows']['MAPE'],
                    'MAPE_times': metrics['times']['MAPE'],
                    'RMSE_flows': metrics['flows']['RMSE'],
                    'RMSE_times': metrics['times']['RMSE']
                })
            except Exception as e:
                print(f"    Test {test_idx+1} échoué: {e}")
                continue
        
        if test_results:
            results_by_noise[noise_level] = _aggregate_results(test_results)
            _print_aggregated_results(noise_level, results_by_noise[noise_level])
    
    network.q_od = list(q_od_original)
    return results_by_noise


def robustness_test_capacity(network, path_list, G, eps_num, config, metrics_original,
                             noise_levels=None, n_tests=10, seed=42):
    """
    Teste la robustesse de LAM avec du bruit sur les capacités (C).
    Version améliorée avec n_tests par niveau.
    """
    if noise_levels is None:
        noise_levels = [0.1, 0.2, 0.3]
    
    print(f"\n{'='*70}")
    print(f"  TEST DE ROBUSTESSE LAM - VARIATION DES CAPACITÉS (C)")
    print(f"  {n_tests} tests par niveau de bruit")
    print(f"{'='*70}\n")
    
    C_original = network.C.copy()
    results_by_noise = {}
    
    for noise_level in noise_levels:
        print(f"\n--- Niveau de bruit: {noise_level*100:.0f}% ({n_tests} tests) ---")
        
        test_results = []
        np.random.seed(seed)
        
        for test_idx in range(n_tests):
            noise = np.random.uniform(-noise_level, noise_level, size=len(C_original))
            network.C = C_original * (1 + noise)
            network.C = np.maximum(network.C, 0.1)
            
            try:
                _, msa_flows_noise, msa_times_noise, _ = MSA_solver(
                    network, config['N_iter'], config['tol'], 
                    config['alpha'], config['beta'], 
                    linearize_bpr=False, eps=None
                )
                
                lam_flows_pred, lam_times_pred, _ = compute_lam_solution(
                    network, path_list, G, eps_num,
                    method=config['lam_method'], 
                    alpha=config['alpha'], 
                    beta=config['beta']
                )
                
                metrics = evaluate_solution(
                    lam_flows_pred, lam_times_pred, 
                    msa_flows_noise, msa_times_noise
                )
                
                test_results.append({
                    'R2_flows': metrics['flows']['R2'],
                    'R2_times': metrics['times']['R2'],
                    'MAPE_flows': metrics['flows']['MAPE'],
                    'MAPE_times': metrics['times']['MAPE'],
                    'RMSE_flows': metrics['flows']['RMSE'],
                    'RMSE_times': metrics['times']['RMSE']
                })
            except Exception as e:
                print(f"    Test {test_idx+1} échoué: {e}")
                continue
        
        if test_results:
            results_by_noise[noise_level] = _aggregate_results(test_results)
            _print_aggregated_results(noise_level, results_by_noise[noise_level])
    
    network.C = C_original
    return results_by_noise


def _aggregate_results(test_results):
    """Calcule les statistiques agrégées (moyenne, std, min, max) des résultats."""
    metrics_keys = ['R2_flows', 'R2_times', 'MAPE_flows', 'MAPE_times', 'RMSE_flows', 'RMSE_times']
    aggregated = {}
    
    for key in metrics_keys:
        values = [r[key] for r in test_results if key in r]
        if values:
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'n_tests': len(values)
            }
    return aggregated


def _print_aggregated_results(noise_level, results):
    """Affiche les résultats agrégés pour un niveau de bruit."""
    n = results.get('R2_flows', {}).get('n_tests', 0)
    print(f"    Résultats sur {n} tests:")
    print(f"    {'Métrique':<15} {'Moyenne':>12} {'Écart-type':>12} {'Min':>10} {'Max':>10}")
    print(f"    {'-'*60}")
    
    for key in ['R2_flows', 'R2_times', 'MAPE_flows', 'MAPE_times']:
        if key in results:
            r = results[key]
            unit = '%' if 'MAPE' in key else ''
            print(f"    {key:<15} {r['mean']:>11.4f}{unit} {r['std']:>11.4f} "
                  f"{r['min']:>9.4f} {r['max']:>9.4f}")


def comprehensive_robustness_analysis(network, path_list, G, eps_num, config, 
                                      metrics_original, noise_levels=None, 
                                      n_tests=10, seed=42):
    """
    Analyse complète de robustesse sur les 3 variables (C, t0, q_od).
    Détermine les seuils de robustesse pour chaque variable.
    
    Args:
        network, path_list, G, eps_num, config, metrics_original: paramètres standards
        noise_levels: Liste des niveaux de bruit (défaut: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
        n_tests: Nombre de tests par niveau
        seed: Graine aléatoire
    
    Returns:
        dict: Résultats complets avec seuils de robustesse identifiés
    """
    if noise_levels is None:
        noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    print("\n" + "="*80)
    print("  ANALYSE COMPLÈTE DE ROBUSTESSE LAM")
    print(f"  Variables testées: Capacités (C), Temps à vide (t0), Demandes OD (q_od)")
    print(f"  Niveaux de bruit: {[f'{n*100:.0f}%' for n in noise_levels]}")
    print(f"  Tests par niveau: {n_tests}")
    print("="*80)
    
    all_results = {}
    
    # 1. Tests sur les capacités
    print("\n" + "-"*40)
    print("  PHASE 1/3: Tests sur les CAPACITÉS")
    print("-"*40)
    all_results['capacity'] = robustness_test_capacity(
        network, path_list, G, eps_num, config, metrics_original,
        noise_levels=noise_levels, n_tests=n_tests, seed=seed
    )
    
    # 2. Tests sur t0
    print("\n" + "-"*40)
    print("  PHASE 2/3: Tests sur les TEMPS À VIDE")
    print("-"*40)
    all_results['t0'] = robustness_test_t0(
        network, path_list, G, eps_num, config, metrics_original,
        noise_levels=noise_levels, n_tests=n_tests, seed=seed
    )
    
    # 3. Tests sur q_od
    print("\n" + "-"*40)
    print("  PHASE 3/3: Tests sur les DEMANDES OD")
    print("-"*40)
    all_results['q_od'] = robustness_test_od_demand(
        network, path_list, G, eps_num, config, metrics_original,
        noise_levels=noise_levels, n_tests=n_tests, seed=seed
    )
    
    # 4. Analyse des seuils de robustesse
    thresholds = analyze_robustness_thresholds(all_results, metrics_original)
    all_results['thresholds'] = thresholds
    
    # 5. Résumé final
    _print_comprehensive_summary(all_results, metrics_original, noise_levels)
    
    # 6. Visualisation
    plot_robustness_comparison(all_results, noise_levels, network.name)
    
    return all_results


def analyze_robustness_thresholds(all_results, metrics_original, 
                                  r2_threshold=0.90, mape_threshold_pct=50):
    """
    Analyse les seuils de robustesse pour chaque variable.
    
    Définition du seuil: niveau de bruit maximal où R² > r2_threshold 
    ET MAPE < MAPE_original * (1 + mape_threshold_pct/100)
    
    Args:
        all_results: Résultats des tests
        metrics_original: Métriques de référence sans bruit
        r2_threshold: Seuil minimum de R² acceptable
        mape_threshold_pct: Augmentation maximale acceptable du MAPE (en %)
    
    Returns:
        dict: Seuils de robustesse par variable
    """
    thresholds = {}
    
    mape_flows_orig = metrics_original['flows']['MAPE']
    mape_times_orig = metrics_original['times']['MAPE']
    mape_flows_max = mape_flows_orig * (1 + mape_threshold_pct / 100)
    mape_times_max = mape_times_orig * (1 + mape_threshold_pct / 100)
    
    for var_name, var_results in all_results.items():
        if var_name == 'thresholds':
            continue
            
        max_robust_level = 0.0
        
        for noise_level in sorted(var_results.keys()):
            res = var_results[noise_level]
            
            r2_flows_mean = res.get('R2_flows', {}).get('mean', 0)
            r2_times_mean = res.get('R2_times', {}).get('mean', 0)
            mape_flows_mean = res.get('MAPE_flows', {}).get('mean', float('inf'))
            mape_times_mean = res.get('MAPE_times', {}).get('mean', float('inf'))
            
            is_robust = (
                r2_flows_mean >= r2_threshold and
                r2_times_mean >= r2_threshold and
                mape_flows_mean <= mape_flows_max and
                mape_times_mean <= mape_times_max
            )
            
            if is_robust:
                max_robust_level = noise_level
        
        thresholds[var_name] = {
            'max_robust_noise_level': max_robust_level,
            'r2_threshold_used': r2_threshold,
            'mape_threshold_used': mape_threshold_pct
        }
    
    return thresholds


def _print_comprehensive_summary(all_results, metrics_original, noise_levels):
    """Affiche un résumé complet de l'analyse de robustesse."""
    print("\n" + "="*90)
    print("  RÉSUMÉ COMPLET DE L'ANALYSE DE ROBUSTESSE")
    print("="*90)
    
    # Tableau comparatif
    print(f"\n{'Variable':<12} | ", end="")
    for nl in noise_levels:
        print(f"  {nl*100:>4.0f}%   |", end="")
    print(f" {'Seuil Rob.':<12}")
    
    print("-" * (14 + 11 * len(noise_levels) + 14))
    
    for var_name in ['capacity', 't0', 'q_od']:
        if var_name not in all_results:
            continue
        
        var_results = all_results[var_name]
        threshold = all_results.get('thresholds', {}).get(var_name, {}).get('max_robust_noise_level', 'N/A')
        
        # Ligne R² flows
        print(f"{var_name:<12} | ", end="")
        for nl in noise_levels:
            if nl in var_results:
                r2 = var_results[nl].get('R2_flows', {}).get('mean', 0)
                print(f"  {r2:>5.3f}  |", end="")
            else:
                print(f"   N/A   |", end="")
        print(f" {threshold*100 if isinstance(threshold, float) else threshold:>6}%")
    
    # Seuils de robustesse
    print("\n" + "-"*50)
    print("SEUILS DE ROBUSTESSE IDENTIFIÉS:")
    print("-"*50)
    
    thresholds = all_results.get('thresholds', {})
    for var_name, th_info in thresholds.items():
        level = th_info.get('max_robust_noise_level', 0)
        print(f"  {var_name:<12}: {level*100:>5.1f}% de bruit max toléré")
    
    print("\n  Critères utilisés:")
    if thresholds:
        first_th = list(thresholds.values())[0]
        print(f"    - R² minimum: {first_th.get('r2_threshold_used', 0.90)}")
        print(f"    - Augmentation MAPE max: {first_th.get('mape_threshold_used', 50)}%")
    
    print("="*90 + "\n")


def plot_robustness_comparison(all_results, noise_levels, network_name=""):
    """
    Génère des graphiques comparatifs de robustesse.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'capacity': '#2ecc71', 't0': '#3498db', 'q_od': '#e74c3c'}
    labels = {'capacity': 'Capacités (C)', 't0': 'Temps à vide (t₀)', 'q_od': 'Demandes OD'}
    
    metrics_to_plot = [
        ('R2_flows', 'R² Flux', axes[0, 0]),
        ('R2_times', 'R² Temps', axes[0, 1]),
        ('MAPE_flows', 'MAPE Flux (%)', axes[1, 0]),
        ('MAPE_times', 'MAPE Temps (%)', axes[1, 1])
    ]
    
    noise_pct = [n * 100 for n in noise_levels]
    
    for metric_key, metric_label, ax in metrics_to_plot:
        for var_name in ['capacity', 't0', 'q_od']:
            if var_name not in all_results:
                continue
            
            var_results = all_results[var_name]
            means = []
            stds = []
            
            for nl in noise_levels:
                if nl in var_results and metric_key in var_results[nl]:
                    means.append(var_results[nl][metric_key]['mean'])
                    stds.append(var_results[nl][metric_key]['std'])
                else:
                    means.append(np.nan)
                    stds.append(0)
            
            means = np.array(means)
            stds = np.array(stds)
            
            ax.plot(noise_pct, means, 'o-', color=colors[var_name], 
                    label=labels[var_name], linewidth=2, markersize=6)
            ax.fill_between(noise_pct, means - stds, means + stds, 
                           color=colors[var_name], alpha=0.2)
        
        ax.set_xlabel('Niveau de bruit (%)', fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if 'R2' in metric_key:
            ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Seuil R²=0.9')
            ax.set_ylim([0, 1.05])
    
    plt.suptitle(f'Analyse de Robustesse LAM - Réseau: {network_name}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'output/robustness_analysis_{network_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Graphique sauvegardé: outputs/robustness_analysis_{network_name}.png")


# ========== FONCTION UTILITAIRE POUR INTÉGRATION DANS MAIN.PY ==========

def run_full_robustness_suite(network, path_list, G, eps_num, msa_flows, msa_times,
                              lam_flows, lam_times, config):
    """
    Fonction wrapper pour exécuter la suite complète de tests de robustesse.
    À appeler depuis main.py.
    
    Args:
        network, path_list, G, eps_num: Paramètres du réseau
        msa_flows, msa_times: Solutions MSA de référence
        lam_flows, lam_times: Solutions LAM de référence
        config: Configuration avec noise_levels optionnel
    
    Returns:
        dict: Tous les résultats de robustesse
    """
    # Calcul des métriques originales
    metrics_original = evaluate_solution(lam_flows, lam_times, msa_flows, msa_times)
    
    # Niveaux de bruit à tester
    noise_levels = config.get('noise_levels', [0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    n_tests = config.get('n_robustness_tests', 10)
    
    # Exécution de l'analyse complète
    results = comprehensive_robustness_analysis(
        network=network,
        path_list=path_list,
        G=G,
        eps_num=eps_num,
        config=config,
        metrics_original=metrics_original,
        noise_levels=noise_levels,
        n_tests=n_tests,
        seed=42
    )
    
    return results