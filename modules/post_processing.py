"""
Module de post-traitement pour l'analyse et la comparaison des solutions.
"""
import numpy as np
from modules.numerical_solvers import MSA_solver
from modules.lam_solvers import compute_lam_solution
from modules.plot_utils import plot_comparison, evaluate_solution


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
        lam_flows_pred, lam_times_pred = compute_lam_solution(
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