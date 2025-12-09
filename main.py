from modules.network import load_network, load_tntp_flows
from modules.numerical_solvers import MSA_solver
from modules.lam_solvers import print_od_times_comparison
from modules.plot_utils import (plot_network_colormap, plot_comparison, evaluate_solution, plot_prediction_graph_dual)
from modules.post_processing import compare_solutions, robustness_test, backcasting_capacity_optimization
from modules.post_processing import (
    robustness_test_t0,
    robustness_test_od_demand,
    robustness_test_capacity,
    comprehensive_robustness_analysis,
    run_full_robustness_suite
)
from modules.lam_solvers_optimized import compute_lam_solution_optimized


def main():
    # ========== CONFIGURATION ==========
    config = {
        'network_name': "sioux_falls",
        'alpha': 0.15,
        'beta': 4,
        'N_iter': 1000,
        'tol': 1e-3,
        'lam_method': 'qp_analytical',
        # Configuration des tests de robustesse
        'test_noise': False,              # Test original (capacités uniquement)
        'test_full_robustness': False,     # NOUVEAU: Suite complète de robustesse
        'noise_levels': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],  # Niveaux de bruit
        'n_robustness_tests': 10,
        'backcasting': False
    }
    
    # ========== CHARGEMENT DU RÉSEAU ==========
    print(f"\n{'='*60}")
    print(f"  CHARGEMENT DU RÉSEAU: {config['network_name'].upper()}")
    print(f"{'='*60}\n")
    
    network = load_network(config['network_name'])
    network.summary()
    network.plot()
    
    # ========== SOLUTION NUMÉRIQUE (MSA) ==========
    print(f"\n{'='*60}")
    print(f"  CALCUL DE LA SOLUTION NUMÉRIQUE (MSA)")
    print(f"{'='*60}\n")
    
    path_list, msa_flows, msa_times, G = MSA_solver(
        network, 
        config['N_iter'], 
        config['tol'], 
        config['alpha'], 
        config['beta'],
        linearize_bpr=False, 
        eps=None
    )
    
    # Charger les flux de référence si disponibles
    try:
        network.flow_ref = load_tntp_flows("data/sioux_falls/SiouxFalls_flow.tntp")
    except:
        network.flow_ref = None
    
    # Niveau de congestion moyen
    eps_num = msa_flows / network.C
    print(f"\nCongestion moyenne du réseau (flow/capacity): {eps_num.mean():.2f}")
    
    # Visualisation MSA
    plot_network_colormap(
        network, msa_flows, 
        value_type="flow", 
        cmap="plasma", 
        show_od_annotations=True,
        network_name=config['network_name'],
        beta=config['beta'],
        suffix="msa"
    )
    
    plot_network_colormap(
        network, msa_times, 
        value_type="travel time", 
        cmap="plasma", 
        show_od_annotations=True,
        network_name=config['network_name'],
        beta=config['beta'],
        suffix="msa"
    )
    
    # ========== SOLUTION ANALYTIQUE (LAM) ==========
    print(f"\n{'='*60}")
    print(f"  CALCUL DE LA SOLUTION ANALYTIQUE (LAM)")
    print(f"{'='*60}\n")
    
    lam_flows, lam_times, t_OD = compute_lam_solution_optimized(
        network, path_list, G, eps_num,
        method='qp_analytical',  # Seule méthode supportée dans la version optimisée
        alpha=config['alpha'], 
        beta=config['beta'],
        use_sparse=True,   # Utilise matrices creuses (automatique pour grands réseaux)
        verbose=True       # Affiche les temps de calcul détaillés
    )
    
    print(f"✓ Solution LAM calculée avec succès (méthode: {config['lam_method']})")
    
    # ========== AFFICHAGE DES TEMPS OD ==========
    print_od_times_comparison(t_OD, path_list, network, G, msa_times)
    
    # ========== COMPARAISON ET ÉVALUATION ==========
    print(f"\n{'='*60}")
    print(f"  COMPARAISON DES SOLUTIONS")
    print(f"{'='*60}\n")
    
    metrics_original = evaluate_solution(lam_flows, lam_times, msa_flows, msa_times)
    
    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ")
    print(f"{'='*60}")
    print(f"  Réseau        : {config['network_name']}")
    print(f"  Méthode LAM   : {config['lam_method']}")
    print(f"  R² (flows)    : {metrics_original['flows']['R2']:.4f}")
    print(f"  R² (times)    : {metrics_original['times']['R2']:.4f}")
    print(f"  MAPE (flows)  : {metrics_original['flows']['MAPE']:.2f}%")
    print(f"  MAPE (times)  : {metrics_original['times']['MAPE']:.2f}%")
    print(f"{'='*60}\n")
    
    plot_comparison(
        lam_flows, lam_times,
        msa_flows, msa_times,
        network_name=network.name,
        beta=config["beta"],
        suffix=config["network_name"]
    )

    plot_prediction_graph_dual(lam_flows, lam_times, msa_flows, msa_times, network, 
        network_name=network.name,
        beta=config["beta"],
        suffix=config["network_name"]
    )

    # ========== TEST DE ROBUSTESSE ORIGINAL (CAPACITÉS) ==========
    if config['test_noise']:
        robustness_test(
            network=network,
            path_list=path_list,
            G=G,
            eps_num=eps_num,
            config=config,
            metrics_original=metrics_original
        )
    
    # ========== NOUVEAU: SUITE COMPLÈTE DE TESTS DE ROBUSTESSE ==========
    if config.get('test_full_robustness', False):
        print(f"\n{'='*60}")
        print(f"  LANCEMENT DE LA SUITE COMPLÈTE DE ROBUSTESSE")
        print(f"{'='*60}\n")
        
        robustness_results = run_full_robustness_suite(
            network=network,
            path_list=path_list,
            G=G,
            eps_num=eps_num,
            msa_flows=msa_flows,
            msa_times=msa_times,
            lam_flows=lam_flows,
            lam_times=lam_times,
            config=config
        )
        
        # Affichage des seuils de robustesse identifiés
        print(f"\n{'='*60}")
        print(f"  SEUILS DE ROBUSTESSE IDENTIFIÉS")
        print(f"{'='*60}")
        thresholds = robustness_results.get('thresholds', {})
        for var_name, th_info in thresholds.items():
            level = th_info.get('max_robust_noise_level', 0)
            print(f"  {var_name:<15}: jusqu'à {level*100:>5.1f}% de bruit")
        print(f"{'='*60}\n")
    
    # ========== ANALYSE COMPARATIVE DÉTAILLÉE ==========
    sol_num = {
        "path_list": path_list, 
        "flows": msa_flows, 
        "times": msa_times, 
        "G": G
    }
    sol_lam = {
        "flows": lam_flows, 
        "times": lam_times,
        "t_OD": t_OD
    }
    
    compare_solutions(sol_num, sol_lam, network)

    # ========== BACKCASTING ==========
    if config['backcasting']:
        result = backcasting_capacity_optimization(
            network, path_list, G, 
            t_OD_target=np.array([35, 35, np.nan, np.nan, np.nan, np.nan, np.nan]),
            eps_num=0.15,
            config=config,
            method='qp_analytical',
            max_iter=500,
            tolerance=1e-4)

        print(result)


if __name__ == "__main__":
    main()