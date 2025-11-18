from modules.network import load_network, load_tntp_flows
from modules.numerical_solvers import MSA_solver
from modules.lam_solvers import compute_lam_solution
from modules.plot_utils import (plot_network_colormap, plot_comparison, evaluate_solution, plot_prediction_graph_dual)
from modules.post_processing import compare_solutions, robustness_test
import numpy as np


def main():
    # ========== CONFIGURATION ==========
    config = {
        'network_name': "toy",
        'alpha': 0.15,
        'beta': 4,
        'N_iter': 1000,
        'tol': 1e-3,
        'lam_method': 'qp_analytical',
        'test_noise': True,
        'noise_levels': [0.1, 0.2, 0.3]
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
    
    lam_flows, lam_times = compute_lam_solution(
        network, path_list, G, eps_num,
        method=config['lam_method'], 
        alpha=config['alpha'], 
        beta=config['beta']
    )
    
    print(f"✓ Solution LAM calculée avec succès (méthode: {config['lam_method']})")
    
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

    plot_prediction_graph_dual(lam_flows, lam_times, msa_flows, msa_times, network, network_name=network.name,
        beta=config["beta"],
        suffix=config["network_name"]
    )

    
    # ========== TEST DE ROBUSTESSE ==========
    if config['test_noise']:
        robustness_test(
            network=network,
            path_list=path_list,
            G=G,
            eps_num=eps_num,
            config=config,
            metrics_original=metrics_original
        )
    
    # ========== ANALYSE COMPARATIVE DÉTAILLÉE ==========
    sol_num = {
        "path_list": path_list, 
        "flows": msa_flows, 
        "times": msa_times, 
        "G": G
    }
    sol_lam = {
        "flows": lam_flows, 
        "times": lam_times
    }
    
    compare_solutions(sol_num, sol_lam, network)


if __name__ == "__main__":
    main()