from modules.network import load_network
from modules.numerical_solvers import MSA_solver
from modules.lam_solvers import compute_lam_solution
from modules.plot_utils import (plot_network_colormap, plot_comparison, evaluate_solution)
import numpy as np

# ========== CONFIGURATION ==========
network_name = "sioux_falls"  # toy ou sioux_falls

# Paramètres BPR
alpha = 0.15
beta = 1

# Paramètres MSA
N_iter = 1000
tole = 1e-3

# Méthode LAM
lam_method = 'linear_system'  # 'qp' ou 'linear_system'

# Paramètres de bruit
test_noise = False  # Activer/désactiver le test de robustesse
noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]  # Niveaux de bruit à tester 

# ========== CHARGEMENT ET VISUALISATION DU RÉSEAU ==========
print(f"\n{'='*60}")
print(f"  CHARGEMENT DU RÉSEAU: {network_name.upper()}")
print(f"{'='*60}\n")

network = load_network(network_name)
network.summary()
network.plot()

# Sauvegarde des capacités originales
C_original = network.C.copy()

# ========== SOLUTION NUMÉRIQUE (MSA) - RÉSEAU ORIGINAL ==========
print(f"\n{'='*60}")
print(f"  CALCUL DE LA SOLUTION NUMÉRIQUE (MSA) - RÉSEAU ORIGINAL")
print(f"{'='*60}\n")

path_list, msa_flows, msa_times, G = MSA_solver(
    network, N_iter, tole, alpha, beta, 
    linearize_bpr=False, eps=None
)

# Niveau de congestion moyen
eps_num = msa_flows / network.C
print(f"\nCongestion moyenne du réseau (flow/capacity): {eps_num.mean():.2f}")

# Visualisation
plot_network_colormap(
    network, msa_flows, 
    value_type="flow", 
    cmap="plasma", 
    show_od_annotations=True
)

plot_network_colormap(
    network, msa_times, 
    value_type="travel time", 
    cmap="plasma", 
    show_od_annotations=True
)

# ========== SOLUTION ANALYTIQUE (LAM) - RÉSEAU ORIGINAL ==========
print(f"\n{'='*60}")
print(f"  CALCUL DE LA SOLUTION ANALYTIQUE (LAM) - RÉSEAU ORIGINAL")
print(f"{'='*60}\n")

lam_flows, lam_times = compute_lam_solution(
    network, path_list, G, eps_num,
    method=lam_method, alpha=alpha, beta=beta
)

print(f"✓ Solution LAM calculée avec succès (méthode: {lam_method})")

# ========== COMPARAISON ET ÉVALUATION - RÉSEAU ORIGINAL ==========
print(f"\n{'='*60}")
print(f"  COMPARAISON DES SOLUTIONS - RÉSEAU ORIGINAL")
print(f"{'='*60}\n")

metrics_original = evaluate_solution(lam_flows, lam_times, msa_flows, msa_times)

print(f"\n{'='*60}")
print(f"  RÉSUMÉ - RÉSEAU ORIGINAL")
print(f"{'='*60}")
print(f"  Réseau        : {network_name}")
print(f"  Méthode LAM   : {lam_method}")
print(f"  R² (flows)    : {metrics_original['flows']['R2']:.4f}")
print(f"  R² (times)    : {metrics_original['times']['R2']:.4f}")
print(f"  MAPE (flows)  : {metrics_original['flows']['MAPE']:.2f}%")
print(f"  MAPE (times)  : {metrics_original['times']['MAPE']:.2f}%")
print(f"{'='*60}\n")

plot_comparison(lam_flows, lam_times, msa_flows, msa_times, network, network_name)


"""

# ========== TEST DE ROBUSTESSE AVEC BRUIT ==========
if test_noise:
    print(f"\n{'='*60}")
    print(f"  TEST DE ROBUSTESSE LAM AVEC BRUIT SUR CAPACITÉS")
    print(f"{'='*60}\n")
    
    results_noise = []
    
    for noise_level in noise_levels:
        print(f"\n--- Test avec bruit de {noise_level*100:.0f}% ---")
        
        # Ajout de bruit aléatoire sur les capacités
        np.random.seed(42)  # Pour reproductibilité
        noise = np.random.uniform(-noise_level, noise_level, size=len(C_original))
        network.C = C_original * (1 + noise)
        
        # Calcul MSA avec nouvelles capacités
        print(f"  Recalcul MSA avec capacités bruitées...")
        _, msa_flows_noise, msa_times_noise, _ = MSA_solver(
            network, N_iter, tol, alpha, beta, 
            linearize_bpr=False, eps=None
        )
        
        # Prédiction LAM SANS recalculer path_list et eps
        # On utilise les path_list et eps_num du réseau original
        print(f"  Prédiction LAM avec paramètres originaux...")
        lam_flows_pred, lam_times_pred = compute_lam_solution(
            network, path_list, G, eps_num,  # On garde eps_num original
            method=lam_method, alpha=alpha, beta=beta
        )
        
        # Évaluation de la prédiction LAM
        metrics_noise = evaluate_solution(lam_flows_pred, lam_times_pred, 
                                         msa_flows_noise, msa_times_noise)
        
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
    
    # Restauration des capacités originales
    network.C = C_original
    
    # Résumé des résultats
    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ TEST DE ROBUSTESSE")
    print(f"{'='*60}")
    print(f"{'Bruit':>8} | {'R² flows':>10} | {'R² times':>10} | {'MAPE flows':>12} | {'MAPE times':>12}")
    print(f"{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")
    print(f"{'0%':>8} | {metrics_original['flows']['R2']:>10.4f} | {metrics_original['times']['R2']:>10.4f} | {metrics_original['flows']['MAPE']:>11.2f}% | {metrics_original['times']['MAPE']:>11.2f}%")
    for res in results_noise:
        print(f"{res['noise_level']*100:>7.0f}% | {res['R2_flows']:>10.4f} | {res['R2_times']:>10.4f} | {res['MAPE_flows']:>11.2f}% | {res['MAPE_times']:>11.2f}%")
    print(f"{'='*60}\n")
    
    # Visualisation optionnelle du dernier cas avec bruit
    print(f"\nVisualisation du cas avec {noise_levels[-1]*100:.0f}% de bruit:")
    plot_comparison(lam_flows_pred, lam_times_pred, msa_flows_noise, msa_times_noise, 
                   network, f"{network_name}_noise_{int(noise_levels[-1]*100)}pct")

"""

"""
from scipy.optimize import minimize_scalar
import numpy as np

def objective_eps(eps, network, path_list, G, msa_flows, msa_times, alpha, beta, method):
    # Fonction objectif : erreur combinée entre LAM et MSA selon eps.
    lam_flows, lam_times = compute_lam_solution(
        network, path_list, G, eps_num=eps,
        method=method, alpha=alpha, beta=beta
    )
    # Normalisation pour éviter les échelles différentes
    err_flows = np.linalg.norm(lam_flows - msa_flows) / np.linalg.norm(msa_flows)
    err_times = np.linalg.norm(lam_times - msa_times) / np.linalg.norm(msa_times)
    
    # pondération 50/50
    return 0.5 * (err_flows + err_times)

# Recherche optimale de eps
res = minimize_scalar(
    objective_eps, 
    bounds=(0.95, 2), 
    method='bounded',
    args=(network, path_list, G, msa_flows, msa_times, alpha, beta, lam_method)
)

eps_opt = res.x
print(f"Valeur optimale de eps : {eps_opt:.3f}")
print(f"Erreur minimale : {res.fun:.4f}")

# Puis recalcul final
lam_flows_opt, lam_times_opt = compute_lam_solution(
    network, path_list, G, eps_num=eps_opt,
    method=lam_method, alpha=alpha, beta=beta
)
"""