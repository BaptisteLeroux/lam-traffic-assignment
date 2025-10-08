from modules.network import load_network
from modules.numerical_solvers import MSA_solver
from modules.lam_solvers import compute_lam_solution
from modules.plot_utils import (
    plot_network_colormap, 
    plot_comparison, 
    evaluate_solution
)

# ========== CONFIGURATION ==========
network_name = "toy"  # toy ou sioux_falls

# Paramètres BPR
alpha = 0.15
beta = 4

# Paramètres MSA
N_iter = 1000
tol = 1e-3

# Méthode LAM
lam_method = 'qp'  # 'qp' ou 'linear_system'

# ========== CHARGEMENT ET VISUALISATION DU RÉSEAU ==========
print(f"\n{'='*60}")
print(f"  CHARGEMENT DU RÉSEAU: {network_name.upper()}")
print(f"{'='*60}\n")

network = load_network(network_name)
network.summary()
network.plot()

# ========== SOLUTION NUMÉRIQUE (MSA) ==========
print(f"\n{'='*60}")
print(f"  CALCUL DE LA SOLUTION NUMÉRIQUE (MSA)")
print(f"{'='*60}\n")

path_list, msa_flows, msa_times, G = MSA_solver(
    network, N_iter, tol, alpha, beta, 
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

# ========== SOLUTION ANALYTIQUE (LAM) ==========
print(f"\n{'='*60}")
print(f"  CALCUL DE LA SOLUTION ANALYTIQUE (LAM)")
print(f"{'='*60}\n")

lam_flows, lam_times = compute_lam_solution(
    network, path_list, G, msa_flows,
    method=lam_method, alpha=alpha, beta=beta
)

print(f"✓ Solution LAM calculée avec succès (méthode: {lam_method})")

# ========== COMPARAISON ET ÉVALUATION ==========
print(f"\n{'='*60}")
print(f"  COMPARAISON DES SOLUTIONS")
print(f"{'='*60}\n")

# Métriques d'erreur
metrics = evaluate_solution(lam_flows, lam_times, msa_flows, msa_times)

# ========== RÉSUMÉ ==========
print(f"\n{'='*60}")
print(f"  RÉSUMÉ")
print(f"{'='*60}")
print(f"  Réseau        : {network_name}")
print(f"  Méthode LAM   : {lam_method}")
print(f"  R² (flows)    : {metrics['flows']['R2']:.4f}")
print(f"  R² (times)    : {metrics['times']['R2']:.4f}")
print(f"  MAPE (flows)  : {metrics['flows']['MAPE']:.2f}%")
print(f"  MAPE (times)  : {metrics['times']['MAPE']:.2f}%")
print(f"{'='*60}\n")

# Graphiques comparatifs
plot_comparison(lam_flows, lam_times, msa_flows, msa_times, network, network_name)