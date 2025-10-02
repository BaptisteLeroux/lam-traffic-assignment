from modules.network import load_network
from modules.numerical_solvers import MSA_solver
from modules.plot_utils import plot_flows_and_times_msa, plot_network_colormap, plot_network_colormap

network_name = "sioux_falls" # toy or sioux_falls
"Paramètres"
#BPR
alpha = 0.15
beta = 4
eps = None # Only if linearize_bpr = True

#MSA solver
N_iter = 500
tol = 1e-3

network = load_network(network_name)

network.summary()
network.plot()

path_list, msa_flows, msa_times = MSA_solver(network, N_iter, tol, alpha, beta, linearize_bpr=False, eps=None)

eps_num = msa_flows / network.C
print("Congestion moyenne du réseau (flow/capacity):", eps_num.mean())

# print (msa_flows)

# plot_flows_and_times_msa(msa_flows, msa_times)

plot_network_colormap(network, msa_flows, value_type="flow", cmap="plasma", show_od_annotations=True)

plot_network_colormap(network, msa_times, value_type="travel time", cmap="plasma", show_od_annotations=True)

#lam_flows, lam_times = lam_solver(path_list, network, alpha, beta, eps)

#plot_results(msa_flows, msa_times, lam_flows, lam_times)

