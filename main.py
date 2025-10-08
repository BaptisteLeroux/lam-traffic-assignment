from modules.network import load_network
from modules.numerical_solvers import MSA_solver
from modules.plot_utils import plot_flows_and_times_msa, plot_network_colormap, plot_network_colormap, plot_lam_vs_num, compute_errors
from modules.lam_solvers import build_gamma_matrix, lam_solver_qp, lam_solver_linear_system, extract_dimensions, build_T_m, build_delta_matrix, build_T_matrix, extract_multiple_indices, build_gamma_m_matrix, build_delta_m_matrix, build_delta_m_delta_mu_delta_um, rref, build_A_Rr_and_r0, build_B_q0

network_name = "toy" # toy or sioux_falls
"Paramètres"
#BPR
alpha = 0.15
beta = 4
eps = None # Only if linearize_bpr = True

#MSA solver
N_iter = 1000
tol = 1e-3

network = load_network(network_name)

network.summary()
network.plot()

path_list, msa_flows, msa_times, G = MSA_solver(network, N_iter, tol, alpha, beta, linearize_bpr=False, eps=None)

eps_num = msa_flows / network.C
print("Congestion moyenne du réseau (flow/capacity):", round(eps_num.mean(),2))

plot_network_colormap(network, msa_flows, value_type="flow", cmap="plasma", show_od_annotations=True)

plot_network_colormap(network, msa_times, value_type="travel time", cmap="plasma", show_od_annotations=True)

gamma = build_gamma_matrix(path_list, network)
delta = build_delta_matrix(path_list, network, G)
T = build_T_matrix(path_list, network, G)

indices = extract_multiple_indices(path_list, delta)
gamma_m = build_gamma_m_matrix(gamma, indices)
delta_m = build_delta_m_matrix(delta, indices)

delta_m, delta_m_u, delta_u_m, final_indices = build_delta_m_delta_mu_delta_um(delta, delta_m,gamma_m, indices, network, path_list)

delta_m_tilde, rank_delta_m = rref(delta_m)

A, Rr, r0, q_od_m = build_A_Rr_and_r0(delta_m, gamma_m, final_indices, network)

B, q0, q_od_u = build_B_q0(delta_m,delta_m_tilde, delta_m_u, network, final_indices)
T_m = build_T_m(T, final_indices)
dimensions = extract_dimensions(delta_m, delta_m_tilde, A, T_m)


alpha_lam, beta_lam = 0.15, 4

lam_flows, lam_times = lam_solver_linear_system(network, final_indices, dimensions, alpha_lam, beta_lam, eps_num, A, T_m, B, r0, q0, delta, delta_u_m, q_od_u, q_od_m)

lam_flows_qp, lam_times_qp = lam_solver_qp(network, final_indices, dimensions, alpha_lam, beta_lam, eps_num, A, B, q0, r0, delta, delta_u_m, q_od_u, q_od_m)

errors_flow = compute_errors(msa_flows, lam_flows_qp, label="Link Flows")
errors_time = compute_errors(msa_times, lam_times_qp, label="Link Travel Times")

plot_lam_vs_num(lam_flows_qp, lam_times_qp, msa_flows, msa_times, network, network_name)

