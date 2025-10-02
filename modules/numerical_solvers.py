from modules import *
import modules.cost_functions as cf

def MSA_solver(network, N_iter, tol, alpha, beta, linearize_bpr, eps=None):

    if linearize_bpr and eps is None:
        raise ValueError("Le paramètre eps doit être fourni pour la linéarisation de BPR.")

    sn = network.sn
    en = network.en
    t0 = network.t0
    C = network.C
    on = network.on
    dn = network.dn
    q_od = network.q_od

    n = len(sn)  # nombre d'arcs
    m = len(on)  # nombre de paires OD
    V = np.zeros(n)  # flux initial sur les arcs
    tol = 1e-5

    # Chemins à chaque itération
    L = [None for _ in range(m)]       # Chemin courant pour chaque OD
    path_list = [[] for _ in range(m)]         # Tous les chemins uniques rencontrés

    # Construction initiale du graphe
    G = nx.DiGraph()
    for i in range(n):
        G.add_edge(sn[i], en[i], weight=t0[i], index=i)

    def compute_travel_times(V, linearize_bpr=False):
        if linearize_bpr == False : 
            return cf.bpr_function(V, t0, C, alpha=alpha, beta=beta)
        else : 
            return cf.linearised_bpr_function(V, t0, C, alpha=alpha, beta=4, eps=eps)
        
    def all_or_nothing_assignment(times):
        """ Charge les flux sur les plus courts chemins pour chaque OD et stocke les chemins """
        aux_flow = np.zeros(n)
        for k in range(m):
            origin = on[k]
            dest = dn[k]
            demand = q_od[k]
            
            # Mise à jour des poids dans le graphe
            for i in range(n):
                G[sn[i]][en[i]]['weight'] = times[i]
            
            try:
                path = nx.shortest_path(G, source=origin, target=dest, weight='weight')
            except nx.NetworkXNoPath:
                continue

            # Stockage du chemin courant
            L[k] = path
            if path not in path_list[k]:
                path_list[k].append(path)

            # Affecter le flux sur le chemin
            for i in range(len(path) - 1):
                link_idx = G[path[i]][path[i+1]]['index']
                aux_flow[link_idx] += demand
        return aux_flow

    # Résolution MSA
    for it in range(N_iter):
        # Calculer le pas pour cette itération
        phi = 1 / (it + 1)  
        
        times = compute_travel_times(V)
        F = all_or_nothing_assignment(times)
        
        V_new = (1 - phi) * V + phi * F
        
        gap = np.linalg.norm(V_new - V, 1)
        if gap < tol and it > 10: # Éviter de stopper trop tôt
            print(f"Convergence atteinte à l’itération {it}, écart = {gap:.6f}")
            break
        V = V_new

    if it == N_iter - 1:
        print(f"Convergence non atteinte après {N_iter} itérations, écart = {gap:.6f}")

    return path_list, V, compute_travel_times(V)

