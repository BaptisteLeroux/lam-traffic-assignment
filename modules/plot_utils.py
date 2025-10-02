from modules import  *

def plot_flows_and_times_msa(flows, times, title_suffix=""):

    n = len(flows)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(np.arange(1, n+1), flows, '-x', label='Numerical flow')
    axs[0].set_title('Links flow' + title_suffix)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(np.arange(1, n+1), times, '-x', label='Numerical travel time')
    axs[1].set_title('Links travel time' + title_suffix)
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_network_with_values(network, flows, times, show_labels=True, node_size=300):

    G = nx.DiGraph()
    for i in range(len(network.sn)):
        G.add_edge(network.sn[i], network.en[i])

    if network.node_coords:
        pos = {node: (x, y) for node, (x, y) in network.node_coords.items()}
    else:
        pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=show_labels, node_size=node_size, arrows=True, edge_color='gray')

    edge_labels = {}
    for i, (u, v) in enumerate(zip(network.sn, network.en)):
        edge_labels[(u, v)] = f"f={flows[i]:.1f}\nt={times[i]:.1f}"

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Network with flows and travel times")
    plt.axis('off')
    plt.show()

def plot_network_colormap(network, values, value_type="flow", cmap="viridis", node_size=300, show_labels=True):

    G = nx.DiGraph()
    for i in range(len(network.sn)):
        G.add_edge(network.sn[i], network.en[i], value=values[i])

    if network.node_coords:
        pos = {node: (x, y) for node, (x, y) in network.node_coords.items()}
    else:
        pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(10, 8))

    edges = list(G.edges())
    edge_values = [G[u][v]['value'] for u, v in edges]

    nx.draw_networkx_nodes(G, pos, node_size=node_size, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    lc = nx.draw_networkx_edges(
        G, pos, edgelist=edges, edge_color=edge_values, edge_cmap=plt.get_cmap(cmap),
        width=2, arrows=True, arrowstyle='-|>', connectionstyle='arc3,rad=0.15', ax=ax
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(edge_values), vmax=max(edge_values)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(f"Link {value_type}")

    ax.set_title(f"Network {value_type} colormap")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_network_colormap_2(network, values, value_type="flow", cmap="viridis", node_size=300, show_labels=True):
    import matplotlib.patches as mpatches

    G = nx.DiGraph()
    for i in range(len(network.sn)):
        G.add_edge(network.sn[i], network.en[i], value=values[i])

    if network.node_coords:
        pos = {node: (x, y) for node, (x, y) in network.node_coords.items()}
    else:
        pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Récupérer les valeurs pour chaque arc dans l'ordre du graphe
    edges = list(G.edges())
    edge_values = [G[u][v]['value'] for u, v in edges]

    # Dessiner les noeuds normaux
    all_nodes = set(network.sn) | set(network.en)
    O_nodes = set(network.on)
    D_nodes = set(network.dn)
    other_nodes = all_nodes - O_nodes - D_nodes

    # Noeuds origine (O) en rouge
    nx.draw_networkx_nodes(G, pos, nodelist=list(O_nodes), node_color='red', node_size=node_size*1.5, ax=ax, label='Origin (O)')
    # Noeuds destination (D) en bleu
    nx.draw_networkx_nodes(G, pos, nodelist=list(D_nodes), node_color='blue', node_size=node_size*1.5, ax=ax, label='Destination (D)')
    # Autres noeuds en gris
    nx.draw_networkx_nodes(G, pos, nodelist=list(other_nodes), node_color='gray', node_size=node_size, ax=ax, label='Other')

    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    # Dessiner les arcs avec la colormap
    lc = nx.draw_networkx_edges(
        G, pos, edgelist=edges, edge_color=edge_values, edge_cmap=plt.get_cmap(cmap),
        width=2, arrows=True, arrowstyle='-|>', connectionstyle='arc3,rad=0.15', ax=ax
    )

    # Ajouter la colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(edge_values), vmax=max(edge_values)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(f"Link {value_type}")

    # Annoter les origines et destinations avec la somme des demandes
    for o in O_nodes:
        total_out = np.sum(network.q_od[network.on == o])
        ax.annotate(f"O\n{total_out:.0f}", pos[o], color='red', fontsize=11, fontweight='bold', ha='center', va='center', xytext=(0, 18), textcoords='offset points')
    for d in D_nodes:
        total_in = np.sum(network.q_od[network.dn == d])
        ax.annotate(f"D\n{total_in:.0f}", pos[d], color='blue', fontsize=11, fontweight='bold', ha='center', va='center', xytext=(0, -18), textcoords='offset points')

    # Légende personnalisée
    legend_handles = [
        mpatches.Patch(color='red', label='Origin (O)'),
        mpatches.Patch(color='blue', label='Destination (D)'),
        mpatches.Patch(color='gray', label='Other node')
    ]
    ax.legend(handles=legend_handles, loc='upper left')

    ax.set_title(f"Network {value_type} colormap\n(O: origin, D: destination, value: total demand)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()