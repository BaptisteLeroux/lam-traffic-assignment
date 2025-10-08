from modules import *
import matplotlib.patches as mpatches

# ========== VISUALISATION SIMPLE ==========

def plot_flows_and_times_msa(flows, times, title_suffix=""):
    """Plot simple des flux et temps de parcours."""
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


# ========== VISUALISATION RÉSEAU AVEC COLORMAP ==========

def _build_graph(network, values: np.ndarray) -> nx.DiGraph:
    """Construit le graphe dirigé avec les valeurs sur les arcs."""
    G = nx.DiGraph()
    for i, (start, end) in enumerate(zip(network.sn, network.en)):
        G.add_edge(start, end, value=values[i])
    return G


def _get_node_positions(network, G: nx.DiGraph) -> dict:
    """Récupère ou calcule les positions des nœuds."""
    if hasattr(network, 'node_coords') and network.node_coords:
        return {node: (x, y) for node, (x, y) in network.node_coords.items()}
    return nx.spring_layout(G, seed=42, k=2, iterations=50)


def _classify_nodes(network, G: nx.DiGraph) -> dict:
    """Classifie les nœuds en origines, destinations et autres."""
    all_nodes = set(G.nodes())
    o_nodes = set(network.on)
    d_nodes = set(network.dn)
    other_nodes = all_nodes - o_nodes - d_nodes
    
    return {
        'origin': o_nodes,
        'destination': d_nodes,
        'other': other_nodes
    }


def _compute_od_statistics(network, node_groups: dict) -> dict:
    """Calcule les statistiques agrégées pour les O/D."""
    stats = {
        'total_origin_demand': 0,
        'total_destination_demand': 0,
        'origin_details': {},
        'destination_details': {}
    }
    
    for o in node_groups['origin']:
        demand = np.sum(network.q_od[network.on == o])
        stats['origin_details'][o] = demand
        stats['total_origin_demand'] += demand
    
    for d in node_groups['destination']:
        demand = np.sum(network.q_od[network.dn == d])
        stats['destination_details'][d] = demand
        stats['total_destination_demand'] += demand
    
    return stats


def _draw_nodes(G, pos, node_groups, node_size, ax):
    """Dessine les nœuds avec des styles différents selon leur type."""
    nx.draw_networkx_nodes(
        G, pos, nodelist=list(node_groups['origin']),
        node_color='#E74C3C', node_size=node_size * 1.5, node_shape='s',
        edgecolors='black', linewidths=2, ax=ax
    )
    
    nx.draw_networkx_nodes(
        G, pos, nodelist=list(node_groups['destination']),
        node_color='#3498DB', node_size=node_size * 1.5, node_shape='D',
        edgecolors='black', linewidths=2, ax=ax
    )
    
    nx.draw_networkx_nodes(
        G, pos, nodelist=list(node_groups['other']),
        node_color='#95A5A6', node_size=node_size, node_shape='o',
        edgecolors='black', linewidths=1, ax=ax
    )


def _draw_edges(G, pos, cmap, ax):
    """Dessine les arcs avec la colormap."""
    edges = list(G.edges())
    edge_values = [G[u][v]['value'] for u, v in edges]
    
    nx.draw_networkx_edges(
        G, pos, edgelist=edges, edge_color=edge_values,
        edge_cmap=plt.get_cmap(cmap), width=2.5, arrows=True,
        arrowstyle='-|>', arrowsize=15, connectionstyle='arc3,rad=0.15', ax=ax
    )


def _draw_od_annotations(pos, node_groups, od_stats, ax):
    """Ajoute les annotations de demande sur les nœuds O/D."""
    for o in node_groups['origin']:
        demand = od_stats['origin_details'][o]
        ax.annotate(
            f"{demand:.0f}", pos[o], color='#C0392B',
            fontsize=9, fontweight='bold', ha='center', va='center',
            xytext=(0, 20), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='#E74C3C', alpha=0.8)
        )
    
    for d in node_groups['destination']:
        demand = od_stats['destination_details'][d]
        ax.annotate(
            f"{demand:.0f}", pos[d], color='#2874A6',
            fontsize=9, fontweight='bold', ha='center', va='center',
            xytext=(0, -20), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='#3498DB', alpha=0.8)
        )


def _add_colorbar(G, cmap, value_type, fig, ax):
    """Ajoute la colorbar pour les valeurs des arcs."""
    edges = list(G.edges())
    edge_values = [G[u][v]['value'] for u, v in edges]
    
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=min(edge_values), vmax=max(edge_values))
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Link {value_type}", fontsize=11, fontweight='bold')


def _add_enhanced_legend(ax, od_stats, show_annotations):
    """Ajoute une légende enrichie avec les statistiques O/D."""
    legend_elements = [
        mpatches.Patch(
            facecolor='#E74C3C', edgecolor='black', linewidth=2,
            label=f"Origin (□) - Total: {od_stats['total_origin_demand']:.0f}"
        ),
        mpatches.Patch(
            facecolor='#3498DB', edgecolor='black', linewidth=2,
            label=f"Destination (◇) - Total: {od_stats['total_destination_demand']:.0f}"
        ),
        mpatches.Patch(
            facecolor='#95A5A6', edgecolor='black',
            label="Intermediate node (○)"
        )
    ]
    
    if show_annotations:
        legend_elements.append(
            mpatches.Patch(
                facecolor='white', edgecolor='gray',
                label="Numbers = node demand"
            )
        )
    
    ax.legend(
        handles=legend_elements, loc='upper left',
        fontsize=10, framealpha=0.95, edgecolor='black'
    )


def plot_network_colormap(network, values: np.ndarray, value_type: str = "flow",
                          cmap: str = "viridis", node_size: int = 300,
                          show_labels: bool = True, show_od_annotations: bool = False,
                          figsize: tuple = (12, 9)):
    """
    Visualise un réseau de transport avec une colormap pour les arcs.
    
    Args:
        network: Objet réseau
        values: Valeurs à afficher sur les arcs
        value_type: Type de valeur (flow, travel time, etc.)
        cmap: Colormap matplotlib
        node_size: Taille des nœuds
        show_labels: Afficher les labels des nœuds
        show_od_annotations: Afficher les annotations O/D
        figsize: Taille de la figure
    """
    G = _build_graph(network, values)
    pos = _get_node_positions(network, G)
    node_groups = _classify_nodes(network, G)
    od_stats = _compute_od_statistics(network, node_groups)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    _draw_nodes(G, pos, node_groups, node_size, ax)
    _draw_edges(G, pos, cmap, ax)
    
    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    
    if show_od_annotations:
        _draw_od_annotations(pos, node_groups, od_stats, ax)
    
    _add_colorbar(G, cmap, value_type, fig, ax)
    _add_enhanced_legend(ax, od_stats, show_od_annotations)
    
    ax.set_title(f"Network {value_type} visualization", 
                fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# ========== COMPARAISON LAM VS NUMÉRIQUE ==========

def plot_comparison(lam_flows, lam_times, msa_flows, msa_times, network, network_name=""):
    """
    Compare les solutions LAM et numériques (MSA).
    
    Args:
        lam_flows, lam_times: Solutions analytiques
        msa_flows, msa_times: Solutions numériques
        network: Objet Network
        network_name: Nom du réseau pour le titre
    """
    n = len(network.sn)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Flux
    axs[0].plot(np.arange(1, n+1), lam_flows, '-o', label='Analytical (LAM)', linewidth=2)
    axs[0].plot(np.arange(1, n+1), msa_flows, '-x', label='Numerical (MSA)', linewidth=2)
    axs[0].set_xlabel('Link index')
    axs[0].set_ylabel('Flow')
    axs[0].set_title(f'Links flow - {network_name}')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Temps
    axs[1].plot(np.arange(1, n+1), lam_times, '-o', label='Analytical (LAM)', linewidth=2)
    axs[1].plot(np.arange(1, n+1), msa_times, '-x', label='Numerical (MSA)', linewidth=2)
    axs[1].set_xlabel('Link index')
    axs[1].set_ylabel('Travel time')
    axs[1].set_title(f'Links travel time - {network_name}')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ========== MÉTRIQUES D'ERREUR ==========

def compute_metrics(y_true, y_pred, label=""):
    """
    Calcule les métriques d'erreur entre prédictions et valeurs réelles.
    
    Args:
        y_true: Valeurs réelles
        y_pred: Valeurs prédites
        label: Label pour l'affichage
    
    Returns:
        dict: Dictionnaire contenant RMSE, MAE, MAPE, R²
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MAPE : {mape:.2f} %")
    print(f"  R²   : {r2:.4f}")
    print(f"{'='*50}\n")
    
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def evaluate_solution(lam_flows, lam_times, msa_flows, msa_times):
    """
    Évalue la qualité de la solution LAM par rapport à MSA.
    
    Args:
        lam_flows, lam_times: Solutions analytiques
        msa_flows, msa_times: Solutions numériques de référence
    
    Returns:
        dict: Métriques pour flows et times
    """
    print("\n" + "="*60)
    print("  ÉVALUATION DE LA SOLUTION ANALYTIQUE (LAM vs MSA)")
    print("="*60)
    
    metrics_flow = compute_metrics(msa_flows, lam_flows, label="Link Flows")
    metrics_time = compute_metrics(msa_times, lam_times, label="Link Travel Times")
    
    return {
        'flows': metrics_flow,
        'times': metrics_time
    }