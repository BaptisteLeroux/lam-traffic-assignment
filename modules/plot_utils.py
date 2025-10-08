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

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional


def plot_network_colormap(
    network,
    values: np.ndarray,
    value_type: str = "flow",
    cmap: str = "viridis",
    node_size: int = 300,
    show_labels: bool = True,
    show_od_annotations: bool = False,
    figsize: Tuple[int, int] = (12, 9)
):
    """
    Visualise un réseau de transport avec une colormap pour les arcs.
    
    Args:
        network: Objet réseau contenant sn, en, on, dn, q_od, node_coords
        values: Valeurs à afficher sur les arcs (flux, coûts, etc.)
        value_type: Type de valeur affichée (pour le label)
        cmap: Colormap matplotlib
        node_size: Taille des nœuds
        show_labels: Afficher les labels des nœuds
        show_od_annotations: Afficher les annotations O/D sur les nœuds
        figsize: Taille de la figure
    """
    # 1. Construction du graphe
    G = _build_graph(network, values)
    
    # 2. Calcul des positions
    pos = _get_node_positions(network, G)
    
    # 3. Classification des nœuds
    node_groups = _classify_nodes(network, G)
    
    # 4. Calcul des statistiques O/D
    od_stats = _compute_od_statistics(network, node_groups)
    
    # 5. Création de la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # 6. Dessin des éléments
    _draw_nodes(G, pos, node_groups, node_size, ax)
    _draw_edges(G, pos, cmap, ax)
    
    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    
    if show_od_annotations:
        _draw_od_annotations(pos, node_groups, od_stats, ax)
    
    # 7. Ajout de la colorbar
    _add_colorbar(G, cmap, value_type, fig, ax)
    
    # 8. Légende améliorée
    _add_enhanced_legend(ax, od_stats, show_od_annotations)
    
    # 9. Finalisation
    ax.set_title(
        f"Network {value_type} visualization",
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def _build_graph(network, values: np.ndarray) -> nx.DiGraph:
    """Construit le graphe dirigé avec les valeurs sur les arcs."""
    G = nx.DiGraph()
    for i, (start, end) in enumerate(zip(network.sn, network.en)):
        G.add_edge(start, end, value=values[i])
    return G


def _get_node_positions(network, G: nx.DiGraph) -> Dict:
    """Récupère ou calcule les positions des nœuds."""
    if hasattr(network, 'node_coords') and network.node_coords:
        return {node: (x, y) for node, (x, y) in network.node_coords.items()}
    return nx.spring_layout(G, seed=42, k=2, iterations=50)


def _classify_nodes(network, G: nx.DiGraph) -> Dict[str, set]:
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


def _compute_od_statistics(network, node_groups: Dict) -> Dict:
    """Calcule les statistiques agrégées pour les O/D."""
    stats = {
        'total_origin_demand': 0,
        'total_destination_demand': 0,
        'origin_details': {},
        'destination_details': {}
    }
    
    # Statistiques par origine
    for o in node_groups['origin']:
        demand = np.sum(network.q_od[network.on == o])
        stats['origin_details'][o] = demand
        stats['total_origin_demand'] += demand
    
    # Statistiques par destination
    for d in node_groups['destination']:
        demand = np.sum(network.q_od[network.dn == d])
        stats['destination_details'][d] = demand
        stats['total_destination_demand'] += demand
    
    return stats


def _draw_nodes(G, pos, node_groups, node_size, ax):
    """Dessine les nœuds avec des styles différents selon leur type."""
    # Origines en rouge
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=list(node_groups['origin']),
        node_color='#E74C3C',  # Rouge vif
        node_size=node_size * 1.5,
        node_shape='s',  # Carré pour les origines
        edgecolors='black',
        linewidths=2,
        ax=ax
    )
    
    # Destinations en bleu
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=list(node_groups['destination']),
        node_color='#3498DB',  # Bleu vif
        node_size=node_size * 1.5,
        node_shape='D',  # Diamant pour les destinations
        edgecolors='black',
        linewidths=2,
        ax=ax
    )
    
    # Autres nœuds en gris
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=list(node_groups['other']),
        node_color='#95A5A6',  # Gris
        node_size=node_size,
        node_shape='o',  # Cercle
        edgecolors='black',
        linewidths=1,
        ax=ax
    )


def _draw_edges(G, pos, cmap, ax):
    """Dessine les arcs avec la colormap."""
    edges = list(G.edges())
    edge_values = [G[u][v]['value'] for u, v in edges]
    
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        edge_color=edge_values,
        edge_cmap=plt.get_cmap(cmap),
        width=2.5,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=15,
        connectionstyle='arc3,rad=0.15',
        ax=ax
    )


def _draw_od_annotations(pos, node_groups, od_stats, ax):
    """Ajoute les annotations de demande sur les nœuds O/D."""
    # Annotations origines
    for o in node_groups['origin']:
        demand = od_stats['origin_details'][o]
        ax.annotate(
            f"{demand:.0f}",
            pos[o],
            color='#C0392B',
            fontsize=9,
            fontweight='bold',
            ha='center',
            va='center',
            xytext=(0, 20),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E74C3C', alpha=0.8)
        )
    
    # Annotations destinations
    for d in node_groups['destination']:
        demand = od_stats['destination_details'][d]
        ax.annotate(
            f"{demand:.0f}",
            pos[d],
            color='#2874A6',
            fontsize=9,
            fontweight='bold',
            ha='center',
            va='center',
            xytext=(0, -20),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#3498DB', alpha=0.8)
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
            facecolor='#E74C3C',
            edgecolor='black',
            linewidth=2,
            label=f"Origin (□) - Total demand: {od_stats['total_origin_demand']:.0f}"
        ),
        mpatches.Patch(
            facecolor='#3498DB',
            edgecolor='black',
            linewidth=2,
            label=f"Destination (◇) - Total demand: {od_stats['total_destination_demand']:.0f}"
        ),
        mpatches.Patch(
            facecolor='#95A5A6',
            edgecolor='black',
            label="Intermediate node (○)"
        )
    ]
    
    if show_annotations:
        legend_elements.append(
            mpatches.Patch(
                facecolor='white',
                edgecolor='gray',
                label="Numbers = node demand"
            )
        )
    
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        fontsize=10,
        framealpha=0.95,
        edgecolor='black'
    )

def plot_lam_vs_num(q_lam, t_lam, q_num, t_num, network, network_name):

    n = len(network.sn)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(np.arange(1, n+1), q_lam, '-o', label='Analytical flow')
    axs[0].plot(np.arange(1, n+1), q_num, '-x', label='Numerical flow')
    axs[0].set_title('Links flow on ' + network_name + ' network')
    axs[0].legend()
    axs[0].grid(True)

    # Deuxième graphique : Temps
    axs[1].plot(np.arange(1, n+1), t_lam, '-o', label='Analytical travel time')
    axs[1].plot(np.arange(1, n+1), t_num, '-x', label='Numerical travel time')
    axs[1].set_title('Links travel time on ' + network_name + ' network')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def compute_errors(y_true, y_pred, label=""):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100  # éviter division par 0
    
    print(f"\n--- {label} ---")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MAPE : {mape:.2f} %")
    print(f"R²   : {r2:.4f}")
    
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}