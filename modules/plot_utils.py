"""
Module unifié pour la visualisation et l'évaluation des solutions LAM vs MSA.
Combine les meilleures versions issues de tes deux modules.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error
)

# ===============================================================
# OUTILS INTERNES
# ===============================================================

def _ensure_output_dir():
    """Crée le dossier output s'il n'existe pas."""
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def _get_output_filename(network_name, beta, plot_type, suffix=""):
    """Génère un nom de fichier standardisé."""
    output_dir = _ensure_output_dir()

    if suffix:
        filename = f"{network_name}_beta{beta}_{plot_type}_{suffix}.png"
    else:
        filename = f"{network_name}_beta{beta}_{plot_type}.png"

    return os.path.join(output_dir, filename)


# ===============================================================
# 1. VISUALISATION SIMPLE (courbes 1D)
# ===============================================================

def plot_flows_and_times_msa(flows, times, title_suffix=""):
    """Plot simple des flux et temps de parcours."""
    n = len(flows)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(np.arange(1, n+1), flows, '-x', label='Flow')
    axs[0].set_title('Links flow' + title_suffix)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(np.arange(1, n+1), times, '-x', label='Travel time')
    axs[1].set_title('Links travel time' + title_suffix)
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


# ===============================================================
# 2. VISUALISATION RÉSEAU AVEC COLORMAP (version améliorée)
# ===============================================================

def _build_graph(network, values: np.ndarray) -> nx.DiGraph:
    G = nx.DiGraph()
    for i, (start, end) in enumerate(zip(network.sn, network.en)):
        G.add_edge(start, end, value=values[i])
    return G


def _get_node_positions(network, G):
    if hasattr(network, 'node_coords') and network.node_coords:
        return {n: (x, y) for n, (x, y) in network.node_coords.items()}
    return nx.spring_layout(G, seed=42, k=2, iterations=50)


def _classify_nodes(network, G):
    all_nodes = set(G.nodes())
    o = set(network.on)
    d = set(network.dn)
    return {
        "origin": o,
        "destination": d,
        "other": all_nodes - o - d
    }


def _compute_od_stats(network, node_groups):
    stats = {"origin": {}, "destination": {}, "totO": 0, "totD": 0}

    for o in node_groups["origin"]:
        demand = np.sum(network.q_od[network.on == o])
        stats["origin"][o] = demand
        stats["totO"] += demand

    for d in node_groups["destination"]:
        demand = np.sum(network.q_od[network.dn == d])
        stats["destination"][d] = demand
        stats["totD"] += demand

    return stats


def plot_network_colormap(
        network, values, value_type="flow",
        cmap="viridis", node_size=300,
        show_labels=True, show_od_annotations=False,
        figsize=(12, 9),
        network_name="network", beta=1, suffix=""
    ):
    """
    Visualisation réseau unifiée (meilleure version).
    """
    G = _build_graph(network, values)
    pos = _get_node_positions(network, G)
    node_groups = _classify_nodes(network, G)
    od_stats = _compute_od_stats(network, node_groups)

    fig, ax = plt.subplots(figsize=figsize)

    # ---- Nœuds ----
    nx.draw_networkx_nodes(G, pos, nodelist=list(node_groups["origin"]),
                           node_color='#E74C3C', node_size=node_size*1.5,
                           node_shape='s', edgecolors='black', linewidths=2, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=list(node_groups["destination"]),
                           node_color='#3498DB', node_size=node_size*1.5,
                           node_shape='D', edgecolors='black', linewidths=2, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=list(node_groups["other"]),
                           node_color='#95A5A6', node_size=node_size,
                           node_shape='o', edgecolors='black', linewidths=1, ax=ax)

    # ---- Arcs ----
    edges = list(G.edges())
    edge_values = [G[u][v]['value'] for u, v in edges]

    nx.draw_networkx_edges(
        G, pos, edgelist=edges,
        edge_color=edge_values, edge_cmap=plt.get_cmap(cmap),
        width=2.5, arrows=True, arrowstyle='-|>',
        arrowsize=15, connectionstyle='arc3,rad=0.15', ax=ax
    )

    # ---- Labels ----
    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    # ---- Annotations OD ----
    if show_od_annotations:
        for o in node_groups['origin']:
            ax.annotate(
                f"{od_stats['origin'][o]:.0f}",
                pos[o], xytext=(0, 20),
                textcoords="offset points",
                ha='center', bbox=dict(
                    boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='#E74C3C', alpha=0.8)
            )
        for d in node_groups['destination']:
            ax.annotate(
                f"{od_stats['destination'][d]:.0f}",
                pos[d], xytext=(0, -20),
                textcoords="offset points",
                ha='center', bbox=dict(
                    boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='#3498DB', alpha=0.8)
            )

    # ---- Colorbar ----
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(
            vmin=np.min(edge_values), vmax=np.max(edge_values))
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax).set_label(f"{value_type}", fontsize=11)

    # ---- Légende ----
    ax.legend(handles=[
        mpatches.Patch(color='#E74C3C', label=f"Origins (Total {od_stats['totO']:.0f})"),
        mpatches.Patch(color='#3498DB', label=f"Destinations (Total {od_stats['totD']:.0f})"),
        mpatches.Patch(color='#95A5A6', label="Intermediate")
    ], fontsize=10)


    ax.set_title(
        f"{value_type.capitalize()} - {network_name} (β={beta})",
        fontsize=14, fontweight='bold'
    )
    ax.axis("off")

    plt.tight_layout()

    filename = _get_output_filename(network_name, beta, value_type, suffix)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"   → Figure sauvegardée : {filename}")

    plt.show()


# ===============================================================
# 3. COMPARAISON LAM vs MSA (version améliorée)
# ===============================================================

def plot_comparison(lam_flows, lam_times, msa_flows, msa_times, network=None,
                    network_name="", beta=1, suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Flux ----
    ax = axes[0]
    ax.scatter(msa_flows, lam_flows, alpha=0.6)
    max_f = max(msa_flows.max(), lam_flows.max())
    ax.plot([0, max_f], [0, max_f], 'r--')

    r2 = r2_score(msa_flows, lam_flows)
    mape = mean_absolute_percentage_error(msa_flows, lam_flows) * 100
    ax.set_title(f"Flows (R² {r2:.3f}, MAPE {mape:.2f}%)")

    # ---- Temps ----
    ax = axes[1]
    ax.scatter(msa_times, lam_times, alpha=0.6, color="orange")
    max_t = max(msa_times.max(), lam_times.max())
    ax.plot([0, max_t], [0, max_t], 'r--')
    r2 = r2_score(msa_times, lam_times)
    mape = mean_absolute_percentage_error(msa_times, lam_times) * 100
    ax.set_title(f"Travel times (R² {r2:.3f}, MAPE {mape:.2f}%)")

    plt.tight_layout()
    filename = _get_output_filename(network_name, beta, "comparison", suffix)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"   → Figure sauvegardée : {filename}")
    plt.show()


# ===============================================================
# 4. METRIQUES D’ERREUR — version unifiée et améliorée
# ===============================================================

def compute_metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "R2": r2_score(y_true, y_pred)
    }


def evaluate_solution(lam_flows, lam_times, msa_flows, msa_times):
    """Évalue globalement la solution LAM."""
    return {
        "flows": compute_metrics(msa_flows, lam_flows),
        "times": compute_metrics(msa_times, lam_times)
    }


def plot_prediction_graph_dual(lam_flows, lam_times, msa_flows, msa_times,
                               network, network_name="", beta=1, suffix=""):
    """
    Trace flux + temps pour LAM vs MSA avec index des liens
    et sauvegarde automatique comme plot_comparison.
    """

    n = len(network.sn)
    x = np.arange(1, n + 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # -------------------------
    # Flux
    # -------------------------
    r2_f = r2_score(msa_flows, lam_flows)
    mape_f = mean_absolute_percentage_error(msa_flows, lam_flows) * 100

    axs[0].plot(x, lam_flows, '-o', label='LAM', linewidth=2)
    axs[0].plot(x, msa_flows, '-x', label='MSA', linewidth=2)
    axs[0].set_xlabel("Link index")
    axs[0].set_ylabel("Flow")
    axs[0].set_title(f"Flows – R²: {r2_f:.3f}, MAPE: {mape_f:.2f}%")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # -------------------------
    # Temps
    # -------------------------
    r2_t = r2_score(msa_times, lam_times)
    mape_t = mean_absolute_percentage_error(msa_times, lam_times) * 100

    axs[1].plot(x, lam_times, '-o', label='LAM', linewidth=2)
    axs[1].plot(x, msa_times, '-x', label='MSA', linewidth=2)
    axs[1].set_xlabel("Link index")
    axs[1].set_ylabel("Travel time")
    axs[1].set_title(f"Travel times – R²: {r2_t:.3f}, MAPE: {mape_t:.2f}%")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # -------------------------
    # Sauvegarde automatique
    # -------------------------
    filename = _get_output_filename(network_name, beta, "prediction_dual", suffix)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"   → Figure sauvegardée : {filename}")

    plt.show()
