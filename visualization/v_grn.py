# plot_grn_multipanel.py
# Usage:
#   python plot_grn_multipanel.py
#
# Output:
#   outputs/appendix/grn_multipanel.png

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx


# =========================
# Minimal ParameterSet mock
# =========================
class ParameterSet:
    def __init__(self, payload: dict):
        self.payload = payload


# =========================
# Random GRN generator
# =========================
def generate_random_grn_params(
    n_genes: int,
    p_act: float = 0.10,
    p_rep: float = 0.10,
    allow_self: bool = True,
    K_range=(0.2, 2.0),
    n_range=(1.0, 4.0),
    seed: int | None = None,
) -> ParameterSet:
    rng = np.random.default_rng(seed)
    regulators = []

    for target in range(n_genes):
        regs_t = []
        for source in range(n_genes):
            if (not allow_self) and (source == target):
                continue

            u = rng.random()
            if u < p_act:
                regs_t.append(
                    {
                        "source": source,
                        "mode": "act",
                        "K": float(rng.uniform(*K_range)),
                        "n": float(rng.uniform(*n_range)),
                    }
                )
            elif u < p_act + p_rep:
                regs_t.append(
                    {
                        "source": source,
                        "mode": "rep",
                        "K": float(rng.uniform(*K_range)),
                        "n": float(rng.uniform(*n_range)),
                    }
                )

        regulators.append(regs_t)

    return ParameterSet(payload={"regulators": regulators})


# =========================
# Single-panel GRN plot
# =========================
def plot_grn_on_ax(ax, params, n_genes, layout="spring", title=None, subtitle=None):
    regulators = params.payload["regulators"]
    G = nx.DiGraph()

    for target in range(n_genes):
        tgt = f"G{target+1}"
        G.add_node(tgt)
        for r in regulators[target]:
            src = f"G{r['source']+1}"
            G.add_node(src)
            G.add_edge(src, tgt, mode=r["mode"])

    if layout == "spring":
        pos = nx.spring_layout(G, seed=0)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=0)

    ax.set_axis_off()

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=650,
        node_color="white",
        edgecolors="black",
        linewidths=1.3,
    )

    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=9,
    )

    act_edges = [(u, v) for u, v, d in G.edges(data=True) if d["mode"] == "act"]
    rep_edges = [(u, v) for u, v, d in G.edges(data=True) if d["mode"] == "rep"]

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=act_edges,
        edge_color="tab:blue",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=13,
        width=1.6,
    )

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=rep_edges,
        edge_color="tab:red",
        style="dashed",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=13,
        width=1.6,
    )

    if title:
        ax.set_title(title, fontsize=11, pad=4)

    if subtitle:
        ax.text(
            0.02, 0.02, subtitle,
            transform=ax.transAxes,
            fontsize=8,
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.7),
        )


# =========================
# 3 x 2 multipanel figure
# =========================
def plot_grn_3x2(panels, out_path, figsize=(12, 15), suptitle=None):
    if len(panels) != 6:
        raise ValueError("3x2 layout requires exactly 6 panels.")

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.reshape(-1)

    for ax, p in zip(axes, panels):
        plot_grn_on_ax(
            ax=ax,
            params=p["params"],
            n_genes=p["n_genes"],
            layout=p.get("layout", "spring"),
            title=f"N = {p['n_genes']} genes",
            subtitle=p.get("desc", ""),
        )

    # global legend
    act_line = mlines.Line2D([], [], color="tab:blue", label="Activation")
    rep_line = mlines.Line2D([], [], color="tab:red", linestyle="dashed", label="Repression")
    fig.legend(
        handles=[act_line, rep_line],
        loc="upper center",
        ncol=2,
        frameon=True,
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=0.995)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved 3x2 GRN figure to: {out_path}")


# =========================
# Main
# =========================
def main():
    gene_sizes = [8, 10, 15, 25, 30,40]

    panels = []
    for i, N in enumerate(gene_sizes):
        params = generate_random_grn_params(
            n_genes=N,
            p_act=0.10,
            p_rep=0.10,
            allow_self=True,
            seed=10 + i,
        )
        panels.append(
            {
                "params": params,
                "n_genes": N,
                "layout": "spring" if N <= 15 else "kamada_kawai",
                "desc": f"Random GRN with {N} genes.",
            }
        )

    plot_grn_3x2(
        panels=panels,
        out_path="outputs/appendix/ap_relation.png",
        figsize=(12, 14),
        # suptitle="Appendix: Randomly Generated Gene Regulatory Networks",
    )


if __name__ == "__main__":
    main()