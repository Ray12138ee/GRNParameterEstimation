import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
from scipy.integrate import solve_ivp


# =========================
# 1) Define the same simple fixed GRN (12 genes)
# =========================
def build_simple_grn(n_genes=20):
    """
    Return parents dict:
      parents[i] = list of tuples (j, mode) where mode in {"act","rep"}
    A simple hand-crafted structure: ring + cross-links.
    """
    parents = {i: [] for i in range(n_genes)}

    # Ring activation: (i-1) -> i  (activation)
    for i in range(n_genes):
        parents[i].append(((i - 1) % n_genes, "act"))

    # Ring repression: (i-2) -> i (repression)
    for i in range(n_genes):
        parents[i].append(((i - 2) % n_genes, "rep"))

    # A few extra cross edges to create non-trivial topology
    extra = [
        (0, 5, "act"),
        (3, 8, "rep"),
        (6, 2, "act"),
        # (9, 1, "rep"),
        # (4, 10, "act"),
        # (7, 11, "rep"),
    ]
    for j, i, m in extra:
        parents[i].append((j, m))

    # allow a couple of self regulations
    parents[2].append((2, "rep"))
    parents[7].append((7, "act"))
    return parents


# =========================
# 2) Parameters and ODE model
# =========================
def sample_initial_params(n_genes, seed=0):
    rng = np.random.default_rng(seed)
    # beta = rng.uniform(0.8, 2.0, size=n_genes)
    # gamma = rng.uniform(0.05, 0.25, size=n_genes)

    # K_act = rng.uniform(0.5, 2.0, size=n_genes)
    # n_act = rng.uniform(1.0, 3.0, size=n_genes)
    # K_rep = rng.uniform(0.5, 2.0, size=n_genes)
    # n_rep = rng.uniform(1.0, 3.0, size=n_genes)

    # x0 = rng.uniform(0.05, 0.5, size=n_genes)
    beta  = rng.uniform(2.0, 5.0, size=n_genes)
    gamma = rng.uniform(0.5, 1.5, size=n_genes)
    n_act = rng.uniform(4.0, 7.0, size=n_genes)
    n_rep = rng.uniform(4.0, 7.0, size=n_genes)
    K_act = rng.uniform(0.2, 0.8, size=n_genes)
    K_rep = rng.uniform(0.2, 0.8, size=n_genes)
    x0 = rng.uniform(0.8, 2.5, size=n_genes)
    return {
        "beta": beta,
        "gamma": gamma,
        "K_act": K_act,
        "n_act": n_act,
        "K_rep": K_rep,
        "n_rep": n_rep,
        "x0": x0,
    }


def hill_act(x, K, n):
    xn = np.power(np.maximum(x, 0.0), n)
    Kn = np.power(K, n)
    return xn / (Kn + xn + 1e-12)


def hill_rep(x, K, n):
    xn = np.power(np.maximum(x, 0.0), n)
    Kn = np.power(K, n)
    return Kn / (Kn + xn + 1e-12)


def make_rhs(parents, params, perturbation):
    beta0 = params["beta"].copy()
    gamma = params["gamma"]
    K_act = params["K_act"]
    n_act = params["n_act"]
    K_rep = params["K_rep"]
    n_rep = params["n_rep"]

    if perturbation == "baseline":
        t_kd = None
        kd_scale = 1.0
    elif perturbation == "kd-full":
        t_kd = 0.0
        kd_scale = 0.3
    else:
        raise ValueError(f"Unknown perturbation: {perturbation}")

    def rhs(t, x):
        if (t_kd is not None) and (t >= t_kd):
            beta = kd_scale * beta0
        else:
            beta = beta0

        dx = np.zeros_like(x)

        for i in range(len(x)):
            acts = [j for (j, m) in parents[i] if m == "act"]
            reps = [j for (j, m) in parents[i] if m == "rep"]

            reg = 1.0
            for j in acts:
                reg *= hill_act(x[j], K_act[i], n_act[i])
            for j in reps:
                reg *= hill_rep(x[j], K_rep[i], n_rep[i])

            dx[i] = beta[i] * reg - gamma[i] * x[i]

        return dx

    return rhs


def simulate(parents, params, t_eval, perturbation):
    rhs = make_rhs(parents, params, perturbation)
    x0 = params["x0"]

    sol = solve_ivp(
        fun=lambda t, x: rhs(t, x),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=x0,
        t_eval=np.asarray(t_eval, dtype=float),
        method="LSODA",
        rtol=1e-7,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y


# =========================
# 3) Noise model
# =========================
def add_noise(y, rng, sigma_abs=0.15, sigma_rel=0.08):
    sigma = sigma_abs + sigma_rel * np.abs(y)
    return y + rng.normal(loc=0.0, scale=sigma, size=y.shape)


# =========================
# 4) Experiment design
# =========================
def get_time_grid(tag, t0=0.0, t1=50.0):
    if tag == "dense":
        return np.linspace(t0, t1, 21)
    elif tag == "sparse":
        return np.linspace(t0, t1, 6)
    else:
        raise ValueError(tag)


def choose_observed_genes(n_genes, coverage, fixed_plot_genes):
    if np.isclose(coverage, 1.0):
        return list(range(n_genes))
    else:
        return list(fixed_plot_genes)


# =========================
# 5) Convert parents dict to ParameterSet-like format
# =========================
def parents_to_parameterset(parents, n_genes):
    regulators = []
    
    for target in range(n_genes):
        regs_t = []
        for source, mode in parents[target]:
            regs_t.append({
                "source": source,
                "mode": mode,
                "K": 1.0,
                "n": 2.0,
            })
        regulators.append(regs_t)
    
    class SimpleParamSet:
        def __init__(self, regulators):
            self.payload = {"regulators": regulators}
    
    return SimpleParamSet(regulators)


# =========================
# 6) 2x2动态图绘制函数
# =========================
def plot_budget_2x2_dynamic(
    parents,
    params,
    combos,
    plot_genes,
    out_path,
    noise_sigma_abs=0.15,
    noise_sigma_rel=0.08,
    n_noisy_reps=3,
    seed=0,
    fig_title=None
):
    """
    绘制2x2布局的动态子图，每个子图包含4行1列的小图
    """
    rng_master = np.random.default_rng(seed)

    # 创建2x2布局的图形
    fig = plt.figure(figsize=(18, 20))
    
    # 创建网格布局，增加垂直间距
    gs_main = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2, top=0.9, bottom=0.1, left=0.1, right=0.95)
    
    # 创建标题的位置
    if fig_title:
        plt.suptitle(fig_title, fontsize=18, fontweight='bold', y=0.97)
    
    for idx, combo in enumerate(combos):
        if idx >= 4:  # 只绘制前4个组合
            break
            
        # 为每个2x2的单元格创建子网格
        row = idx // 2
        col = idx % 2
        
        # 在2x2的每个单元格中创建4x1的子网格，增加行间距
        inner_gs = gs_main[row, col].subgridspec(len(plot_genes), 1, hspace=0.4)
        
        # 获取子图的坐标轴
        sub_axs = [fig.add_subplot(inner_gs[k, 0]) for k in range(len(plot_genes))]
        
        obs_cov = combo["obs_cov"]
        t_tag = combo["time"]
        pert = combo["pert"]
        run_id = combo["run"]

        t_eval = get_time_grid(t_tag)
        t, X_clean = simulate(parents, params, t_eval, pert)
        observed = choose_observed_genes(len(params["beta"]), obs_cov, plot_genes)
        
        # 设置每个组合的标题，使用相对位置
        title_y = 0.925 - row * 0.44
        fig.text(col * 0.5 + 0.28, title_y, 
                f"Run {run_id}: obs={obs_cov}, time={t_tag} ({len(t_eval)}), pert={pert}", 
                ha='center', fontsize=12, fontweight='bold', transform=fig.transFigure)

        for k, gid in enumerate(plot_genes):
            ax = sub_axs[k]
            y_clean = X_clean[gid]

            # clean (thick)
            ax.plot(t, y_clean, linewidth=2.2, label="clean")

            # noisy reps (thin)
            for rep in range(n_noisy_reps):
                rng = np.random.default_rng(rng_master.integers(0, 10**9))
                y_noisy = add_noise(
                    y_clean,
                    rng=rng,
                    sigma_abs=noise_sigma_abs,
                    sigma_rel=noise_sigma_rel,
                )
                ax.plot(t, y_noisy, linewidth=1.0, alpha=0.9, label="noisy" if rep == 0 else None)

            # mark if this gene is observed under the coverage setting
            obs_flag = "observed" if gid in observed else "unobserved"
            ax.set_title(f"G{gid+1} ({obs_flag})", fontsize=10, pad=5)  # 增加标题的上下间距

            # 设置y轴标签
            ax.set_ylabel("protein", fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            
            # 只在最下面的子图显示x轴标签
            if k == len(plot_genes) - 1:
                ax.set_xlabel("time", fontsize=9, labelpad=6)
            else:
                # 隐藏x轴标签
                ax.set_xlabel("")
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', length=0)  # 隐藏刻度

            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
            
            # 调整y轴范围，为标题留出空间
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(y_min, y_max * 1.1)  # 增加10%的顶部空间

            # legends only on first mini-plot to reduce clutter
            if k == 0:
                # 将图例放在更合适的位置
                ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.8)
    
    # 调整整体布局，使用更宽松的设置
    plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.08, hspace=0.3, wspace=0.2)
    
    # 添加整体图例
    clean_line = mlines.Line2D([], [], color="black", label="Clean signal", linewidth=2.2)
    noisy_line = mlines.Line2D([], [], color="black", linewidth=1.0, alpha=0.7, label="Noisy replicates")
    fig.legend(
        handles=[clean_line, noisy_line],
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=11,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.5, 0.01)
    )
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)
# =========================
# 7) GRN网络图绘制函数
# =========================
def plot_grn_network(parents, n_genes, out_path, figsize=(16, 8)):
    """
    绘制单独的GRN网络图
    """
    # Convert parents to ParameterSet
    params = parents_to_parameterset(parents, n_genes)
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 计算网络统计信息
    total_edges = sum(len(edges) for edges in parents.values())
    act_edges = sum(1 for edges in parents.values() for _, mode in edges if mode == "act")
    rep_edges = total_edges - act_edges
    self_edges = sum(1 for i, edges in parents.items() for j, _ in edges if i == j)
    
    regulators = params.payload["regulators"]
    G = nx.DiGraph()

    for target in range(n_genes):
        tgt = f"G{target+1}"
        G.add_node(tgt)
        for r in regulators[target]:
            src = f"G{r['source']+1}"
            G.add_node(src)
            G.add_edge(src, tgt, mode=r["mode"])

    # 使用spring布局
    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=100)
    ax.set_axis_off()

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=900,
        node_color="white",
        edgecolors="black",
        linewidths=1.8,
    )

    # Draw node labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=11,
        font_weight="bold",
    )

    # Separate activation and repression edges
    act_edges_list = [(u, v) for u, v, d in G.edges(data=True) if d["mode"] == "act"]
    rep_edges_list = [(u, v) for u, v, d in G.edges(data=True) if d["mode"] == "rep"]

    # Draw activation edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=act_edges_list,
        edge_color="tab:blue",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        width=1.8,
        connectionstyle="arc3,rad=0.1",
    )

    # Draw repression edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=rep_edges_list,
        edge_color="tab:red",
        style="dashed",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        width=1.8,
        connectionstyle="arc3,rad=0.1",
    )

    # 设置标题
    ax.set_title(f"Gene Regulatory Network (N={n_genes} genes)", 
                 fontsize=16, fontweight="bold", pad=20)
    
    # 添加统计信息框
    stats_text = f"Total edges: {total_edges}\nActivation edges: {act_edges}\nRepression edges: {rep_edges}\nSelf-regulation edges: {self_edges}"
    
    # 放置统计信息在右上角
    ax.text(0.98, 0.98, stats_text, 
            transform=ax.transAxes,
            fontsize=12, 
            fontfamily='monospace',
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.95, 
                     edgecolor="gray", linewidth=1.5))
    
    # 添加图例
    act_line = mlines.Line2D([], [], color="tab:blue", label="Activation", linewidth=2)
    rep_line = mlines.Line2D([], [], color="tab:red", linestyle="dashed", label="Repression", linewidth=2)
    ax.legend(handles=[act_line, rep_line], loc='upper left', fontsize=12, 
              fancybox=True, shadow=True, framealpha=0.9)
    
    # 调整布局
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    



# =========================
# 8) Main函数
# =========================
def main():
    # fixed design
    n_genes = 25
    parents = build_simple_grn(n_genes=n_genes)
    params = sample_initial_params(n_genes=n_genes, seed=1)

    # Choose 4 genes to visualize
    plot_genes = [0, 9, 2, 6]  # G1, G10, G3, G7

    # 8 combinations from your Table 1
    all_combos = [
        {"run": 1, "obs_cov": 0.3, "time": "dense",  "pert": "baseline"},
        {"run": 2, "obs_cov": 0.3, "time": "dense",  "pert": "kd-full"},
        {"run": 3, "obs_cov": 0.3, "time": "sparse", "pert": "baseline"},
        {"run": 4, "obs_cov": 0.3, "time": "sparse", "pert": "kd-full"},
        {"run": 5, "obs_cov": 1.0, "time": "dense",  "pert": "baseline"},
        {"run": 6, "obs_cov": 1.0, "time": "dense",  "pert": "kd-full"},
        {"run": 7, "obs_cov": 1.0, "time": "sparse", "pert": "baseline"},
        {"run": 8, "obs_cov": 1.0, "time": "sparse", "pert": "kd-full"},
    ]

    # 将8个组合分成两组，每组4个
    combos_group1 = all_combos[:4]  # 组合1-4
    combos_group2 = all_combos[4:]  # 组合5-8

    # 绘制第一张预算图（组合1-4）
    out_path1 = "outputs/appendix/fig_budget_1.png"
    plot_budget_2x2_dynamic(
        parents=parents,
        params=params,
        combos=combos_group1,
        plot_genes=plot_genes,
        out_path=out_path1,
        noise_sigma_abs=0.05,
        noise_sigma_rel=0.01,
        n_noisy_reps=2,
        seed=12,
        fig_title="Budget Experiments: Runs 1-4 (Obs=0.3)"
    )

    # 绘制第二张预算图（组合5-8）
    out_path2 = "outputs/appendix/fig_budget_2.png"
    plot_budget_2x2_dynamic(
        parents=parents,
        params=params,
        combos=combos_group2,
        plot_genes=plot_genes,
        out_path=out_path2,
        noise_sigma_abs=0.1,
        noise_sigma_rel=0.1,
        n_noisy_reps=1,
        seed=12,
        fig_title="Budget Experiments: Runs 5-8 (Obs=1.0)"
    )

    # 绘制GRN网络图
    out_path3 = "outputs/appendix/grn_network.png"
    plot_grn_network(
        parents=parents,
        n_genes=n_genes,
        out_path=out_path3,
        figsize=(16, 12)
    )


if __name__ == "__main__":
    main()