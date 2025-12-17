# models/grn_model.py

from __future__ import annotations
import json
import csv
import os
import numpy as np
import yaml

from .base_model import DynamicalModel, ParameterSet
# from visualization.v_grn import plot_gene_regulatory_network

class GRNModel(DynamicalModel):


    def __init__(self, config):
        if isinstance(config, str):
            with open(config, "r") as f:
                config = yaml.safe_load(f)

        super().__init__(config)
        self._build_from_config(config)




    def _build_from_config(self, config):
        model_cfg = config["model"]

        self.n_genes = int(model_cfg["n_genes"])
        self.indexing = model_cfg.get("indexing", "1-based")
        self.priors = config.get("priors", {})
        self.network_cfg = config.get("network", {})

        self.time_sampling_cfg = config.get("time_sampling", {})
        self.perturbation_cfg = config.get("perturbation", {})
        self.noise_cfg = config.get("noise", {})

        if self.indexing not in ("0-based", "1-based"):
            raise ValueError
        
        if not self.priors:
            raise ValueError

        # state name：m1..mN, p1..pN
        state_names = []
        for i in range(self.n_genes):
            state_names.append(f"m{i+1}")
        for i in range(self.n_genes):
            state_names.append(f"p{i+1}")
        self._state_names = state_names

        param_names = []
        for key in ["beta", "gamma", "k", "delta", "alpha0", "m0", "p0"]:
            for i in range(self.n_genes):
                param_names.append(f"{key}_{i+1}")
        self._parameter_names = param_names



    #  sampling helpers
    def _get_prior(self, name):
        priors = self.priors
        if name not in priors:
            raise KeyError(f"Prior for parameter '{name}' not found.")
        return priors[name]

    def _sample_scalar(self, prior, rng):
        dist = prior["dist"]
        if dist == "uniform":
            return rng.uniform(prior["low"], prior["high"])
        elif dist == "loguniform":
            low = np.log(prior["low"])
            high = np.log(prior["high"])
            return np.exp(rng.uniform(low, high))
        elif dist == "normal":
            return rng.normal(prior["mean"], prior["std"])
        elif dist == "choice":
            return rng.choice(prior["choices"])
        elif dist == "fixed":
            return prior["value"]
        else:
            raise ValueError(f"unknown distribution: {dist}")
        

    
    def noise_hyperparameters(self, mode: str | None = None) -> tuple[float, float]:
        """
        根据 noise 配置返回 (sigma_rel, abs_frac)。
        mode 为 None 时使用 config 中的 noise.mode。
        """
        cfg = self.noise_cfg or {}
        if mode is None:
            mode = cfg.get("mode", "medium")

        presets = cfg.get("presets", {})
        if presets and mode in presets:
            ncfg = presets[mode]
            sigma_rel = float(ncfg.get("sigma_rel", 0.1))
            abs_frac = float(ncfg.get("abs_frac", 0.01))
        else:
            default = {
                "low":         (0.05, 0.01),
                "medium":      (0.10, 0.01),
                "high":        (0.20, 0.01),
                "abs_baseline": (0.10, 0.05),
            }
            sigma_rel, abs_frac = default.get(mode, (0.10, 0.01))

        return sigma_rel, abs_frac


    # hill function 
    def hill_act(self, p, K, n):
        return p**n / (K**n + p**n)

    def hill_rep(self, p, K, n):
        return K**n / (K**n + p**n)

    def _regulation_i(self, i, p_vec, params):
        """
        f_i(p): 第 i 个基因的调控函数 这里用的是 params.payload 里采样出来的 regulators。
        """
        alpha0 = params.payload["alpha0"][i]
        regs = params.payload["regulators"][i]

        if len(regs) == 0:
            return alpha0

        prod = 1.0
        for r in regs:
            pj = p_vec[r["source"]]
            K = r["K"]
            nh = r["n"]
            if r["mode"] == "act":
                prod *= self.hill_act(pj, K, nh)
            else:
                prod *= self.hill_rep(pj, K, nh)

        return alpha0 + (1.0 - alpha0) * prod


    def state_names(self):
        return list(self._state_names)


    def parameter_names(self):
        return list(self._parameter_names)


    def sample_ground_truth_parameters(self, rng):
        """
        核心函数：
        - 按 priors 采样每个基因的 beta/gamma/k/delta/alpha0/m0/p0
        - 按 network 超参数随机生成调控网络结构
          + 对每条边采样 K, n
        """

        n = self.n_genes
        payload = {}

        # 1) 基因级参数
        gene_param_keys = ["beta", "gamma", "k", "delta", "alpha0", "m0", "p0"]
        for key in gene_param_keys:
            prior = self._get_prior(key)
            vals = np.zeros(n)
            for i in range(n):
                vals[i] = self._sample_scalar(prior, rng)
            payload[key] = vals

        # 2) 调控网络结构 + 边上的 K, n
        net = self.network_cfg
        edge_prob = float(net.get("edge_prob", 0.2))
        allow_self = bool(net.get("allow_self", False))
        max_in_degree = int(net.get("max_in_degree", n - 1))
        min_in_degree = int(net.get("min_in_degree", 0))
        frac_activation = float(net.get("frac_activation", 0.5))

        prior_K = self._get_prior("K")
        prior_n = self._get_prior("n")

        regulators = [[] for _ in range(n)]

        # 简单 ER 图 + 入度截断
        for i in range(n):
            candidates = []
            for j in range(n):
                if (not allow_self) and (i == j):
                    continue
                if rng.uniform() < edge_prob:
                    candidates.append(j)

            rng.shuffle(candidates)
            if len(candidates) > max_in_degree:
                candidates = candidates[:max_in_degree]

            if len(candidates) < min_in_degree:
                possible = [j for j in range(n) if (allow_self or j != i)]
                rng.shuffle(possible)
                for j in possible:
                    if j not in candidates:
                        candidates.append(j)
                        if len(candidates) >= min_in_degree:
                            break

            # 为每个父节点 j 决定 act/rep，并采样 K, n
            for j in candidates:
                mode = "act" if rng.uniform() < frac_activation else "rep"
                K_val = self._sample_scalar(prior_K, rng)
                n_val = self._sample_scalar(prior_n, rng)
                regulators[i].append(
                    {
                        "source": j,   # 0-based index
                        "mode": mode,
                        "K": K_val,
                        "n": n_val,
                    }
                )

        payload["regulators"] = regulators

        names = self.parameter_names()

        return ParameterSet(payload=payload, names=names)

    def initial_state(self, params, rng, mode: str | None = None):
        m0 = params.payload["m0"].copy()
        p0 = params.payload["p0"].copy()

        m0, p0, chosen = self._apply_perturbation_to_initial_state(m0, p0, rng, mode=mode)

        # NEW: store chosen genes for logging
        self.last_chosen_genes = chosen  # list[int]

        return np.concatenate([m0, p0])


    def _apply_perturbation_to_initial_state(self, m0, p0, rng, mode: str | None = None):
        cfg = self.perturbation_cfg or {}
        if mode is None:
            mode = cfg.get("mode", "baseline")

        presets = cfg.get("presets", {})

        if mode == "baseline":
            return m0, p0, [] 
        
        p_cfg = presets.get(mode, {})
        n = self.n_genes
        n_genes = int(p_cfg.get("n_genes", 5))

        idx_all = np.arange(n)
        rng.shuffle(idx_all)
        chosen = idx_all[:n_genes].astype(int).tolist() 
        if mode == "oe":
            factor = float(p_cfg.get("factor", 3.0))
            p0[chosen] *= factor

        elif mode == "kd":
            factor = float(p_cfg.get("factor", 0.5))
            p0[chosen] *= factor

        elif mode == "kd_full":
            p0[chosen] = 0.0

        return m0, p0, chosen   # NEW: return chosen list
        


    def time_grid(self, mode: str | None = None) -> np.ndarray:
        """
        根据 time_sampling 配置返回时间网格 t_eval。
        如果 mode 为 None，就用 config 中的 time_sampling.mode。
        """
        cfg = self.time_sampling_cfg or {}
        if mode is None:
            mode = cfg.get("mode", "dense")

        presets = cfg.get("presets", {})
        if presets and mode in presets:
            ts = presets[mode]
            t0 = float(ts.get("time_start", 0.0))
            t1 = float(ts.get("time_end", 50.0))
            n_tp = int(ts.get("n_timepoints", 21))
        else:
            # fallback
            default_n = {"dense": 21, "medium": 11, "sparse": 6}
            t0, t1 = 0.0, 50.0
            n_tp = default_n.get(mode, 21)

        return np.linspace(t0, t1, n_tp)
    



    def rhs(self, t, y, params):

        """
        dy/dt = f(t, y; params)
        y = [m1..mN, p1..pN]
        """
        n = self.n_genes
        m = y[:n]
        p = y[n:]

        beta = params.payload["beta"]
        gamma = params.payload["gamma"]
        k = params.payload["k"]
        delta = params.payload["delta"]

        dm = np.empty(n)
        dp = np.empty(n)

        for i in range(n):
            fi = self._regulation_i(i, p, params)
            dm[i] = beta[i] * fi - gamma[i] * m[i]
            dp[i] = k[i] * m[i] - delta[i] * p[i]

        return np.concatenate([dm, dp])
    
    
    def observation_indices(self, observed_genes: np.ndarray) -> list[int]:
        """
        给定被观测 gene 的 0-based 索引，返回对应的蛋白在 state 向量中的索引。
        state: [m1..mN, p1..pN] → p_i 的索引是 n_genes + i
        """
        n = self.n_genes
        observed_genes = np.asarray(observed_genes, dtype=int)
        return list(n + observed_genes)



    '''
    save the gene regulatory network structure figure.
    '''

    def plot_gene_network(self, params, path_prefix):

        out_dir = os.path.dirname(path_prefix)
        os.makedirs(out_dir, exist_ok=True)

        out_path = path_prefix + "_grn.png"

        plot_gene_regulatory_network(
            params,
            self.n_genes,
            out_path,
            layout="spring"
        )