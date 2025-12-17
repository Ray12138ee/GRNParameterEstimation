# simulator/GRN_synthetic.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np
import yaml
from scipy.integrate import solve_ivp

from models.grn_model import GRNModel
from models.base_model import ParameterSet  # 如果不需要类型提示可以删掉这个 import


@dataclass
class GRNConditionResult:

    t: np.ndarray                    # shape (T,)
    clean_state: np.ndarray          # shape (2N, T)
    clean_obs: np.ndarray            # shape (n_obs, T)
    noisy_obs: np.ndarray            # shape (n_reps, n_obs, T)
    perturbed_genes: list[int] 
    
@dataclass
class GRNSyntheticExperimentResult:
    """
    总的返回结构：
      - model: GRNModel 实例
      - params_true: ground-truth ParameterSet
      - observed_genes: 被观测的基因索引（0-based）
      - results: dict[(time_mode, perturb_mode, noise_mode) -> GRNConditionResult]
    """
    model: GRNModel
    params_true: ParameterSet
    observed_genes: np.ndarray
    results: Dict[Tuple[str, str, str], GRNConditionResult]


def select_observed_genes(
    model: GRNModel,
    obs_cfg: dict,
) -> np.ndarray:
    """
    根据 experiment.observation 的配置，选择被观测的基因子集。

    obs_cfg:
      mode: "all" or "subset"
      fraction_observed: float in (0,1]
      random_seed: int
    """
    mode = obs_cfg.get("mode", "subset")
    if mode == "all":
        return np.arange(model.n_genes, dtype=int)

    frac = float(obs_cfg.get("fraction_observed", 0.3))
    n_obs = max(1, int(round(model.n_genes * frac)))
    seed = int(obs_cfg.get("random_seed", 123))

    rng = np.random.default_rng(seed)
    observed_genes = rng.choice(model.n_genes, size=n_obs, replace=False)
    observed_genes = np.sort(observed_genes.astype(int))
    return observed_genes


def add_gaussian_noise(
    clean_obs: np.ndarray,
    sigma_rel: float,
    abs_frac: float,
    n_reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    对 clean_obs 加 Gaussian 噪声，生成 n_reps 个技术重复。

    clean_obs: shape (n_obs, T)
    返回 noisy_obs: shape (n_reps, n_obs, T)

    噪声标准差：
        sigma = sigma_rel * |y| + abs_frac * y_scale
    其中 y_scale 可以用全局均值/标准差，这里用 clean_obs 的均值绝对值。
    """
    n_obs, T = clean_obs.shape
    noisy = np.empty((n_reps, n_obs, T), dtype=float)

    # 避免全 0 时噪声为 0，给一个 scale baseline
    y_abs_mean = np.mean(np.abs(clean_obs))
    if y_abs_mean <= 0.0:
        y_abs_mean = 1.0

    for r in range(n_reps):
        sigma = sigma_rel * np.abs(clean_obs) + abs_frac * y_abs_mean
        eps = rng.normal(loc=0.0, scale=sigma)
        noisy[r] = clean_obs + eps

    return noisy



def run_grn_synthetic_experiment(exp_config_path: str) -> GRNSyntheticExperimentResult:
    """
    读取 grn_experiment.yaml，构建 GRNModel，采样一套 ground-truth 参数，
    然后对指定的 time_sampling / perturbation / noise 模式组合生成合成数据。

    返回 GRNSyntheticExperimentResult，全部在内存中，不写磁盘。
    """

    # 1) 读 experiment 配置
    with open(exp_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    model_config_path = exp_cfg["model_config"]
    seed_global = int(exp_cfg.get("random_seed", 0))
    rng_global = np.random.default_rng(seed_global)

    # 2) 构建 GRNModel + ground-truth 参数
    model = GRNModel(model_config_path)
    params_true = model.sample_ground_truth_parameters(rng_global)

    # 3) 选择被观测的基因子集
    obs_cfg = exp_cfg["observation"]
    observed_genes = select_observed_genes(model, obs_cfg)
    obs_idx = model.observation_indices(observed_genes)

    # 4) 要跑哪些模式
    time_sampling_modes = list(exp_cfg.get("time_sampling_modes", ["dense"]))
    perturbation_modes = list(exp_cfg.get("perturbation_modes", ["baseline"]))
    noise_modes = list(exp_cfg.get("noise_modes", ["medium"]))

    n_reps = int(exp_cfg.get("n_replicates", 1))

    results: Dict[Tuple[str, str, str], GRNConditionResult] = {}

    # 5) 穷举所有组合
    for t_mode in time_sampling_modes:
        # 时间网格
        t_eval = model.time_grid(mode=t_mode)
        t0, t1 = float(t_eval[0]), float(t_eval[-1])

        for p_mode in perturbation_modes:
            # 为这个 (t_mode, p_mode) 组合生成一个初始状态
            rng_cond = np.random.default_rng(rng_global.integers(0, 2**32 - 1))
            y0 = model.initial_state(params_true, rng_cond, mode=p_mode)
            chosen = getattr(model, "last_chosen_genes", [])



            # 5.1 先做一次 ODE 仿真（clean state）
            def rhs_wrapper(t, y):
                return model.rhs(t, y, params_true)

            sol = solve_ivp(
                fun=rhs_wrapper,
                t_span=(t0, t1),
                y0=y0,
                t_eval=t_eval,
                vectorized=False,
                rtol=1e-6,
                atol=1e-8,
            )

            if not sol.success:
                raise RuntimeError(
                    f"ODE integration failed for (t_mode={t_mode}, perturb_mode={p_mode}): "
                    f"{sol.message}"
                )

            clean_state = sol.y  # shape (2N, T)
            clean_obs = clean_state[obs_idx, :]  # shape (n_obs, T)

            # 5.2 对不同噪声模式加噪，生成多个重复
            for n_mode in noise_modes:
                sigma_rel, abs_frac = model.noise_hyperparameters(mode=n_mode)

                rng_noise = np.random.default_rng(
                    rng_global.integers(0, 2**32 - 1)
                )

                noisy_obs = add_gaussian_noise(
                    clean_obs=clean_obs,
                    sigma_rel=sigma_rel,
                    abs_frac=abs_frac,
                    n_reps=n_reps,
                    rng=rng_noise,
                )

                key = (t_mode, p_mode, n_mode)
                results[key] = GRNConditionResult(
                    t=t_eval,
                    clean_state=clean_state,
                    clean_obs=clean_obs,
                    noisy_obs=noisy_obs,
                     perturbed_genes=chosen,   
                )

    return GRNSyntheticExperimentResult(
        model=model,
        params_true=params_true,
        observed_genes=observed_genes,
        results=results,
    )

