from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

import pypesto.optimize as optimize
import pypesto.engine as engine
import pypesto.store as store


# ---------------------------
# 简单成功率定义
# ---------------------------
def compute_success_rate(result, rtol: float = 1e-2) -> float:
    fvals = [
        r.fval for r in result.optimize_result.list
        if r is not None and r.fval is not None and np.isfinite(r.fval)
    ]
    if not fvals:
        return 0.0
    f_best = min(fvals)
    thr = f_best + rtol * max(1.0, abs(f_best))
    return float(np.mean([fv <= thr for fv in fvals]))


# ---------------------------
# Fallback: a pure-Python least-squares "dummy PEtab"
# ---------------------------
def build_fallback_problem(petab_yaml: Path):
    """
    纯 Python fallback，不依赖 AMICI。
    从 petab.yaml 读 measurements.tsv，构造一个“常数模型”：
        y_hat = c_obs  (每个 observable 一个常数)
    目标函数：sum_{all points} (y - c_obs)^2

    这个模型能保证：
    - 每个数据集都能跑 multi-start（很快）
    - 输出 best_fval / success_rate 让你把 pipeline 跑通、写论文结构

    注意：这不是你的 GRN mechanistic 拟合，只是应急替代。
    """
    import yaml as _yaml
    import pandas as pd
    import pypesto

    petab_dir = petab_yaml.parent
    y = _yaml.safe_load(petab_yaml.read_text())
    prob = y["problems"][0]
    meas_path = (petab_dir / prob["measurement_files"][0]).resolve()

    meas = pd.read_csv(meas_path, sep="\t")
    if "observableId" not in meas.columns or "measurement" not in meas.columns:
        raise ValueError("Fallback needs measurements.tsv with columns: observableId, measurement")

    obs_ids = sorted(meas["observableId"].astype(str).unique().tolist())
    obs_to_idx = {o: i for i, o in enumerate(obs_ids)}

    y_meas = meas["measurement"].astype(float).to_numpy()
    o_idx = meas["observableId"].astype(str).map(obs_to_idx).to_numpy()

    dim = len(obs_ids)

    # objective: f(x) = sum_i (y_i - x[o_i])^2
    def fun(x):
        r = y_meas - x[o_idx]
        return float(np.dot(r, r))

    # gradient
    def grad(x):
        g = np.zeros_like(x)
        r = y_meas - x[o_idx]
        # d/dx_k sum (y - x[o])^2 = -2 * sum_{i:o_i=k} (y_i - x_k)
        for k in range(dim):
            mk = (o_idx == k)
            if np.any(mk):
                g[k] = -2.0 * np.sum(r[mk])
        return g

    lb = np.full(dim, -1e6)
    ub = np.full(dim, 1e6)
    x0 = np.zeros(dim)

    problem = pypesto.Problem(
        objective=pypesto.Objective(fun=fun, grad=grad),
        lb=lb,
        ub=ub,
        x_guesses=[x0],
        x_names=[f"c_{o}" for o in obs_ids],
    )
    return problem, {"fallback_model": "per-observable constant", "n_params": dim}


def main():
    ap = argparse.ArgumentParser(description="Run multi-start optimization for ONE PEtab problem")
    ap.add_argument("--petab_yaml", required=True, help="Path to petab.yaml")
    ap.add_argument("--n_starts", type=int, default=10, help="Number of starts")
    ap.add_argument("--n_procs", type=int, default=1, help="Parallel processes")
    ap.add_argument("--optimizer", choices=["scipy", "fides"], default="scipy")
    ap.add_argument("--out_hdf5", default="result.hdf5")
    ap.add_argument("--summary_json", default="summary.json")
    ap.add_argument("--prefer_amici", action="store_true",
                    help="Try AMICI first; if it fails, fall back to pure Python dummy model.")
    args = ap.parse_args()

    petab_yaml = Path(args.petab_yaml).resolve()
    out_hdf5 = Path(args.out_hdf5).resolve()
    out_hdf5.parent.mkdir(parents=True, exist_ok=True)

    used_backend = None
    backend_error = None
    extra = {}

    # ---------------------------
    # 1) Build problem (AMICI or fallback)
    # ---------------------------
    problem = None
    if args.prefer_amici:
        try:
            import pypesto.petab
            importer = pypesto.petab.PetabImporter.from_yaml(str(petab_yaml), simulator_type="amici")
            problem = importer.create_problem()
            used_backend = "amici"
        except Exception as e:
            backend_error = f"{type(e).__name__}: {e}"
            problem, extra = build_fallback_problem(petab_yaml)
            used_backend = "fallback"
    else:
        # default: go directly to fallback to avoid AMICI headaches
        problem, extra = build_fallback_problem(petab_yaml)
        used_backend = "fallback"

    # ---------------------------
    # 2) Optimizer
    # ---------------------------
    if args.optimizer == "scipy":
        optimizer = optimize.ScipyOptimizer()
    else:
        optimizer = optimize.FidesOptimizer()

    # ---------------------------
    # 3) Engine
    # ---------------------------
    if args.n_procs > 1:
        eng = engine.MultiProcessEngine(n_procs=args.n_procs)
    else:
        eng = engine.SingleCoreEngine()

    # ---------------------------
    # 4) Multi-start optimization
    # ---------------------------
    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=args.n_starts,
        engine=eng,
        progress_bar=True,
    )

    # ---------------------------
    # 5) Save result
    # ---------------------------
    store.write_result(result, str(out_hdf5), overwrite=True)

    # ---------------------------
    # 6) Summary
    # ---------------------------
    fvals = [
        r.fval for r in result.optimize_result.list
        if r is not None and r.fval is not None and np.isfinite(r.fval)
    ]

    summary = {
        "petab_yaml": str(petab_yaml),
        "backend": used_backend,
        "backend_error": backend_error,
        "optimizer": args.optimizer,
        "n_starts": args.n_starts,
        "n_procs": args.n_procs,
        "n_valid_starts": len(fvals),
        "best_fval": (min(fvals) if fvals else None),
        "success_rate": compute_success_rate(result),
        **extra,
    }

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] Optimization finished")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()