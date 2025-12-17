import yaml, copy
from pathlib import Path
from export import export_grn_petab_with_sbml

base = yaml.safe_load(open("configs/grn_experiment.yaml"))

out_root = Path("data_gene200")
out_root.mkdir(exist_ok=True)

for net in [0,1,2]:
    for obs in [0.3,1.0]:
        for t in ["dense","sparse"]:
            for p in ["baseline","kd_full"]:
                cfg = copy.deepcopy(base)
                cfg["experiment"]["random_seed"] = net
                cfg["experiment"]["observation"]["mode"] = "all" if obs==1.0 else "subset"
                cfg["experiment"]["observation"]["fraction_observed"] = obs
                cfg["experiment"]["time_sampling_modes"] = [t]
                cfg["experiment"]["perturbation_modes"] = [p]
                cfg["experiment"]["noise_modes"] = ["medium"]

                name = f"net{net}__obs{obs}__t_{t}__p_{p}"
                d = out_root / name
                d.mkdir(parents=True, exist_ok=True)

                yml = d / "experiment.yaml"
                yaml.safe_dump(cfg, open(yml,"w"))

                export_grn_petab_with_sbml(str(yml), d/"petab")
                print("[OK]", name)
