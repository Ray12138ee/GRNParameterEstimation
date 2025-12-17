from __future__ import annotations
import os
from pathlib import Path
import yaml
import numpy as np
import libsbml

from simulator.GRN_synthetic import run_grn_synthetic_experiment


# ----------------------------
# Small IO helper
# ----------------------------
def _write_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


# ----------------------------
# SBML export (Scheme B)
# ----------------------------
def export_grn_to_sbml(model, params, sbml_path: Path):
    n = model.n_genes
    regs = params.payload["regulators"]

    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel()
    m.setId("grn_model")

    comp = m.createCompartment()
    comp.setId("cell")
    comp.setSize(1.0)
    comp.setConstant(True)

    # species with fixed initial values
    for i in range(1, n + 1):
        s = m.createSpecies()
        s.setId(f"m{i}")
        s.setCompartment("cell")
        s.setInitialAmount(float(params.payload["m0"][i-1]))
        s.setConstant(False)

    for i in range(1, n + 1):
        s = m.createSpecies()
        s.setId(f"p{i}")
        s.setCompartment("cell")
        s.setInitialAmount(float(params.payload["p0"][i-1]))
        s.setConstant(False)

    def add_param(pid, val):
        p = m.createParameter()
        p.setId(pid)
        p.setValue(float(val))
        p.setConstant(True)

    for key in ["beta", "gamma", "k", "delta", "alpha0"]:
        for i in range(n):
            add_param(f"{key}_{i+1}", params.payload[key][i])

    for i in range(n):
        for r in regs[i]:
            j = r["source"] + 1
            add_param(f"K_{i+1}_{j}", r["K"])
            add_param(f"n_{i+1}_{j}", r["n"])

    for i in range(1, n + 1):
        terms = []
        for r in regs[i-1]:
            j = r["source"] + 1
            if r["mode"] == "act":
                terms.append(f"pow(p{j},n_{i}_{j})/(pow(K_{i}_{j},n_{i}_{j})+pow(p{j},n_{i}_{j}))")
            else:
                terms.append(f"pow(K_{i}_{j},n_{i}_{j})/(pow(K_{i}_{j},n_{i}_{j})+pow(p{j},n_{i}_{j}))")

        fi = f"alpha0_{i}" if not terms else f"alpha0_{i}+(1-alpha0_{i})*({'*'.join(terms)})"

        rr = m.createRateRule()
        rr.setVariable(f"m{i}")
        rr.setMath(libsbml.parseL3Formula(f"beta_{i}*({fi})-gamma_{i}*m{i}"))

        rr = m.createRateRule()
        rr.setVariable(f"p{i}")
        rr.setMath(libsbml.parseL3Formula(f"k_{i}*m{i}-delta_{i}*p{i}"))

    libsbml.writeSBMLToFile(doc, str(sbml_path))


# ----------------------------
# PEtab export (robust)
# ----------------------------
def export_grn_petab_with_sbml(exp_config, out_dir):
    res = run_grn_synthetic_experiment(exp_config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    export_grn_to_sbml(res.model, res.params_true, out / "model.xml")

    # observables
    obs_rows = []
    for g in res.observed_genes:
        gid = g + 1
        obs_rows.append([
            f"obs_p{gid}",
            f"p{gid}",
            "noiseParameter1*abs(observable)+noiseParameter2",
            "normal",
            f"protein_{gid}",
        ])
    _write_tsv(out / "observables.tsv",
               ["observableId","observableFormula","noiseFormula","noiseDistribution","observableName"],
               obs_rows)

    # single condition
    _write_tsv(out / "conditions.tsv",
               ["conditionId","conditionName"],
               [["cond0","baseline"]])

    # measurements
    meas = []
    for (_,_,nm), cr in res.results.items():
        for r in range(cr.noisy_obs.shape[0]):
            for i, gid in enumerate(res.observed_genes):
                for t, val in zip(cr.t, cr.noisy_obs[r,i]):
                    meas.append([
                        f"obs_p{gid+1}",
                        "cond0",
                        float(t),
                        float(val),
                        f"sigma_rel_{nm};sigma_abs_{nm}",
                        f"rep{r+1}",
                    ])
    _write_tsv(out / "measurements.tsv",
               ["observableId","simulationConditionId","time","measurement","noiseParameters","replicateId"],
               meas)

    # parameters
    rows = []
    for pid in res.model.parameter_names():
        rows.append([pid,pid,"lin",1e-6,1e2,1.0,1])
    for nm in {k[2] for k in res.results}:
        rows.append([f"sigma_rel_{nm}",f"sigma_rel_{nm}","lin",1e-8,1.0,0.1,0])
        rows.append([f"sigma_abs_{nm}",f"sigma_abs_{nm}","lin",1e-8,1e6,1.0,0])
    _write_tsv(out / "parameters.tsv",
               ["parameterId","parameterName","parameterScale","lowerBound","upperBound","nominalValue","estimate"],
               rows)

    with open(out / "petab.yaml","w") as f:
        yaml.safe_dump({
            "format_version":1,
            "parameter_file":"parameters.tsv",
            "problems":[{
                "sbml_files":["model.xml"],
                "condition_files":["conditions.tsv"],
                "observable_files":["observables.tsv"],
                "measurement_files":["measurements.tsv"],
            }]
        }, f)
