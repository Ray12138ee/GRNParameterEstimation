#!/usr/bin/env bash
set -e

# ============================
# 用户可调参数
# ============================

ROOT_DIR="data_gene200"        # 24 个 net* 文件夹所在目录
N_STARTS=10            # multi-start 次数（Phase I 用 5~10 足够）
N_PROCS=8              # 每个任务用多少进程
PAR_JOBS=4             # 同时跑多少个数据集（4×8=32 核）
OPTIMIZER="scipy"      # scipy / fides

# ============================
# 防止 BLAS 抢核（非常重要）
# ============================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[INFO] ROOT_DIR=$ROOT_DIR"
echo "[INFO] PAR_JOBS=$PAR_JOBS, N_PROCS=$N_PROCS, N_STARTS=$N_STARTS"

# ============================
# 收集所有 petab.yaml
# ============================
find "$ROOT_DIR" -type f -path "*/petab/petab.yaml" | sort > petab_list.txt

N_TOTAL=$(wc -l < petab_list.txt)
echo "[INFO] Found $N_TOTAL PEtab problems"

# ============================
# 并行运行
# ============================
cat petab_list.txt | xargs -n 1 -P "$PAR_JOBS" -I {} bash -c '
  PETAB_YAML="$1"
  RUN_DIR="$(dirname "$PETAB_YAML")/.."
  FIT_DIR="$RUN_DIR/fit"

  mkdir -p "$FIT_DIR"

  echo "[RUN] $RUN_DIR"

  python multistart.py \
    --petab_yaml "$PETAB_YAML" \
    --n_starts '"$N_STARTS"' \
    --n_procs '"$N_PROCS"' \
    --optimizer '"$OPTIMIZER"' \
    --out_hdf5 "$FIT_DIR/result.hdf5" \
    --summary_json "$FIT_DIR/summary.json"

  echo "[DONE] $RUN_DIR"
' _ {}

echo "[OK] ALL FINISHED"