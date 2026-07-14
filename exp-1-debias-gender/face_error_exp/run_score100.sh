#!/bin/bash
# Shard the 12 t-ranges across 6 GPUs (each holds ~7GB, capped ~12.5GB so OUR job
# OOMs rather than the co-tenant's if memory runs out).
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -f exp100/scores_shard*.npz
GPUS=(0 1 2 3 4 5 6 7)      # all 8 GPUs, ~1 t-range each (each holds ~9GB, capped ~12.5GB)
NSH=${#GPUS[@]}
pids=()
for i in $(seq 0 $((NSH-1))); do
  g=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$g python -u score100.py --shard $i --nshards $NSH \
      > "score100_$i.log" 2>&1 &
  pids+=($!)
done
echo "launched $NSH scorers on GPUs ${GPUS[*]}: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
echo "ALL_SCORE100_DONE fail=$fail"
