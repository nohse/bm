#!/bin/bash
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p exp100/scores_c
rm -f exp100/scores_c/scores_shard*.npz
GPUS=(0 1 2 3 4 5 6 7); NSH=${#GPUS[@]}; pids=()
for i in $(seq 0 $((NSH-1))); do
  g=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$g python -u score100.py --shard $i --nshards $NSH \
      --cfg exp100c_config --outdir exp100/scores_c > "score100c_$i.log" 2>&1 &
  pids+=($!)
done
echo "launched $NSH round-3 scorers: ${pids[*]}"
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
echo "ALL_SCORE100C_DONE fail=$fail"
