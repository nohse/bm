#!/bin/bash
# Round-2 scoring: shard (t-range x seed) jobs over 8 GPUs. Output -> exp100/scores_b/
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p exp100/scores_b
rm -f exp100/scores_b/scores_shard*.npz
GPUS=(0 1 2 3 4 5 6 7)
NSH=${#GPUS[@]}
pids=()
for i in $(seq 0 $((NSH-1))); do
  g=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$g python -u score100.py --shard $i --nshards $NSH \
      --cfg exp100b_config --outdir exp100/scores_b > "score100b_$i.log" 2>&1 &
  pids+=($!)
done
echo "launched $NSH round-2 scorers: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
echo "ALL_SCORE100B_DONE fail=$fail"
