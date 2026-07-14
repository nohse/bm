#!/bin/bash
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p exp100/scores_attn; rm -f exp100/scores_attn/scores_shard*.npz
GPUS=(0 1 2 3 4 5 6 7); NSH=${#GPUS[@]}; pids=()
for i in $(seq 0 $((NSH-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} python -u score_attn.py --shard $i --nshards $NSH > "score_attn_$i.log" 2>&1 &
  pids+=($!)
done
echo "launched $NSH attn scorers: ${pids[*]}"
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
echo "ALL_ATTN_DONE fail=$fail"
