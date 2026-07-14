#!/bin/bash
# Fan out residual-error scoring of the 40 images over 8 GPUs (shard by image).
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
rm -f results_occ/shard_*.json
NSH=8
pids=()
for i in $(seq 0 $((NSH-1))); do
  CUDA_VISIBLE_DEVICES=$i python -u score_occ_worker.py --shard $i --nshards $NSH \
      > "occ_score_$i.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} scoring workers: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
echo "ALL_SCORE_DONE fail=$fail"
