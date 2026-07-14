#!/bin/bash
# Fan out the cond-only t-range x prompt search over 8 GPUs (shard by t-range).
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results_search
rm -f results_search/shard_*.json
NSH=8
K=${1:-25}
pids=()
for i in $(seq 0 $((NSH-1))); do
  CUDA_VISIBLE_DEVICES=$i python -u search_worker.py --shard $i --nshards $NSH --K $K \
      > "search_$i.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} search workers (K=$K): ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
echo "ALL_SEARCH_DONE fail=$fail"
