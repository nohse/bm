#!/bin/bash
# Fan out occupation-image generation over 8 GPUs (one worker per GPU).
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
rm -f occ_candidates/shard_*.json
NSH=8
pids=()
for i in $(seq 0 $((NSH-1))); do
  CUDA_VISIBLE_DEVICES=$i python -u gen_occ_worker.py --shard $i --nshards $NSH --seeds 8 \
      > "occ_gen_$i.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} workers: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
echo "ALL_WORKERS_DONE fail=$fail"
