#!/bin/bash
# Fan out genuine no-face generation over 8 GPUs.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p noface_candidates
rm -f noface_candidates/shard_*.json
NSH=8
pids=()
for i in $(seq 0 $((NSH-1))); do
  CUDA_VISIBLE_DEVICES=$i python -u gen_noface_worker.py --shard $i --nshards $NSH --seeds 2 \
      > "nf_gen_$i.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} noface workers: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
echo "ALL_NF_DONE fail=$fail"
