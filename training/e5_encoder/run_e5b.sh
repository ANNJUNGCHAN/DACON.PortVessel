#!/usr/bin/env bash
# e5b: second e5-small encoder (seed offset 1, augmentation v2, 6000 steps); 5 folds + full.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for f in 0 1 2 3 4 -1; do
  python -X utf8 train_e5.py --fold $f --steps 6000 --name e5b --seedoff 1 --augv 2 > logs/train_e5b_f$f.log 2>&1
  echo "$(date '+%T') fold $f exit $? $(grep -o '{\"clean.*' logs/train_e5b_f$f.log | head -c 200)"
done
echo ALL_DONE
