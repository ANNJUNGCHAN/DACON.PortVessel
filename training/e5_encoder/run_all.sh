#!/usr/bin/env bash
# e5ko: 5 fold models + full model, sequential (one GPU). Logs per fold.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for f in 0 1 2 3 4 -1; do
  python -X utf8 train_e5.py --fold $f > logs/train_f$f.log 2>&1
  echo "$(date '+%T') fold $f exit $? $(grep -o '{\"clean.*' logs/train_f$f.log | head -c 300)"
done
echo ALL_DONE
