#!/bin/bash
# HP sweep wave 3: muon endpoint — second seed confirm, step scaling, and the
# bs1024 wall-clock variant (samples-fair vs 30k@512).
cd /jax_dir/notebooks
P=../data/aug2026/probe
run() {
  name=$1; shift
  echo "=== START $name $(date -u +%H:%M:%S) ===" >> $P/hp_sweep_wave3.log
  python analysis/train_probe_hp.py "$@" --out-prefix $P/hp_$name > $P/hp_$name.log 2>&1 \
    && echo "=== OK $name ===" >> $P/hp_sweep_wave3.log \
    || echo "=== FAIL $name ===" >> $P/hp_sweep_wave3.log
}
run muon_lr5e3_30k_s2   --optimizer muon --schedule cosine --lr 0.005 --steps 30000 --seed 2
run muon_bs1024_lr7e3   --optimizer muon --schedule cosine --lr 0.007 --batch-size 1024 --steps 15000
run muon_lr5e3_50k      --optimizer muon --schedule cosine --lr 0.005 --steps 50000
echo "=== WAVE3 COMPLETE ===" >> $P/hp_sweep_wave3.log
