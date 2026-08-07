#!/bin/bash
# HP sweep wave 3b (resume): the muon wall-clock decider first, then endpoint
# scaling, then the bs2048 stretch goal (samples-fair vs 30k@512 throughout).
cd /jax_dir/notebooks
P=../data/aug2026/probe
run() {
  name=$1; shift
  echo "=== START $name $(date -u +%H:%M:%S) ===" >> $P/hp_sweep_wave3b.log
  python analysis/train_probe_hp.py "$@" --out-prefix $P/hp_$name > $P/hp_$name.log 2>&1 \
    && echo "=== OK $name ===" >> $P/hp_sweep_wave3b.log \
    || echo "=== FAIL $name ===" >> $P/hp_sweep_wave3b.log
}
run muon_bs1024_lr7e3 --optimizer muon --schedule cosine --lr 0.007 --batch-size 1024 --steps 15000
run muon_lr5e3_50k    --optimizer muon --schedule cosine --lr 0.005 --steps 50000
run muon_bs2048_lr1e2 --optimizer muon --schedule cosine --lr 0.01 --batch-size 2048 --steps 7500
echo "=== WAVE3B COMPLETE ===" >> $P/hp_sweep_wave3b.log
