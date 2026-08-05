#!/bin/bash
# HP sweep wave 1 — sequential GPU queue; each run logs to probe/hp_<name>.log
cd /jax_dir/notebooks
P=../data/aug2026/probe
run() {
  name=$1; shift
  echo "=== START $name $(date -u +%H:%M:%S) ===" >> $P/hp_sweep_wave1.log
  python analysis/train_probe_hp.py "$@" --out-prefix $P/hp_$name > $P/hp_$name.log 2>&1 \
    && echo "=== OK $name ===" >> $P/hp_sweep_wave1.log \
    || echo "=== FAIL $name ===" >> $P/hp_sweep_wave1.log
}
run cosine_lr3e4   --schedule cosine --lr 3e-4
run cosine_lr1e3   --schedule cosine --lr 1e-3
run plateau_lr6e4  --schedule plateau --lr 6e-4
run ademamix_lr3e4 --optimizer ademamix --schedule plateau --lr 3e-4
run lion_lr1e4     --optimizer lion --schedule plateau --lr 1e-4
run muon_lr2e2     --optimizer muon --schedule cosine --lr 0.02 --steps 12000
run muon_lr5e3     --optimizer muon --schedule cosine --lr 0.005 --steps 12000
echo "=== WAVE1 COMPLETE ===" >> $P/hp_sweep_wave1.log
