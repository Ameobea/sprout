#!/bin/bash
# HP sweep wave 2: muon endpoint question (does 4x step-efficiency convert to a
# better ceiling?), cosine confirmation, muon LR refine, activation + batch + sfree.
cd /jax_dir/notebooks
P=../data/aug2026/probe
run() {
  name=$1; shift
  echo "=== START $name $(date -u +%H:%M:%S) ===" >> $P/hp_sweep_wave2.log
  python analysis/train_probe_hp.py "$@" --out-prefix $P/hp_$name > $P/hp_$name.log 2>&1 \
    && echo "=== OK $name ===" >> $P/hp_sweep_wave2.log \
    || echo "=== FAIL $name ===" >> $P/hp_sweep_wave2.log
}
run muon_lr5e3_30k  --optimizer muon --schedule cosine --lr 0.005 --steps 30000
run cosine_lr3e4_s2 --schedule cosine --lr 3e-4 --seed 2
run muon_lr1e2      --optimizer muon --schedule cosine --lr 0.01 --steps 12000
run cosine_lr45e5   --schedule cosine --lr 4.5e-4
run sfree_lr1e3     --optimizer sfree --schedule none --lr 1e-3
run act_gelu        --activation gelu --schedule plateau --lr 3e-4
run bs1024_lr45e5   --batch-size 1024 --lr 4.5e-4 --schedule plateau
echo "=== WAVE2 COMPLETE ===" >> $P/hp_sweep_wave2.log
