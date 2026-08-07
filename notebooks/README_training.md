First have to pull + process raw profile data, following guide in README.md through step 7.

Then, run the `vectorize_training_data.ipynb` notebook.  This filters + converts the training data into sparse vectors that can be easily loaded during training.

Now, we can train the model.

 * Run `just launch-jax` to spin up a Docker container for JAX which will give GPU acceleration with AMD GPU
 * Run `docker attach rocm_jax` to get a shell into that container
 * Run `apt install libdw1` to install a missing dep inside the container
 * Install Flax inside the container by running this:

```
cat > constraints.txt <<'EOF'
jax==0.7.1
jaxlib==0.7.1
jax-rocm7-pjrt==0.7.1
jax-rocm7-plugin==0.7.1
EOF

pip install --no-cache-dir -c constraints.txt "flax<0.12"
```

This ensures that the flax version doesn't pull in a new jax version and break the environment.

Inside the attached container, we run `cd notebooks && python train.py`

This will run and spit out weights into the data directory.

The model can then be tested using the `infer.py` script.

## Training recipe update (2026-08-06, HP sweep — see artifact e3c017b5)

- **Schedule**: cosine (warmup 1000, decay to end, end_value lr/100) replaces
  reduce_on_plateau — confirmed better on 2 seeds, free.
- **Production retrains**: muon (optax.contrib.muon) lr 5e-3@bs512-scale
  (7e-3@bs1024, 1e-2@bs2048) + cosine — beats the adam ceiling on BOTH heads
  (2-seed confirmed; 50k run: trusted MAE 0.4438 vs 0.4507 control). Cost:
  NS GEMMs saturate the GPU (~10x wall-clock at bs512; bs1024+ amortizes to
  ~3x, ~2h full run). Run overnight/idle only — it starves the desktop.
  Everyday experimentation stays adam 3e-4 + cosine (22 min).
- LR ceilings (adam): 6e-4 hurts, 1e-3 diverges. lion/ademamix/schedule-free
  lose at drop-in LRs; gelu ~ swish; bs1024 quality-neutral (halves steps).
