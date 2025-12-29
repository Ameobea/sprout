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
