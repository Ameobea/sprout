1. Run `./dump-table.sh` to dump `mal-user-animelists` server-side (streamed through zstd, staged on ameo.dev, rsync'd down — resumable) into `../data/mal-user-animelists.tsv.zst`. ~13x compressed; benchmarked ~5x faster than the old chunked-query notebook (`download-animelists.ipynb`, now obsolete). The `--raw` TSV needs no unescaping: the JSON blobs contain no literal tabs/newlines.
2. Run `process-collected-profiles.ipynb` to convert it into individual rankings and filter it.  This also generates `all-anime-ids.json`.
   NOTE: input is now the .tsv.zst — read via `zstd -dc` subprocess stream (tab-split, not csv.reader) so the raw dump never hits disk uncompressed. The notebook still has the old CSV reader; adapt when running the next cycle.
3. `curl -X POST http://localhost:3080/populate-anime-metadata\?populateNulls\=true\&token\=asdf` to fill metadata table with placeholders (local dev server + MySQL tunnel)
4. `while true; do curl -X POST http://localhost:3080/populate-anime-metadata\?token\=asdf && echo "" && sleep 1.2; done` to fill in missing metadata from MAL API
5. Run `./dump-table.sh anime-metadata ../data/anime-metadata.tsv.zst` (replaces the old manual CSV export; small enough to just `zstd -d` in place)
6. Run `process-collected-metadata` script to convert metadata to `./work/data/processed-metadata.csv`.  This also generates `./work/data/corpus_ids.json`.
<!-- 7. Run `embedding_gen` script to build cooccurrence matrices and produce the `cooccurrence_matrix_wextra.npy` output file
1. Run `pymde.ipynb` to process the cooccurrence matrix and produce an embedding with PyMDE with desired parameters.  This will export the embedding to a .w2v file.
2. Run `emblaze` (in the browser if you want to use the embedding viz) to generate `data/projected_embedding.json` which is loaded by the webapp backend
3.  Start up the sveltekit app and if you're lucky it should be working with the updated embedding
 -->

Then, follow the steps in `README_training.md` to train the model

Then, generate embedding by running `extract_embeddings.py` from within the rocm container, and then `project_model_emnedding.ipynb` (outside container)
