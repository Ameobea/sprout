1. Run `download-animelists.ipynb` to download the collected user anime profiles from the DB and write them into `./work/data/mal-user-animelists.csv`
2. Run `process-collected-profiles.ipynb` to convert it into individual rankings and filter it.  This also generates `all-anime-ids.json`
3. You'll probably want to delete or compress the `data/mal-user-animelists.csv` at this point.  It wil be enormous and isn't needed after this point.
4. Use `curl -X POST http://localhost:3080/populate-anime-metadata\?populateNulls\=true\&token\=asdf` to fill metadata table with placeholders
5. Use `while true; do curl -X POST http://localhost:3080/populate-anime-metadata\?token\=asdf && echo "" && sleep 1.2; done` to fill in missing metadata from MAL API
6. Download metadata table as CSV and move to `./work/data/anime-metadata.csv`. Include header row.
7. Run `process-collected-metadata` script to convert metadata to `./work/data/processed-metadata.csv`.  This also generates `./work/data/corpus_ids.json`.
<!-- 7. Run `embedding_gen` script to build cooccurrence matrices and produce the `cooccurrence_matrix_wextra.npy` output file
1. Run `pymde.ipynb` to process the cooccurrence matrix and produce an embedding with PyMDE with desired parameters.  This will export the embedding to a .w2v file.
2. Run `emblaze` (in the browser if you want to use the embedding viz) to generate `data/projected_embedding.json` which is loaded by the webapp backend
3.  Start up the sveltekit app and if you're lucky it should be working with the updated embedding
 -->

Then, follow the steps in `README_training.md` to train the model

Then, generate embedding by running `extract_embeddings.py` from within the rocm container, and then `project_model_emnedding.ipynb` (outside container)
