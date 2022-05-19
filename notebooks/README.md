1. Download CSV exports from the DB for user anime profiles and move it into `./work/data/mal-user-animelists.csv`
2. Run `process-collected-profiles` to convert it into individual rankings and filter it.  This also generates `all-anime-ids.json`
3. Use `curl -X POST http://localhost:3080/populate-anime-metadata\?populateNulls\=true\&token\=asdf` to fill metadata table with placeholders
4. Use `while true; do curl -X POST http://localhost/populate-anime-metadata\?token\=j23k4sidjkl234785348348348348 && echo ""; done` to fill in missing metadata from MAL API
5. Download metadata table as CSV and move to `./work/data/anime-metadata.csv`
6. Run `process-collected-metadata` script to convert metadata to `./work/data/processed-metadata.csv`
7. Run `embedding_gen` script to build cooccurrence matrices and produce the `cooccurrence_matrix_wextra.npy` output file
8. Run `pymde.ipynb` to process the cooccurrence matrix and produce an embedding with PyMDE with desired parameters.  This will export the embedding to a .w2v file.
9. Run `emblaze` (in the browser if you want to use the embedding viz) to generate `data/projected_embedding.json` which is loaded by the webapp backend
10. Start up the sveltekit app and if you're lucky it should be working with the updated embedding

Now, we train the recommendation model.  The embedding is used as the corpus of anime to recommend from.  Currently, whenever the embedding is changed, the model needs to be re-trained from scratch since the indices of the model's output correspond to the indices of the embedding.

1. Clear out the processed metadata table from the `raw_animelists.db` sqlite database in the data directory because it stores indices that reference the old embedding
2. Pull up `http://localhost:3080/recommendation/training/train` and let it run until the model is downloaded at the end.
3. Rename the model + copy into the `data/tfj_models/your_model_name` directory
4. Update `ModelName` enum in the code and places the enum is used
