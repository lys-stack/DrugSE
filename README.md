

# DrugSE

## Environment Reproduce

- In order to get DrugSE, you need to clone this repo:

  ```
  https://github.com/lys-stack/DrugSE
  ```

## Files

* **Data** -- Data used in the training and testing process, including drugs, side effects, and drug-side effect frequencies.
* **class_model.py** -- Model file for predicting drug-side effect associations.
* **regress_model.py** -- Model file for predicting drug-side effect frequencies.
* **fusion.py** -- Code file for feature fusion using a similarity matrix.

## Datasets

* **664_drug_drug_scores.csv** -- Compound IDs for each drug were obtained, and compound-compound association scores were extracted from the STITCH database.

* **664_drug_fingerprint_jaccard_similarity_matrix_new.csv** -- SMILES strings were obtained from the STITCH database and subsequently converted into 2048-dimensional vectors using RDKit. The Jaccard coefficient was then applied to quantify the structural similarity between the compounds.

* **kvplm_Side_Effect_Similarity_Matrix.csv** -- A similarity matrix was derived from biomedical text data on side effects collected from Wikipedia and PubChem. To ensure no data leakage occurred, all descriptions involving drug-side effect relationships were filtered out.

* **semantic.csv** -- The semantic descriptors for each side effect were represented using a directed acyclic graph, following an existing measurement approach.

* **word_new.csv** -- Side effect terms were embedded into a 300-dimensional pre-trained vector space, where cosine similarity was utilized as the metric to assess semantic relatedness.

* **Drug-Side_Effect_Frequency664.csv** -- This file is the frequency matrix of drug side effects.

  

## Run for warm-start

- Warm-start file for classification status

  ```
  python class_warm_start.py
  ```

- Warm-start file for classification status

  ```
  python regress_warm_start.py
  ```

## Run for cold-start

- Cold-start file for classification status

  ```
  python class_cold_start.py
  ```

- Cold-start file for classification status

  ```
  python regress_cold_start.py
  ```
