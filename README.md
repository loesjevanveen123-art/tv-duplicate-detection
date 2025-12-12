# F1-Optimized MinHash–LSH + MSM Duplicate Detection Pipeline

This repository implements a fully automatic duplicate-detection pipeline for the **TVs-all-merged** dataset.  
The system combines **self-built MinHash**, **banded LSH**, and an **MSM similarity model**, followed by **F1-based tuning** and a **5× bootstrap evaluation**.

---

## 0. What This Project Is About

This project focuses on developing a **scalable solution** for product duplicate detection in heterogeneous Web shop data.  
Consumers rely on accurate product aggregation, yet product descriptions across online retailers often differ in terminology, formatting, attribute naming, and completeness. Because manual aggregation is infeasible, an automated approach is needed to determine when two product descriptions refer to the **same physical product**.

A major challenge is scalability: comparing all possible product pairs becomes computationally infeasible as dataset size grows.  
To address this, the assignment requires using **Locality-Sensitive Hashing (LSH)** to drastically reduce the number of comparisons while still retrieving the majority of true duplicates.

The dataset used here consists of **1,624 televisions** collected from four Web shops (Amazon, Newegg, BestBuy, and TheNerds). Each product contains a title, multiple key–value attributes, and a `modelID` used only for evaluation. Since identifiers like modelID are often missing in real-world settings, the algorithm must detect duplicates **without relying on them**.

The assignment emphasizes scalability over algorithmic complexity. Therefore, this project implements:

- a **hybrid model-word representation** derived from titles and structured numeric attributes,  
- a **self-built MinHash–LSH pipeline** to generate candidate pairs efficiently,  
- a **weighted MSM similarity model** for classifying duplicates,  
- evaluation using **pair quality**, **pair completeness**, **F1\***, and **F1**,  
- and a **bootstrap procedure (5×)** to obtain robust averaged metrics.

This pipeline aims to achieve strong F1 performance while comparing only a tiny fraction of all possible pairs.

---

## 0.1 Structure of the Code and How to Use the Code

The repository is organised into modular R scripts:

- `load_data.R` — data loading  
- `cleaning.R` — text cleaning and unit normalisation  
- `model_words.R` — extraction of hybrid model-word tokens  
- `vocabulary_index.R` — vocabulary building and token indexing  
- `minhash.R` — custom MinHash implementation  
- `lsh.R` — banded LSH candidate generation  
- `msm_similarity.R` — computation of MSM similarity scores  
- `evaluation.R` — F1, PQ, PC, and related metrics  
- `tuning.R` — grid search over LSH bands, MSM weights, and thresholds  
- `bootstrap.R` — five-fold bootstrap evaluation  
- `pipeline.R` — wrapper functions for running all stages

This modular structure allows each stage of the pipeline to be run independently or combined as needed.
Load the above scripts.
This repository implements a fully automatic duplicate-detection pipeline for the **TVs-all-merged** dataset.  
The system combines **self-built MinHash**, **banded LSH**, and an **MSM similarity model**, followed by **full F1-based tuning** and a **5× bootstrap evaluation**.

---

## 1. Data Loading

**Function:** `load_tv_data(path)`  
Loads the JSON file (`TVs-all-merged.json`) and extracts:

- `id` (internal index)
- `modelID` (gold duplicate label)
- `title` (raw product title)
- `features` (vendor feature maps)

Returns a tidy tibble with all products.

---

##  2. Cleaning and Normalisation

**Functions:**  
`normalize_units_string`, `clean_key`, `clean_tv_data`

Cleaning operations:

- Lowercasing and unit normalisation (`"`, *-inch*, *inches*, hz/hertz → unified forms)
- Removing punctuation
- Normalising feature keys
- Producing:
  - `title_clean`
  - `features_clean` (cleaned values)
  - `features_keys_clean` (cleaned keys)

This ensures all text follows consistent formatting.

---

##  3. Model-Word Extraction

Hybrid representation combining title tokens and feature-value patterns.

 Title-based model words
`extract_model_words_title(title)`:

- Lowercases and tokenises title
- Keeps tokens:
  - length ≥ 4
  - contain at least one letter AND one digit
- Removes generic TV words (e.g., *1080p*, *tv*, *led*, *inch*, *hdr*, …)

 Value-based model words
`extract_model_words_values(values)`:

- Extracts strong structured patterns:
  - resolutions (`1920x1080`, etc.)
  - Hz values (`100 hz`, `120 hz`, …)

These two sources are merged into a compact token set per product.

---

##  4. Vocabulary + Indexing

**Functions:**  
`build_vocabulary`, `build_indexed_sets`

- Builds global vocabulary of all unique model words.
- Converts each product’s token set into a vector of integer IDs.
- Required for fast MinHash computation.

---

##  5. MinHash Signatures

**Function:** `compute_minhash_signatures`

- Implements MinHash **from scratch**
- 100 independent hash functions
- For each product, computes the minimum hashed value over its token IDs
- Produces signature matrix `S`:
  - rows = hash functions
  - columns = products

---

##  6. Locality-Sensitive Hashing (LSH)

**Function:** `lsh_candidates(S, bands)`

- Splits signature matrix into `bands`
- Within each band, identical signature slices → same bucket
- All pairs in same bucket become candidate pairs
- Deduplicates pairs globally

Outputs a tibble of candidate pairs `(i, j)`.

---

##  7. Similarity Computation (MSM)

Combines three similarity sources:

1. **Key-value similarity (C_kv)**  
   - Matches cleaned keys across products  
   - Computes q-gram similarity between their cleaned values

2. **Feature value model-word similarity (C_hsm)**  
   - Jaccard of model words extracted from feature values

3. **Title model-word similarity (C_tmw)**  
   - Jaccard of model words extracted from titles

Final similarity:

MSM = w_kv * C_kv + w_hsm * C_hsm + w_tmw * C_tmw

**Function:** `predict_msm` assigns MSM scores to all LSH candidate pairs.

---

##  8. Evaluation (Final Classification)

**Function:** `evaluate_final`

Given predicted duplicate pairs (after thresholding MSM scores):

- Enumerates **all possible product pairs**
- Computes:
  - TP, FP, FN, TN
  - Precision, recall
  - F1 measure

Used to assess the end-to-end performance of the pipeline.

---

##  9. F1-Optimized Tuning

**Function:** `tune_all(df)`

Performs a full grid search over:

- **LSH bands**: `{5, 10, 20, 25, 50}`
- **MSM weights**:
  - `w_kv ∈ {0.6, 0.7, 0.8, 0.9}`
  - `w_hsm ∈ {0.05, 0.1, 0.2}`
  - `w_tmw = 1 − w_kv − w_hsm`
- **Thresholds**: `0.05 ... 0.95`

For each combination:

1. Run LSH  
2. Compute MSM scores  
3. Threshold scores  
4. Evaluate F1  
5. Keep the highest-F1 configuration

Output includes:

- best band count
- best MSM threshold
- best weights
- resulting precision, recall, F1

---

##  10. Full Pipeline Run

**Function:** `run_pipeline_full(path)`

- Loads and cleans entire dataset
- Runs full tuning
- Returns the **globally best configuration**

---

##  11. Bootstrap Evaluation (5×)

**Function:** `bootstrap_pipeline`

For each bootstrap iteration:

1. Draw sample with replacement  
2. Use out-of-bag items as test set  
3. Recompute:
   - model words  
   - MinHash  
   - LSH candidates  
   - MSM scores  
4. Apply tuned threshold  
5. Evaluate precision, recall, F1  

Outputs:

- results per iteration
- bootstrap means:
  - `F1_mean`
  - `precision_mean`
  - `recall_mean`

---

##  12. LSH-Stage Metrics (Blocking Quality)

**Function:** `evaluate_lsh_stage`

Computes:

- `PQ` = duplicate density among candidate pairs  
- `PC` = fraction of true duplicates retrieved  
- `F1*` = harmonic mean of PQ and PC  
- `Nc` = number of candidate pairs  
- `Dn` = true duplicate count  
- `Df` = true duplicates retrieved by LSH  

Used to measure blocking (LSH) performance *before* MSM classification.

---

##  13. Best-Config Re-Evaluation

**Function:** `recompute_for_best`

Recomputes everything for the best configuration:

- LSH PQ / PC / F1*
- Final precision / recall / F1

Used to verify the tuned performance.

---

##  14. F1 vs. Fraction of Comparisons Curve

**Function:** `lsh_f1_curve`

- Fixes MSM weights and MSM threshold
- Varies only **bands**:
  - `{2, 4, 5, 10, 20, 25, 50, 100}`
- For each band count:
  - Runs LSH
  - Computes fraction of comparisons
  - Computes F1
- Returns tibble with columns:
  - `bands`
  - `fraction`
  - `F1`

A ggplot is provided to generate a scientific-style curve.

## 15. LSH Blocking Efficiency Curve

**Function:** `lsh_stage_curve`

- Varies only the **LSH band configuration**
- For each band count:
  - Runs LSH blocking
  - Computes:
    - **Pair Quality (PQ)**
    - **Pair Completeness (PC)**
    - **F1\*** (blocking efficiency)
    - **Fraction of comparisons**
- Returns a tibble with:
  - `bands`
  - `fraction`
  - `PQ`
  - `PC`
  - `F1_star`

Several ggplots are provided to visualise how blocking effectiveness changes with the number of retained comparisons.

