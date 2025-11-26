# tv-duplicate-detection

This project implements an end-to-end pipeline to detect duplicate TV products across different shops using:

- **Text normalization & tokenization**
- **MinHash signatures**
- **Locality-Sensitive Hashing (LSH)**
- **MSM similarity** (a model that combines several similarity components)
- **Bootstrapped evaluation & LSH parameter tuning**
- **Visualization of performance–efficiency trade-offs**

The input is a JSON file with TV product descriptions from multiple shops. The output is a set of metrics, tuned LSH parameters, and plots showing how well the blocking strategy (MinHash + LSH) works.

## 1. Requirements & Setup

### Software

- **R** (version 4.x or later recommended)
- R packages:
  - `jsonlite`
  - `dplyr`
  - `purrr`
  - `stringr`
  - `tibble`
  - `tidyr`
  - `digest`
  - `ggplot2`

You can install the packages with:

### Data

The script assumes a JSON file with merged TV product records.
Each element in the JSON corresponds to one product record, with (at least) the fields:

* `shop`
* `modelID`
* `title`
* `featuresMap` (nested key–value structure of attributes)

> **Important:** Update `json_path` to the correct location of your JSON file before running the code.

---

## 2. High-level Overview

The pipeline consists of the following steps:

1. **Load and flatten the TV product data** into a single data frame `tv_df`.
2. **Preprocess text** (title + features) and create token representations suitable for MinHash and LSH.
3. **Build auxiliary structures** for MSM similarity:

   * key–value maps
   * model words from titles and attribute values
4. **Define MSM similarity function** combining:

   * attribute value q-gram similarity,
   * value model word overlap,
   * title model word overlap.
5. **Generate MinHash signatures** for all products.
6. **Apply LSH** to generate candidate duplicate pairs.
7. **Define gold labels** (true duplicates) based on `modelID`.
8. **Compute evaluation metrics** for a test subset.
9. **Bootstrap evaluation** for robustness.
10. **Tune LSH parameters** over a grid of configurations.
11. **Plot trade-offs** (e.g. F1*, pair quality, completeness vs. fraction of comparisons).

---

## 3. Code Structure

The code is organized in numbered sections:

### 3.1. Libraries and Helper Operator

* Loads all required libraries.
* Defines `%||%` (a simple null coalescing operator: returns `a` if not `NULL`, otherwise `b`).

### 3.2. Data Loading & Preprocessing

Key objects and steps:

* `json_path`: path to the TV JSON file.
* `tv_json <- fromJSON(json_path, simplifyVector = FALSE)`: read data as nested list.
* `tv_list <- purrr::flatten(tv_json)`: each element is one product record.
* `tv_df`: a flat tibble with columns:

  * `shop`
  * `modelID`
  * `title`
  * `features_text` (flattened values from `featuresMap`)
  * `id` (row index)
* Text preprocessing:

  * `preprocess_text()` normalizes text:

    * lowercasing
    * removing non-alphanumeric characters
    * condensing whitespace
  * `full_text_raw = title + features_text`
  * `full_text = preprocess_text(full_text_raw)`

### 3.3. Product Representation: Tokens

* `extract_tokens(txt)`:

  * Normalizes text.
  * Splits into words.
  * Builds **bigrams** (`"word1_word2"`) from consecutive words.
  * Returns **unique bigrams** as token set for each product.
* `tv_tokens`: list of tokens per product.
* `vocab`: global mapping from token string to token ID.
* `tv_token_ids`: for each product, a set of token IDs.

These token-ID sets are the basis for MinHash signatures.

### 3.4. Extra Structures for MSM

* `kv_list`: list of key–value maps per product derived from `featuresMap`.
  Values are concatenated strings for each key.

Utility functions:

* `jaccard_set(a, b)`: Jaccard similarity between two sets.
* `qgram_tokens(x, q = 3)`: extract q-gram characters from a string (default `q = 3`).

Model words:

* `extract_title_model_words(s)`:

  * Finds tokens in the **title** that contain both letters and digits (e.g. model codes like `"55uh615v"`).
* `extract_value_model_words(s)`:

  * Extracts **numeric** or **numeric+unit** tokens from attribute values.
  * Strips units so `"55inch"` and `"55in"` both give `"55"` etc.
* Precomputed lists:

  * `title_mw`: model words from product titles.
  * `value_mw`: model words from feature values.

### 3.5. MSM Similarity

`msm_sim(id1, id2, w1 = 1/3, w2 = 1/3, w3 = 1/3, q = 3)` computes the combined MSM similarity between two products:

1. **`s_kv_match`**
   Average Jaccard similarity on **q-grams** of attribute values, restricted to attributes (keys) present in **both** products.

2. **`s_kv_model`**
   Jaccard similarity between **model words extracted from values** (`value_mw`).

3. **`s_title`**
   Jaccard similarity between **model words extracted from titles** (`title_mw`).

The final MSM similarity is:

```text
MSM = w1 * s_kv_match + w2 * s_kv_model + w3 * s_title
```

By default, all three components have equal weight.

### 3.6. MinHash Implementation

* `generate_minhash_params(n_tokens, n_hash = 100, seed = 42)`:

  * Finds a prime `P` just above `2 * n_tokens + 1`.
  * Samples random hash parameters `a` and `b`.
  * Returns a list `{a, b, P}`.

* `minhash_signature(token_id_list, a, b, P)`:

  * Builds an `n_hash x n_items` signature matrix.
  * For each item, computes MinHash values over its token IDs.
  * Each column = MinHash signature of one product.

Global parameters:

```r
N_HASH <- 100
mh_params <- generate_minhash_params(length(vocab), n_hash = N_HASH, seed = 123)
```

### 3.7. LSH (Banding) & Theory Functions

Theory:

* `lsh_prob_same_bucket(s, b, r)`:

  * Probability that two items with Jaccard similarity `s` land in at least one common bucket for an LSH setup with:

    * `b` bands
    * `r` rows per band

* `lsh_curve(b, r, s_vals = seq(0, 1, by = 0.01))`:

  * Builds a tibble for plotting `P(same bucket)` vs `s`.

Banding & candidate generation:

* `lsh_candidate_pairs(sig_mat, bands, rows_per_band)`:

  * Splits the signature matrix into `bands` blocks of `rows_per_band` rows.
  * For each band:

    * Hashes signatures into buckets using `digest` (`xxhash64`).
    * Finds items that share the same bucket in that band.
    * Generates unique candidate pairs `(i, j)` across all bands.

### 3.8. Gold Labels (Duplicates)

* `is_duplicate_pair_vec(i_vec, j_vec, model_ids)`:

  * Returns a logical vector indicating if each pair `(i, j)` is a **true duplicate**:

    * `modelID[i] == modelID[j]`
    * both non-empty and non-`NA`.

This acts as the **ground truth** for evaluation.

### 3.9. Metrics for One Test Subset

* `compute_metrics_for_test(...)`:

  * Inputs:

    * `test_idx`: indices of test products (global indices).
    * `cand_pairs_local`: candidate pairs in local indices (`i_local`, `j_local`) + `pred_dup` flags.
    * `modelID_global`: full vector of `modelID`s.
    * `test_to_global`: mapping from local test indices to global ones.
  * Steps:

    * Map candidate pairs to global IDs.
    * Define **universe of all pairs** within the test set (`combn(test_idx, 2)`).
    * Determine:

      * **True positives (TP)**, **false positives (FP)**,
      * **false negatives (FN)**, **true negatives (TN)**.
    * Compute:

      * Precision, recall, F1.
      * **Pair quality**: TP / (# candidate pairs).
      * **Pair completeness**: TP / (total true duplicate pairs).
      * **F1***: harmonic mean of pair quality & pair completeness.
      * **Comparison fraction**: (# candidate pairs) / (# all pairs in test set).

### 3.10. Bootstrapping (MSM)

* `run_bootstrap(...)`:

  * Inputs include:

    * `tv_df`, `tv_token_ids`
    * LSH config: `n_hash`, `bands`, `rows_per_band`
    * MSM threshold `sim_threshold`
    * number of bootstrap iterations `B`
    * `mh_params` for hashing
  * For each bootstrap iteration:

    1. Draw a bootstrap sample of indices (with replacement).
    2. Define train = unique sampled indices; test = remaining indices.
    3. Subset token IDs for the test set.
    4. Compute MinHash signatures on test subset.
    5. Run LSH to get candidate pairs (local).
    6. Compute MSM similarity for each candidate and flag `pred_dup` using `sim_threshold`.
    7. Call `compute_metrics_for_test()` to get metrics.
  * Returns a tibble of metrics for each bootstrap (`bootstrap` column marks the iteration).

### 3.11. LSH Tuning

* `tune_lsh(...)`:

  * Defines a grid of configurations:

    * `n_hash_values` (e.g. `c(100)`)
    * `bands_values` (e.g. `c(10, 20, 25, 50)`)
    * `thresholds` for MSM similarity (e.g. `c(0.1, 0.2, 0.3)`)
  * Keeps only configs where `rows_per_band = n_hash / bands` is an integer.
  * For each config:

    * Runs `run_bootstrap()` with `B` iterations.
    * Aggregates metrics:

      * average F1
      * average F1*
      * average pair quality
      * average pair completeness
      * average comparison fraction
  * Returns `tuning_results`, a tibble with metrics per configuration.

### 3.12. Final Evaluation & Plots

Main calls:

```r
tuning_results <- tune_lsh(...)
best_cfg <- tuning_results %>%
  arrange(desc(avg_F1_star)) %>%
  slice(1)

n_hash_best      <- best_cfg$n_hash
bands_best       <- best_cfg$bands
rows_per_band_bt <- best_cfg$rows_per_band
sim_thr_best     <- best_cfg$sim_threshold

bootstrap_results <- run_bootstrap(
  tv_df         = tv_df,
  tv_token_ids  = tv_token_ids,
  n_hash        = n_hash_best,
  bands         = bands_best,
  rows_per_band = rows_per_band_bt,
  sim_threshold = sim_thr_best,
  B             = 5,
  mh_params     = mh_params
)
```

* `tuning_results`: metrics across all tested LSH configurations.
* `best_cfg`: chosen configuration with highest `avg_F1_star`.
* `bootstrap_results`: metrics for the selected configuration across `B = 5` bootstraps.
* Summary of final performance:

```r
bootstrap_results %>%
  summarise(
    avg_F1                  = mean(F1),
    avg_F1_star             = mean(F1_star),
    avg_pair_quality        = mean(pair_quality),
    avg_pair_completeness   = mean(pair_completeness),
    avg_comparison_fraction = mean(comparison_fraction)
  )
```

#### Plots

Several ggplot visualizations are produced:

1. **Bootstrap variability plots**:

   * Boxplots + jitter for F1, F1*, pair quality, pair completeness, comparison fraction across bootstraps.
2. **Curves**:

   * Pair completeness vs fraction of comparisons.
   * Pair quality vs fraction of comparisons.
   * F1* vs fraction of comparisons.
   * F1 vs fraction of comparisons.

---

## 4. How to Run the Code

1. **Open R or RStudio.**

2. **Load the script** containing the code (e.g. `source("your_script.R")`), or paste the code into an R script file and run it.

3. **Check / update the data path:**

   ```r
   json_path <- "/path/to/your/TVs-all-merged.json"
   ```

4. **Run the full script** from top to bottom. This will:

   * Load and preprocess the data.
   * Build MinHash and LSH structures.
   * Define MSM similarity.
   * Perform LSH tuning (`tuning_results`).
   * Choose the best configuration (`best_cfg`).
   * Run bootstrap evaluation with best parameters (`bootstrap_results`).
   * Generate various ggplot visualizations.

5. **Inspect the outputs:**

   * `tuning_results`:

     * Compare configurations by `avg_F1_star`, `avg_pair_quality`, `avg_pair_completeness`, and `avg_comparison_fraction`.
   * `best_cfg`:

     * See the chosen `n_hash`, `bands`, `rows_per_band`, and `sim_threshold`.
   * `bootstrap_results`:

     * Evaluate stability of performance metrics across bootstrap samples.
   * Plots:

     * Visualize the trade-offs between blocking effectiveness and computational cost.

---

## 5. Customizing the Experiment

You can easily adjust aspects of the pipeline:

* **Change MSM threshold**:

  ```r
  thresholds = c(0.1, 0.2, 0.3, 0.4)
  ```

* **Explore other LSH configs**:

  ```r
  n_hash_values = c(50, 100, 200)
  bands_values  = c(10, 20, 25, 50)
  ```

* **Increase bootstrap iterations** for more stable estimates:

  ```r
  B = 10
  ```

* **Adjust MSM weights** in `msm_sim()` if you want to emphasize certain components:

  ```r
  msm_sim(id1, id2, w1 = 0.4, w2 = 0.4, w3 = 0.2)
  ```

---

## 6. Conceptual Summary

* The project tackles **duplicate detection / entity matching** for TV products.
* Uses a **blocking strategy** (MinHash + LSH) on normalized text to avoid comparing all `O(N²)` pairs.
* Applies **MSM similarity** to candidate pairs to decide whether they are duplicates.
* Evaluates performance with **pair quality**, **pair completeness**, and **F1***, measuring both effectiveness and efficiency.
* Uses **bootstrapping** and **grid search** over LSH parameters to find robust, high-performing configurations.

This README should give enough context to understand what the code does, how it is structured, and how to run and extend it.

```
```
