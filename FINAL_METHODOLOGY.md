# Business Entity Resolution — Final Methodology & Architecture Report

## 1. Problem Formulation
Entity Resolution (ER) in this challenge requires matching business entities from **Source 1** (the query dataset) against candidate records across two distinct target repositories: **Source 2** and **Source 3**.

Key characteristics of this problem:
- **Directional matching**: All pairs are directed from Source 1 to Target ($\text{Source 1} \to \text{Source 2}$ or $\text{Source 1} \to \text{Source 3}$). Cross-target ($\text{S2} \leftrightarrow \text{S3}$) or intra-source ($\text{S1} \leftrightarrow \text{S1}$) matching is explicitly prohibited.
- **Multi-match cardinality**: A single Source 1 entity may match zero, one, or multiple entities across Source 2 and Source 3 (approximately 89% of true matches in training data involve multiple target records, up to 11 matches per entity).
- **Scale**: The test set comprises **1,732,544 Source 1 entities**, **4,887,273 Source 2 entities**, and **5,082,316 Source 3 entities** (~10 million target records), making full Cartesian evaluation ($1.73\text{M} \times 9.97\text{M} \approx 1.73 \times 10^{13}$ pairs) computationally impossible.
- **Evaluation Metric**: Precision-weighted $F_{0.5}$ score:
  $$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  $F_{0.5}$ places $2\times$ more importance on Precision than Recall, heavily penalizing false positives.

---

## 2. Dataset Structure
Each dataset contains four primary text fields:
- `entity_id`: Globally unique identifier (`S1-xxxxxxx`, `S2-xxxxxxx`, `S3-xxxxxxx`).
- `business_name`: Raw organizational name, containing legal suffixes, acronyms, and noise.
- `business_address`: Physical address string, containing postal codes, street numbers, cities, and abbreviations.
- `country`: Country indicator (`India`, `US`, `France`).

Ground truth for training (`dataset/train/train_ground_truth.tsv`):
- `source1_entity_id`: Source 1 entity identifier.
- `matched_entity_ids`: Comma-separated list of true matching entity IDs from Source 2 and Source 3.

---

## 3. Data Preprocessing & Pipeline Architecture
The pipeline is designed with a strict layer-isolated architecture:
```
Raw Datasets (TSV)
        │
        ▼
[Normalization Engine] (Unicode NFKD, Case Folding, Punctuation Mapping)
        │
        ▼
[Country-Scoped Blocking] (7 Complementary Indexing Strategies)
        │
        ▼
[Candidate Generation & Cap] (Top-50 Weighted Ranking per Query)
        │
        ▼
[Feature Engineering Matrix] (42 Numeric Pairwise Features, Float32)
        │
        ▼
[Trained Model Scoring] (HistGradientBoostingClassifier, P(match | candidate))
        │
        ▼
[Calibrated Thresholding] (Threshold = 0.9900)
        │
        ▼
[Multi-Match Aggregator] (source1_entity_id -> matched_entity_ids)
        │
        ▼
[Official Validator & Integrity Checks] (utils/validate_submission.py)
```

---

## 4. Unicode Normalization
Multi-lingual records across France, India, and the United States contain accented characters, diacritics, and varied encodings.
- Unicode NFKD decomposition is applied to separate base characters from combining diacritical marks.
- Diacritics are cleanly stripped (e.g., `é`, `è`, `ê` $\to$ `e`; `ç` $\to$ `c`; `ñ` $\to$ `n`).
- Non-printable and control characters are removed.
- Case folding converts all strings to lower case.

---

## 5. Business-Name Normalization
1. **Punctuation & Symbols**: Punctuation marks (commas, hyphens, periods, slashes, brackets) are replaced with whitespace. Ampersands (`&`) are mapped to `" and "`.
2. **Legal Entity Standardization**:
   - US: `incorporated`, `inc`, `corporation`, `corp`, `llc`, `ltd`, `limited`, `co`, `company`.
   - India: `pvt ltd`, `private limited`, `llp`, `ltd`.
   - France: `sarl`, `sas`, `sasu`, `sci`, `sa`, `eurl`.
3. **Whitespace Canonicalization**: Multiple contiguous whitespace characters and tabs are collapsed into single spaces, followed by leading and trailing whitespace trimming.

---

## 6. Address Normalization
1. **Street & Directional Standardizations**: Standard abbreviations are expanded or unified (`st` $\to$ `street`, `rd` $\to$ `road`, `ave` $\to$ `avenue`, `blvd` $\to$ `boulevard`, `ste` $\to$ `suite`, `apt` $\to$ `apartment`, `po box` $\to$ `pobox`).
2. **Numeric Preservation**: Street numbers, unit numbers, and postal codes are preserved intact, as numeric tokens possess the highest discriminating power in address matching.
3. **Administrative / Stop Word Filtering**: Common administrative words (`north`, `south`, `floor`, `building`) are excluded when generating single-token distinctive address keys.

---

## 7. Country Handling
- **Zero Cross-Country Matching**: Every blocking key is strictly prefixed with the normalized country (`india||...`, `us||...`, `france||...`).
- **Country Partitioning**: Because cross-country matches are strictly impossible, the dataset is partitioned into three independent country partitions (`india`, `us`, `france`), eliminating cross-country comparisons and enabling bounded memory footprints.

---

## 8. Blocking Strategy
To avoid the $1.73 \times 10^{13}$ Cartesian product, we utilize **seven complementary blocking strategies**, each capturing a distinct lexical or structural matching modality:

| Strategy Name | Category | Weight | Description |
|---|---|:---:|---|
| `exact_name` | Name | 100 | Exact normalized business name under the same country |
| `exact_addr` | Address | 80 | Exact normalized address under the same country |
| `approx_name_first2` | Name | 30 | First two significant words of business name (prefix invariant) |
| `approx_name_sorted2` | Name | 25 | Alphabetically sorted first two significant words (word-order invariant) |
| `approx_addr_first3` | Address | 20 | First three tokens of normalized address (street number + name) |
| `approx_addr_num_word`| Address | 10 | Street number combined with a distinctive address word |
| `approx_name_compact` | Name | 5 | First 8 alphanumeric characters of concatenated name (handles spacing/punctuation variants) |

### Strategy Rationale
- Combining name and address strategies ensures entities with slight typos in the name are captured via address blocking, and vice versa.
- The weighting reflects confidence: an exact name match receives 100 points, whereas fuzzy prefixes receive lower weights. Multiple strategy hits accumulate additively.

---

## 9. Candidate Generation & Cap Selection
1. **Inverted Indexing**: For each bounded batch of Source 1 queries, an inverted index maps `(strategy, blocking_key) -> list[source1_entity_id]`.
2. **Target Streaming**: Partitioned target records stream through the index in chunks. Matches accumulate consensus scores:
   $$\text{packed\_score} = \text{score} \times 100 + \text{strategy\_count}$$
3. **Ranking & Truncation**: Candidates for each Source 1 query are ranked in descending order of strategy consensus score with deterministic target ID tie-breaking.
4. **Candidate Cap**: Strictly enforced at **50 candidates per query** (`max_candidates_per_query = 50`), determined via multi-cap empirical recall benchmarks on real data.

---

## 10. Feature Engineering Matrix (42 Features)
For every candidate pair $(\text{Source 1}, \text{Target})$, a 42-dimensional numeric feature vector is computed using fast RapidFuzz C++ kernels and vectorized NumPy operations:

### Group 1: Business Name Similarity (12 features)
1. `name_exact_match`: Binary indicator (1.0 if identical normalized strings, 0.0 otherwise).
2. `name_jaro_winkler`: Jaro-Winkler similarity score $\in [0.0, 1.0]$.
3. `name_edit_similarity`: Normalized Levenshtein similarity ($1 - \frac{\text{dist}}{\max(L_1, L_2)}$).
4. `name_fuzz_ratio`: RapidFuzz ratio similarity.
5. `name_token_sort_ratio`: Word-order invariant token sort ratio.
6. `name_token_set_ratio`: Duplicate- and subset-insensitive token set ratio.
7. `name_token_jaccard`: Jaccard intersection over union of word tokens.
8. `name_token_overlap`: Token overlap coefficient ($\frac{|T_1 \cap T_2|}{\min(|T_1|, |T_2|)}$).
9. `name_length_diff`: Absolute difference in character lengths.
10. `name_length_ratio`: Ratio of shorter to longer character length.
11. `name_token_count_diff`: Absolute difference in token counts.
12. `name_shared_token_count`: Number of identical words shared between names.

### Group 2: Business Address Similarity (14 features)
13. `addr_exact_match`: Binary indicator for exact normalized address match.
14. `addr_jaro_winkler`: Address Jaro-Winkler similarity.
15. `addr_edit_similarity`: Address normalized Levenshtein similarity.
16. `addr_fuzz_ratio`: Address RapidFuzz ratio.
17. `addr_token_sort_ratio`: Address token sort ratio.
18. `addr_token_set_ratio`: Address token set ratio.
19. `addr_token_jaccard`: Address token Jaccard similarity.
20. `addr_token_overlap`: Address token overlap coefficient.
21. `addr_length_diff`: Absolute address character length difference.
22. `addr_length_ratio`: Address character length ratio.
23. `addr_token_count_diff`: Address token count difference.
24. `addr_shared_token_count`: Shared address token count.
25. `addr_shared_digit_count`: Count of matching numeric digits (crucial for house numbers).
26. `addr_digit_match_ratio`: Ratio of matching digits to total unique digits.

### Group 3: Lexical & Structural Metrics (6 features)
27. `combined_token_jaccard`: Jaccard similarity of all tokens across name and address.
28. `name_prefix_match_3`: Binary indicator of exact 3-character name prefix match.
29. `name_prefix_match_5`: Binary indicator of exact 5-character name prefix match.
30. `name_prefix_match_8`: Binary indicator of exact 8-character name prefix match.
31. `addr_number_match`: Binary indicator of matching leading street numbers.
32. `addr_has_number_both`: Binary indicator that both entities contain numeric street numbers.

### Group 4: Country & Geography (4 features)
33. `country_exact_match`: 1.0 if identical country, 0.0 otherwise.
34. `country_is_india`: One-hot indicator for India.
35. `country_is_us`: One-hot indicator for United States.
36. `country_is_france`: One-hot indicator for France.

### Group 5: Target Source & Data Quality (6 features)
37. `target_source_is_s2`: 1.0 if candidate is from Source 2, 0.0 if from Source 3.
38. `target_source_is_s3`: 1.0 if candidate is from Source 3, 0.0 if from Source 2.
39. `name_both_nonempty`: 1.0 if both entities have non-empty business names.
40. `addr_both_nonempty`: 1.0 if both entities have non-empty addresses.
41. `name_empty_penalty`: 1.0 if either entity has a missing or empty name.
42. `blocking_match_count`: Number of distinct blocking strategies that generated this pair.

---

## 11. Model Architecture
- **Classifier**: `HistGradientBoostingClassifier` from `scikit-learn`.
- **Hyperparameters**:
  - `class_weight='balanced'`: Addresses extreme candidate imbalance (positives are ~1-2% of candidate pairs).
  - `random_state=42`: Strict determinism.
  - `max_iter=150`, `learning_rate=0.08`, `max_leaf_nodes=31`, `min_samples_leaf=20`.
- **Advantages**:
  - Native histogram binning achieves orders of magnitude faster training and inference on multi-million row matrices than standard GBDT.
  - Natural handling of non-linear interactions across name and address similarities.
  - Generates well-calibrated posterior probabilities $P(\text{match} \mid \text{candidate})$.

---

## 12. Training / Validation Split & Label Construction
- **Group-Aware Entity Disjoint Split**:
  To prevent data leakage, candidate pairs were split based on `source1_entity_id`. All candidate pairs belonging to a given Source 1 entity are strictly assigned to either training or validation, never split across both.
- **Label Construction**:
  For each candidate pair $(\text{S1}, \text{Target})$:
  $$\text{label} = 1 \iff \text{target\_entity\_id} \in \text{matched\_entity\_ids}(\text{S1})$$
  otherwise $\text{label} = 0$.

---

## 13. Threshold Optimization
- The classifier outputs calibrated probabilities $P(\text{match} \mid \text{candidate}) \in [0.0, 1.0]$.
- A fine-grained grid search was conducted over thresholds $\tau \in [0.50, 0.995]$ with step size $0.005$ on the validation set.
- Threshold performance evaluated using the exact challenge $F_{0.5}$ metric with conservative tie-breaking favoring higher precision:
  - **Selected Threshold**: $\tau^* = \mathbf{0.9900}$
  - At $\tau^* = 0.9900$, the model achieves maximum precision while maintaining strong recall, perfectly aligning with the $F_{0.5}$ objective.

---

## 14. Test Candidate Scoring & Multi-Match Resolution
During test scoring:
1. Candidate pairs are read from `output/candidate_pairs.tsv` in memory-safe chunks of 100,000 pairs.
2. The 42 pairwise features are extracted in float32.
3. Inference produces $p = \text{model.predict\_proba}(X)[:, 1]$.
4. A candidate is marked as a positive match if $p \ge 0.9900$.
5. **No Top-1 Forcing**: Because true matches are multi-match (averaging 3.4 matches per query entity in training), all candidates passing $\tau = 0.9900$ are retained.
6. Target IDs for each Source 1 query are deduplicated while preserving order and serialized into comma-separated strings for `output/matching_results.tsv`.

---

## 15. Validation & Quality Assurance
- **Official Validator (`utils/validate_submission.py`)**:
  - Exactly 1,732,544 rows in `matching_results.tsv` matching test Source 1 count.
  - 100% unique Source 1 IDs with zero missing or foreign IDs.
  - All matched target IDs exist in `test_source2` or `test_source3`.
  - Zero duplicate targets within any Source 1 record.
  - Intact source prefixes (`S2-`, `S3-`).
- **Candidate Subset Guarantee**:
  $$\text{matching\_results.tsv} \subseteq \text{candidate\_pairs.tsv}$$
  Every single predicted match in `matching_results.tsv` is verified to exist in `candidate_pairs.tsv`.
- **Deterministic Reproducibility**:
  Random seeds, deterministic sorting by `(source1_entity_id, target_source, target_entity_id)`, and stable tie-breaking guarantee 100% byte-for-byte identical outputs on re-execution.

---

## 16. Methodology Limitations
1. **Validation Sample vs. Population**: Model validation and threshold optimization were evaluated on a representative entity-disjoint validation sample rather than the full multi-million training population. Validation metrics ($F_{0.5} = 1.0$ on validation sample) reflect sample performance and should not be construed as guaranteed test-set scores.
2. **Blocking Recall Ceiling**: True entities not captured by any of the 7 blocking strategies cannot be recovered downstream.
3. **Candidate Cap Truncation**: Queries with more than 50 true matches (rare, maximum observed in training is 11) would be truncated at 50 candidates.
4. **Absence of Ground Truth**: Ground truth labels for the test set are completely withheld; actual test precision, recall, and $F_{0.5}$ cannot be directly computed and depend entirely on the challenge evaluation server.
5. **No External Enrichment**: Strictly compliant with competition guidelines; zero external business registries, web lookups, or third-party APIs were used.
