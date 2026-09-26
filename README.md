# SCOPE: supplementary results

This repository holds the supplementary results of the paper on SCOPE, a set-completion head fused with a closed-form item-item base for multimodal recommendation. It contains this document and nothing else: no code, no data, no features. Every number below comes from a completed run of the study.

Contents

- [Conventions](#conventions)
- [A. Amazon-Electronics (63K items)](#a-amazon-electronics-63k-items)
- [B. New users with truncated context, and items without interactions](#b-new-users-with-truncated-context-and-items-without-interactions)
- [C. Closed-form content kernels and the head on the strongest published kernel](#c-closed-form-content-kernels-and-the-head-on-the-strongest-published-kernel)
- [D. Matched neighbour comparison in full](#d-matched-neighbour-comparison-in-full)
- [E. Coverage, breadth, user-activity strata and popularity concentration](#e-coverage-breadth-user-activity-strata-and-popularity-concentration)
- [F. Further analyses](#f-further-analyses)
- [G. Published versus reproduced baselines, and baseline configurations](#g-published-versus-reproduced-baselines-and-baseline-configurations)
- [H. Feature provenance](#h-feature-provenance)
- [I. Implementation details](#i-implementation-details)

## Conventions

- Metrics are test Recall@K (R@K) and NDCG@K (N@K), K in {10, 20}, computed by ranking the full catalog for every user with the user's training items masked. Values are printed with four decimals, rounded half-up from the stored value.
- The split is the 8:1:1 random split of the MMRec release; the datasets are Amazon Baby, Sports, Clothing and Electronics and MicroLens.
- Unless a table says otherwise, single-seed values use seed 2024 and three-seed values use seeds 2024, 2025 and 2026; "pstd" is the population standard deviation over the three seeds.
- "Base" is the closed-form base of the paper: one-hop EASE on the interaction Gram matrix plus a text-kNN affinity term, with its ridge weight and text weight selected on validation. "Set view" or "head" is the masked set-completion head. SCOPE-v1 fuses the base and the head; SCOPE-G adds one propagation step over a frozen item graph; SCOPE-v2 and SCOPE-U compose the base and the head with the scores of FREEDOM or GUME. On Electronics the base is a sparse co-occurrence proxy (Section A).
- Fusion and composition weights are selected by grid search on validation R@20; the grids are listed in Section I.
- Paired tests are user-level paired bootstraps with B = 10,000 resamples, two-sided; a p of 0 in a table means that no resample crossed zero (p < 2 x 10^-4). Holm adjustment is applied within the family named in each table. "n.s." means not significant at 0.05 after Holm. MDE is the minimum detectable effect at 80% power.
- Relative changes are 100 (a - b) / b on the four-decimal values, printed with one decimal.

## A. Amazon-Electronics (63K items)

This section reports the Electronics results, which the paper does not include, on the sparse proxy base that replaces the dense closed form on this catalog.

Dataset: 192,403 users, 63,001 items, 1,689,188 interactions, training density 0.0103%. A dense 63,001 x 63,001 item-item matrix takes 15.9 GB in 32-bit floats, so the dense closed form of the paper is not fitted here.

Settings specific to Electronics:

| Setting | Value |
|---|---|
| Base | sparse co-occurrence kNN proxy, k = 100, no text term |
| SCOPE-G propagation | K = 3 steps over the co-occurrence graph only (no text graph) |
| SCOPE-U views and weights (seed 2024) | FREEDOM 2.0, LGMRec 3.0, set view 1.0, co-occurrence base 0.5 |
| SCOPE-v2 | not instantiated on Electronics |
| Set-view scoring in SCOPE-v1 and SCOPE-U | L1-normalized logits (a variant of the cosine head of the paper) |
| SCOPE-G training | same L1-normalized scoring and a down-scaled text seed of the item table; no saved checkpoint; reported as run |

Disclosure. The set view on Electronics was scored with an L1-normalized variant of the cosine head, not with the cosine score used on the other four datasets, and SCOPE-G on Electronics uses K = 3 propagation steps over a co-occurrence-only graph instead of one step over the mixed co-occurrence and text graph. The Electronics rows are therefore not the same scorer as the SCOPE rows of the paper.

Table A1. Electronics, seed 2024. Learned baselines that were not run on Electronics (EASE, ADMM-SLIM, GRCN, DA-MRS, DRAGON, COHESION, SMORE, GUME) and the dense content closed forms of Section C are omitted.

| Method | R@10 | N@10 | R@20 | N@20 |
|---|---|---|---|---|
| LightGCN | 0.0358 | 0.0201 | 0.0533 | 0.0246 |
| MMGCN | 0.0216 | 0.0115 | 0.0339 | 0.0147 |
| VBPR | 0.0160 | 0.0080 | 0.0263 | 0.0106 |
| BM3 | 0.0375 | 0.0214 | 0.0543 | 0.0258 |
| MGCN | 0.0342 | 0.0192 | 0.0510 | 0.0235 |
| LATTICE | 0.0390 | 0.0218 | 0.0578 | 0.0267 |
| MENTOR | 0.0390 | 0.0216 | 0.0582 | 0.0266 |
| FREEDOM | 0.0381 | 0.0209 | 0.0588 | 0.0263 |
| LGMRec | 0.0394 | 0.0218 | 0.0597 | 0.0270 |
| DiffMM | 0.0314 | 0.0179 | 0.0463 | 0.0218 |
| LLMRec | 0.0320 | 0.0179 | 0.0481 | 0.0221 |
| RLMRec | 0.0381 | 0.0210 | 0.0568 | 0.0259 |
| SCOPE-v1 (proxy base + set view) | 0.0439 | 0.0256 | 0.0634 | 0.0307 |
| SCOPE-G (K = 3, co-occurrence graph) | 0.0510 | 0.0304 | 0.0701 | 0.0354 |
| SCOPE-U (FREEDOM + LGMRec + proxy base + set view) | 0.0478 | 0.0273 | 0.0703 | 0.0331 |

Table A2. Electronics, three seeds: mean (pstd) and per-seed values.

| Variant | R@20 mean (pstd) | R@20 per seed | N@20 mean (pstd) | N@20 per seed |
|---|---|---|---|---|
| SCOPE-v1 | 0.0637 (0.0002) | 0.0634 / 0.0639 / 0.0639 | 0.0308 (0.0001) | 0.0307 / 0.0309 / 0.0309 |
| SCOPE-G | 0.0703 (0.0001) | 0.0701 / 0.0703 / 0.0704 | 0.0354 (0.0000) | 0.0354 / 0.0355 / 0.0354 |
| SCOPE-U | 0.0705 (0.0001) | 0.0703 / 0.0706 / 0.0706 | 0.0331 (0.0000) | 0.0331 / 0.0332 / 0.0331 |

Table A3. Electronics, reproduced versus published R@20 / N@20. "Indep." marks values taken from an independent reproduction (EGRA, arXiv 2508.16170, Table I) rather than from the method's own paper.

| Method | Ours | Published | Delta % | Published source |
|---|---|---|---|---|
| GUME | -- / -- | 0.0680 / 0.0310 | -- | arXiv 2407.12338 Table 2 |
| DA-MRS | -- / -- | 0.0644 / 0.0299 (indep.) | -- | EGRA Table I |
| LGMRec | 0.0597 / 0.0270 | 0.0625 / 0.0287 (indep.) | -4.5 / -5.9 | EGRA Table I |
| FREEDOM | 0.0588 / 0.0263 | 0.0601 / 0.0273 | -2.2 / -3.7 | EGRA Table I (equals the MMRec README) |
| MENTOR | 0.0582 / 0.0266 | 0.0655 / 0.0300 | -11.1 / -11.3 | GUME, arXiv 2407.12338 Table 2 |
| MGCN | 0.0510 / 0.0235 | 0.0650 / 0.0302 | -21.5 / -22.2 | GUME, arXiv 2407.12338 Table 2 |
| BM3 | 0.0543 / 0.0258 | 0.0648 / 0.0302 | -16.2 / -14.6 | BM3, arXiv 2207.05969 Table 3 |
| LATTICE | 0.0578 / 0.0267 | -- | -- | -- |
| GRCN | -- / -- | 0.0529 / 0.0241 | -- | BM3, arXiv 2207.05969 Table 3 |
| LightGCN | 0.0533 / 0.0246 | 0.0540 / 0.0250 | -1.3 / -1.6 | BM3, arXiv 2207.05969 Table 3 |
| VBPR | 0.0263 / 0.0106 | 0.0458 / 0.0202 | -42.6 / -47.5 | BM3, arXiv 2207.05969 Table 3 |
| MMGCN | 0.0339 / 0.0147 | 0.0331 / 0.0141 | +2.4 / +4.3 | BM3, arXiv 2207.05969 Table 3 |

The best value of any compared method from either source on Electronics is GUME's published 0.0680 R@20 and 0.0310 N@20; SCOPE-U (0.0703 / 0.0331) is 3.4% and 6.8% above them.

Interpretation. On Electronics the three SCOPE variants score above every re-run baseline on all four metrics, with LGMRec the strongest re-run baseline (0.0597 R@20, 0.0270 N@20), and SCOPE-U stays above GUME's published Electronics numbers, which we did not reproduce. Two caveats bound this reading. The Electronics scorer is not the scorer of the paper: the base is a co-occurrence proxy without the text term, the set view is L1-scored, and SCOPE-G propagates three steps over a co-occurrence-only graph. And the reproduction gaps of Table A3 are wide for MENTOR, MGCN, BM3 and VBPR (-11% to -48%), so the baseline side of this table is weaker than the published state of these methods on Electronics; only LightGCN, FREEDOM, LGMRec and MMGCN are reproduced within 6%. SCOPE-G leads SCOPE-U on N@20 here (0.0354 vs 0.0331) while the two are level on R@20, which differs from the four datasets of the paper, where the composed variant leads on both metrics.

## B. New users with truncated context, and items without interactions

This section reports the context-truncation diagnostic that measures how SCOPE-v1 and SCOPE-G score users from one to five items, and the accuracy of the base's text term on items without training interactions.

Disclosure. The diagnostic is not user-disjoint: the users are training users whose context is truncated to their first k training items at scoring time, so the base, the head and the item graph were fitted with those users' other interactions. It measures robustness to short contexts, not accuracy on users unseen in fitting. The few-shot scoring of Table B1 used the L1-normalized variant of the cosine head, and on Sports it used a base ridge weight of 800 instead of the validation-selected 1500; the cosine-scored differences of Table B3 differ from Table B1 by at most 0.0006 per cell.

Table B1. R@20 on the truncated-context users (Baby 4,382 users, Sports 7,858, Clothing 5,200) when the scorer sees k items. "Base on k items" is the base scored from the same k items; "content kNN" ranks by text similarity to the k items; "session kNN" is an item-kNN over the k items; PopRec ranks by popularity. The last two columns give the paired difference to the base on k items with its bootstrap p.

| Dataset | k | SCOPE-v1 | SCOPE-G | Base on k items | Content kNN | Session kNN | PopRec | SCOPE-v1 minus base (p) | SCOPE-G minus base (p) |
|---|---|---|---|---|---|---|---|---|---|
| Baby | 1 | 0.0453 | 0.0463 | 0.0356 | 0.0189 | 0.0429 | 0.0516 | +0.0097 (0) | +0.0107 (0) |
| Baby | 2 | 0.0658 | 0.0642 | 0.0531 | 0.0243 | 0.0506 | 0.0516 | +0.0128 (0) | +0.0111 (0) |
| Baby | 3 | 0.0739 | 0.0733 | 0.0631 | 0.0237 | 0.0487 | 0.0516 | +0.0108 (0) | +0.0103 (0) |
| Baby | 5 | 0.0843 | 0.0811 | 0.0805 | 0.0324 | 0.0570 | 0.0516 | +0.0038 (0.1264) | +0.0006 (0.8046) |
| Sports | 1 | 0.0478 | 0.0460 | 0.0358 | 0.0204 | 0.0419 | 0.0367 | +0.0120 (0) | +0.0102 (0) |
| Sports | 2 | 0.0692 | 0.0680 | 0.0578 | 0.0270 | 0.0485 | 0.0367 | +0.0113 (0) | +0.0102 (0) |
| Sports | 3 | 0.0823 | 0.0815 | 0.0706 | 0.0299 | 0.0533 | 0.0367 | +0.0117 (0) | +0.0109 (0) |
| Sports | 5 | 0.1009 | 0.0972 | 0.0878 | 0.0361 | 0.0603 | 0.0367 | +0.0131 (0) | +0.0094 (0) |
| Clothing | 1 | 0.0372 | 0.0379 | 0.0342 | 0.0310 | 0.0256 | 0.0173 | +0.0030 (0.0086) | +0.0038 (0) |
| Clothing | 2 | 0.0551 | 0.0539 | 0.0507 | 0.0404 | 0.0280 | 0.0173 | +0.0043 (0.0014) | +0.0031 (0.0130) |
| Clothing | 3 | 0.0664 | 0.0641 | 0.0642 | 0.0465 | 0.0307 | 0.0173 | +0.0023 (0.1268) | -0.0001 (0.9772) |
| Clothing | 5 | 0.0830 | 0.0817 | 0.0810 | 0.0527 | 0.0376 | 0.0173 | +0.0020 (0.1552) | +0.0007 (0.5952) |

Table B2. Head alone (cosine scoring, independent of the base) on random k-item subsets of the same users, pooled R@20, against a session item-kNN and PopRec.

| Dataset | Scorer | k = 1 | k = 2 | k = 3 | k = 5 | Full context |
|---|---|---|---|---|---|---|
| Baby | SCOPE head | 0.0453 | 0.0594 | 0.0650 | 0.0763 | 0.0832 |
| Baby | Session kNN | 0.0338 | 0.0457 | 0.0558 | 0.0679 | -- |
| Baby | PopRec | 0.0511 | 0.0511 | 0.0511 | 0.0511 | -- |
| Sports | SCOPE head | 0.0427 | 0.0607 | 0.0703 | 0.0830 | 0.0989 |
| Sports | Session kNN | 0.0326 | 0.0463 | 0.0577 | 0.0696 | -- |
| Sports | PopRec | 0.0346 | 0.0346 | 0.0346 | 0.0346 | -- |
| Clothing | SCOPE head | 0.0333 | 0.0452 | 0.0530 | 0.0638 | 0.0767 |
| Clothing | Session kNN | 0.0166 | 0.0228 | 0.0275 | 0.0363 | -- |
| Clothing | PopRec | 0.0164 | 0.0164 | 0.0164 | 0.0164 | -- |

Table B3. Paired differences of the cosine-scored head: head minus session kNN, and head minus the base scored on the same k items (delta R@20, bootstrap p).

| Dataset | k | Head minus session kNN (p) | Head minus base on k items (p) |
|---|---|---|---|
| Baby | 1 | +0.0009 (0.7666) | +0.0083 (0.0060) |
| Baby | 2 | +0.0099 (0.0076) | +0.0077 (0.0372) |
| Baby | 3 | +0.0208 (0) | +0.0062 (0.1010) |
| Sports | 1 | +0.0056 (0.0114) | +0.0117 (0) |
| Sports | 2 | +0.0208 (0) | +0.0115 (0) |
| Sports | 3 | +0.0238 (0) | +0.0066 (0.0230) |
| Clothing | 1 | +0.0120 (0) | +0.0037 (0.1272) |
| Clothing | 2 | +0.0174 (0) | -0.0053 (0.0604) |
| Clothing | 3 | +0.0241 (0) | -0.0093 (0.0032) |

Table B4. Items without training interactions: 20% of the items are removed from the fit, and the base ranks them through its text term only. "Full-base ceiling" is the same items scored by a base fitted with them (in-sample), and retention is the ratio of the two.

| Dataset | Held-out items | Content-only R@20 | Full-base ceiling R@20 | Retention |
|---|---|---|---|---|
| Baby | 1,410 | 0.0457 | 0.0739 | 62% |
| Sports | 3,671 | 0.0542 | 0.0876 | 62% |
| Clothing | 4,606 | 0.0788 | 0.0870 | 91% |

Interpretation. With one to three items of context, SCOPE-v1 and SCOPE-G score 0.0097 to 0.0128 R@20 above the base on Baby and Sports (all p = 0); on Clothing the margins are +0.0023 to +0.0043 for SCOPE-v1 and -0.0001 to +0.0038 for SCOPE-G. At five items the margin closes on Baby and Clothing (p = 0.13 to 0.80) and stays at +0.0131 (SCOPE-v1) and +0.0094 (SCOPE-G) on Sports. The head alone follows the same pattern against a session kNN (Table B3, significant from k = 2 on Baby and from k = 1 on Sports and Clothing), but against the base it is level or below on Clothing at k = 2 and k = 3, where the text term of the base is strong. Because every scorer here was fitted with these users' remaining interactions, these margins overstate what a user unseen in fitting would receive, and the paper makes no empirical new-user claim from them; user-inductive scoring is stated in the paper as a property of the construction only. On items without interactions the text term retains 62% of the in-sample base on Baby and Sports and 91% on Clothing.

## C. Closed-form content kernels and the head on the strongest published kernel

This section reports the family of content-augmented closed forms fitted on one shared validation grid next to the base of the paper, and the set-completion head fused with the strongest of them on each dataset.

The family: EASE; the base of the paper ("Base-published", the values used throughout the paper, and "Base-retuned", the same form re-tuned on the shared grid); CEASE and Add-EASE with the text embedding ("-emb") or the tag matrix ("-tags"); FEASE with the prior only, on tf-idf tags or on embeddings; FEASE-full on tf-idf tags; L3AE; and two controlled swaps, Add-EASE-z (Add-EASE with the base's z-scored text term) and FEASE-kNN (FEASE with the base's text-kNN affinity). Every hyper-parameter is selected on validation R@20 by coordinate tuning on the shared grid; when a selected value lies on the edge of the shared grid, the grid is extended along a ladder and the extension is logged ("edge"). Seed 2024 (the closed forms are seed-free; the seed fixes the evaluation environment). Holm-adjusted p-values are against Base-published, over the family of 22 tests per dataset.

On MicroLens, whose benchmark release has no item metadata, the tag matrix and the tf-idf vectors are built from each video's English title and category label in the official MicroLens-100k release (the benchmark's item IDs map onto the official video IDs: every one of its 705,174 interactions is found in the official interaction list). The same tokenisation and frequency thresholds as on the Amazon datasets apply; the fields MicroLens lacks (description, brand) stay empty, and user comments and like/view counts are not used.

Table C1. Shared grid (coordinate tuning; the ladders are the edge extensions).

| Parameter | Shared grid | Edge-extension ladder |
|---|---|---|
| lambda (ridge; EASE, base, CEASE) | 50, 100, 200, 400, 800, 1500, 3000 | 6.25 ... 24000 (13 points, factor 2) |
| a (base text weight) | 0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5 | 0 ... 5.0 (11 points) |
| w (CEASE-emb weight) | 1, 3, 10, 30, 100, 300, 1000 | 0.03 ... 30000 (13 points) |
| alpha (CEASE-tags / FEASE-full) | 0 ... 1.0 in steps of 0.05 (21 points) | 0 ... 1.15 (24 points) |
| beta (Add-EASE) | 0 ... 1.0 in steps of 0.05 (21 points) | same as the shared grid |
| lambda_T, embedding (Add-EASE-emb) | 1, 5, 10, 50, 100, 500 | 0.05 ... 10000 (12 points) |
| lambda_T, tags (Add-EASE-tags) | 10, 50, 100, 200, 500, 1000 | 0.5 ... 10000 (12 points) |
| lambda_R (Add-EASE) | the lambda grid | the lambda ladder |
| lambda_F (L3AE) | 0.1, 0.5, 1, 5, 10, 50, 100 | 0.005 ... 5000 (13 points) |
| rho (FEASE) | 0, 0.25, 0.5, 0.75, 1.0 | same as the shared grid |
| rho (L3AE) | 0, 0.5, 1.0, 1.5, 2.0, 3.0 | 0 ... 6.0 (9 points) |
| smult (FEASE prior scale) | 0.25, 1.0, 4.0 | 1/256 ... 256 (9 points, factor 4) |
| alpha_F (FEASE-full) | 0.1, 0.2, 0.4 | 0.0125 ... 3.2 (9 points) |
| lambda_tot (FEASE, L3AE) | the lambda grid | the lambda ladder |
| centering of the embedding | raw, cent | -- |
| tag variant | jeunen, uniform | -- |
| gamma (fusion weight of the head, Table C8) | 0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0 | -- |

Table C2. Baby (selected kernel: L3AE). Holm p against Base-published.

| Model | Selected hyper-parameters | Val R@20 | R@10 | N@10 | R@20 | N@20 | Holm p R@20 | Holm p N@20 | Grid edge |
|---|---|---|---|---|---|---|---|---|---|
| EASE | lam 400 | 0.0824 | 0.0561 | 0.0327 | 0.0825 | 0.0396 | 0.0044 | 0.0044 | |
| Base-retuned | lam 800, a 0.5 | 0.0939 | 0.0646 | 0.0371 | 0.0925 | 0.0443 | -- | -- | |
| Base-published | lam 800, a 0.5 | 0.0939 | 0.0646 | 0.0371 | 0.0925 | 0.0443 | -- | -- | |
| CEASE-emb | cent, w 10, lam 400 | 0.0958 | 0.0662 | 0.0384 | 0.0946 | 0.0457 | 0.5641 | 0.0240 | |
| CEASE-tags | uniform, alpha 0.35, lam 400 | 0.0892 | 0.0599 | 0.0337 | 0.0882 | 0.0410 | 0.0224 | 0.0044 | |
| Add-EASE-emb | cent, lam_T 500, beta 0.8, lam_R 400 | 0.0976 | 0.0675 | 0.0386 | 0.0957 | 0.0459 | 0.1032 | 0.0072 | edge |
| Add-EASE-tags | jeunen, lam_T 10, beta 0.05, lam_R 400 | 0.0929 | 0.0608 | 0.0342 | 0.0889 | 0.0414 | 0.0560 | 0.0044 | edge |
| FEASE-tfidf | lam_tot 1500, rho 0.5, smult 1.0 | 0.0977 | 0.0641 | 0.0366 | 0.0951 | 0.0446 | 0.3388 | 1 | |
| FEASE-emb | lam_tot 3000, cent, rho 0.5, smult 1.0 | 0.0973 | 0.0662 | 0.0373 | 0.0947 | 0.0447 | 0.5641 | 1 | edge |
| FEASE-full-tfidf | lam_tot 1500, alpha 0.025, rho 0.5, smult 1.0 | 0.0977 | 0.0640 | 0.0366 | 0.0950 | 0.0446 | 0.3388 | 1 | edge |
| L3AE | cent, lam_F 100, rho 1.5, lam_tot 400 | 0.0980 | 0.0675 | 0.0388 | 0.0950 | 0.0459 | 0.3388 | 0.0072 | edge |
| Add-EASE-z (swap) | lam_T 100, a 1.5, lam_R 1500 | 0.0976 | 0.0668 | 0.0381 | 0.0950 | 0.0454 | 0.3388 | 0.0884 | edge |
| FEASE-kNN (swap) | lam_tot 400, rho 0.75, smult 1.0 | 0.0947 | 0.0641 | 0.0372 | 0.0934 | 0.0447 | 1 | 0.6339 | |

Table C3. Sports (selected kernel: Add-EASE-emb).

| Model | Selected hyper-parameters | Val R@20 | R@10 | N@10 | R@20 | N@20 | Holm p R@20 | Holm p N@20 | Grid edge |
|---|---|---|---|---|---|---|---|---|---|
| EASE | lam 400 | 0.0931 | 0.0680 | 0.0402 | 0.0934 | 0.0468 | 0.0044 | 0.0044 | |
| Base-retuned | lam 1500, a 0.5 | 0.1071 | 0.0762 | 0.0440 | 0.1077 | 0.0521 | -- | -- | |
| Base-published | lam 1500, a 0.5 | 0.1071 | 0.0762 | 0.0440 | 0.1077 | 0.0521 | -- | -- | |
| CEASE-emb | cent, w 10, lam 400 | 0.1114 | 0.0796 | 0.0464 | 0.1137 | 0.0553 | 0.0044 | 0.0044 | |
| CEASE-tags | uniform, alpha 0.3, lam 200 | 0.1019 | 0.0724 | 0.0420 | 0.1035 | 0.0501 | 0.0044 | 0.0044 | edge |
| Add-EASE-emb | cent, lam_T 1, beta 0.25, lam_R 800 | 0.1137 | 0.0799 | 0.0459 | 0.1152 | 0.0551 | 0.0044 | 0.0044 | edge |
| Add-EASE-tags | uniform, lam_T 50, beta 0.05, lam_R 800 | 0.1095 | 0.0776 | 0.0430 | 0.1110 | 0.0516 | 0.0044 | 0.2328 | |
| FEASE-tfidf | lam_tot 800, rho 0.25, smult 4.0 | 0.1117 | 0.0800 | 0.0459 | 0.1132 | 0.0545 | 0.0044 | 0.0044 | edge |
| FEASE-emb | lam_tot 400, cent, rho 1.0, smult 4.0 | 0.1101 | 0.0793 | 0.0459 | 0.1114 | 0.0543 | 0.0044 | 0.0044 | edge |
| FEASE-full-tfidf | lam_tot 800, alpha 0.05, rho 0.25, smult 4.0 | 0.1116 | 0.0800 | 0.0459 | 0.1132 | 0.0545 | 0.0044 | 0.0044 | edge |
| L3AE | cent, lam_F 1, rho 0.5, lam_tot 400 | 0.1133 | 0.0807 | 0.0468 | 0.1160 | 0.0559 | 0.0044 | 0.0044 | |
| Add-EASE-z (swap) | lam_T 10, a 2.0, lam_R 800 | 0.1119 | 0.0793 | 0.0459 | 0.1139 | 0.0549 | 0.0044 | 0.0044 | edge |
| FEASE-kNN (swap) | lam_tot 400, rho 0.75, smult 1.0 | 0.1093 | 0.0777 | 0.0451 | 0.1108 | 0.0537 | 0.0044 | 0.0044 | |

Table C4. Clothing (selected kernel: Add-EASE-tags).

| Model | Selected hyper-parameters | Val R@20 | R@10 | N@10 | R@20 | N@20 | Holm p R@20 | Holm p N@20 | Grid edge |
|---|---|---|---|---|---|---|---|---|---|
| EASE | lam 400 | 0.0563 | 0.0425 | 0.0256 | 0.0574 | 0.0294 | 0.0044 | 0.0044 | |
| Base-retuned | lam 1500, a 1.0 | 0.0914 | 0.0618 | 0.0344 | 0.0909 | 0.0418 | -- | -- | |
| Base-published | lam 1500, a 0.7 | 0.0913 | 0.0631 | 0.0355 | 0.0910 | 0.0426 | -- | -- | |
| CEASE-emb | cent, w 30, lam 400 | 0.1000 | 0.0698 | 0.0386 | 0.0974 | 0.0457 | 0.0044 | 0.0044 | |
| CEASE-tags | jeunen, alpha 0.8, lam 200 | 0.0994 | 0.0677 | 0.0374 | 0.0981 | 0.0451 | 0.0044 | 0.0044 | edge |
| Add-EASE-emb | cent, lam_T 5, beta 0.45, lam_R 400 | 0.1009 | 0.0703 | 0.0388 | 0.0985 | 0.0459 | 0.0044 | 0.0044 | |
| Add-EASE-tags | jeunen, lam_T 500, beta 0.35, lam_R 400 | 0.1013 | 0.0693 | 0.0383 | 0.0990 | 0.0458 | 0.0044 | 0.0044 | |
| FEASE-tfidf | lam_tot 400, rho 0.75, smult 4.0 | 0.0934 | 0.0663 | 0.0372 | 0.0943 | 0.0443 | 0.0056 | 0.0044 | edge |
| FEASE-emb | lam_tot 200, cent, rho 1.0, smult 16.0 | 0.0912 | 0.0654 | 0.0368 | 0.0912 | 0.0434 | 0.7821 | 0.1482 | edge |
| FEASE-full-tfidf | lam_tot 400, alpha 0.4, rho 0.5, smult 4.0 | 0.1000 | 0.0696 | 0.0387 | 0.0991 | 0.0462 | 0.0044 | 0.0044 | edge |
| L3AE | cent, lam_F 10, rho 1.0, lam_tot 400 | 0.1012 | 0.0709 | 0.0392 | 0.0990 | 0.0464 | 0.0044 | 0.0044 | |
| Add-EASE-z (swap) | lam_T 0.5, a 3.0, lam_R 3000 | 0.0983 | 0.0685 | 0.0378 | 0.0967 | 0.0450 | 0.0044 | 0.0044 | edge |
| FEASE-kNN (swap) | lam_tot 3000, rho 0.75, smult 0.25 | 0.0939 | 0.0634 | 0.0353 | 0.0937 | 0.0429 | 0.0044 | 0.1784 | edge |

Table C5. MicroLens (selected kernel: Add-EASE-tags; tag and tf-idf side information from each video's title and category label, see above).

| Model | Selected hyper-parameters | Val R@20 | R@10 | N@10 | R@20 | N@20 | Holm p R@20 | Holm p N@20 | Grid edge |
|---|---|---|---|---|---|---|---|---|---|
| EASE | lam 200 | 0.1262 | 0.0908 | 0.0501 | 0.1249 | 0.0590 | 0.0044 | 0.0044 |  |
| Base-retuned | lam 200, a 0.3 | 0.1283 | 0.0926 | 0.0510 | 0.1280 | 0.0601 | -- | -- |  |
| Base-published | lam 400, a 0.3 | 0.1279 | 0.0918 | 0.0506 | 0.1276 | 0.0598 | -- | -- |  |
| CEASE-emb | cent, w 10, lam 200 | 0.1304 | 0.0938 | 0.0516 | 0.1292 | 0.0607 | 0.0056 | 0.0044 |  |
| CEASE-tags | jeunen, alpha 0.75, lam 200 | 0.1306 | 0.0942 | 0.0519 | 0.1293 | 0.0610 | 0.0044 | 0.0044 |  |
| Add-EASE-emb | raw, lam_T 10, beta 0.45, lam_R 200 | 0.1310 | 0.0938 | 0.0516 | 0.1293 | 0.0608 | 0.0044 | 0.0044 |  |
| Add-EASE-tags | jeunen, lam_T 100, beta 0.3, lam_R 200 | 0.1316 | 0.0942 | 0.0519 | 0.1294 | 0.0611 | 0.0044 | 0.0044 |  |
| FEASE-tfidf | lam_tot 200, rho 0.75, smult 1.0 | 0.1299 | 0.0936 | 0.0515 | 0.1288 | 0.0606 | 0.0180 | 0.0044 |  |
| FEASE-emb | lam_tot 200, cent, rho 0.5, smult 4.0 | 0.1290 | 0.0931 | 0.0512 | 0.1277 | 0.0602 | 0.7437 | 0.1720 | edge |
| FEASE-full-tfidf | lam_tot 200, alpha 0.4, rho 0.75, smult 1.0 | 0.1311 | 0.0943 | 0.0519 | 0.1300 | 0.0612 | 0.0044 | 0.0044 | edge |
| L3AE | raw, lam_F 10, rho 1.0, lam_tot 200 | 0.1308 | 0.0940 | 0.0517 | 0.1293 | 0.0609 | 0.0044 | 0.0044 |  |
| Add-EASE-z (swap) | lam_T 10, a 1.0, lam_R 400 | 0.1300 | 0.0924 | 0.0507 | 0.1285 | 0.0601 | 0.0800 | 0.3372 |  |
| FEASE-kNN (swap) | lam_tot 100, rho 0.75, smult 1.0 | 0.1286 | 0.0940 | 0.0520 | 0.1281 | 0.0608 | 0.3372 | 0.0044 |  |

Table C6. Validation ranking of the literature pool (validation R@20, seed 2024) and the validation-selected kernel against the test-best member of the pool.

| Dataset | Validation ranking (val R@20) | Selected on validation | Selected test R@20 / N@20 | Test-best member (test R@20) |
|---|---|---|---|---|
| Baby | L3AE 0.0980, FEASE-tfidf 0.0977, FEASE-full-tfidf 0.0977, Add-EASE-emb 0.0976, FEASE-emb 0.0973, CEASE-emb 0.0958, Add-EASE-tags 0.0929, CEASE-tags 0.0892 | L3AE | 0.0950 / 0.0459 | Add-EASE-emb (0.0957) |
| Sports | Add-EASE-emb 0.1137, L3AE 0.1133, FEASE-tfidf 0.1117, FEASE-full-tfidf 0.1116, CEASE-emb 0.1114, FEASE-emb 0.1101, Add-EASE-tags 0.1095, CEASE-tags 0.1019 | Add-EASE-emb | 0.1152 / 0.0551 | L3AE (0.1160) |
| Clothing | Add-EASE-tags 0.1013, L3AE 0.1012, Add-EASE-emb 0.1009, CEASE-emb 0.1000, FEASE-full-tfidf 0.1000, CEASE-tags 0.0994, FEASE-tfidf 0.0934, FEASE-emb 0.0912 | Add-EASE-tags | 0.0990 / 0.0458 | FEASE-full-tfidf (0.0991) |
| MicroLens | Add-EASE-tags 0.1316, FEASE-full-tfidf 0.1311, Add-EASE-emb 0.1310, L3AE 0.1308, CEASE-tags 0.1306, CEASE-emb 0.1304, FEASE-tfidf 0.1299, FEASE-emb 0.1290 | Add-EASE-tags | 0.1294 / 0.0611 | FEASE-full-tfidf (0.1300) |

On each dataset the validation-selected kernel is not the test-best member of the pool; the selection is kept as made on validation. On MicroLens FEASE-full-tfidf is the test-best member (0.1300 R@20 against 0.1294 for the selected Add-EASE-tags) with a validation score 0.0006 below it (on unrounded values).

Table C7. Pre-committed rules of the family study.

| Rule | Outcome |
|---|---|
| Number of literature kernels at or above Base-published on test R@20 or N@20 | Baby 6, Sports 7, Clothing 8, MicroLens 8 |
| The base is one member of the family on | Baby, Sports, Clothing, MicroLens (all four) |
| An earlier statement that the base alone outranks every re-run baseline on MicroLens | withdrawn: CEASE-emb, CEASE-tags, Add-EASE-emb, Add-EASE-tags, FEASE-tfidf, FEASE-full-tfidf and L3AE exceed it on both metrics (Holm p below 0.05) |
| Re-tune of the base on the shared grid, Baby | unchanged (lam 800, a 0.5); seed SD of R@20 0.0006 |
| Re-tune, Sports | unchanged (lam 1500, a 0.5); seed SD of R@20 0.0006 |
| Re-tune, Clothing | a 0.7 to 1.0; test R@20 -0.0000, N@20 -0.0008 (Holm p 0.0016), beyond the seed SD of 0.0001 |
| Re-tune, MicroLens | lam 400 to 200; test R@20 +0.0004 (Holm p 0.8183), within its seed SD of 0.0005; N@20 +0.0003 (Holm p 0.0154), beyond its seed SD of 0.0001 |

The re-tune is advisory: the paper keeps Base-published as its base on every dataset and reports the literature kernels as prior art next to it.

Table C8. SCOPE-v1 on the strongest published kernel: z(set head) + gamma z(kernel), gamma selected on validation per head seed from the gamma grid of Table C1, with the deployed head checkpoints of the paper unchanged. "Kernel alone" repeats the selected kernel's test values.

| Dataset | Kernel | Seed | gamma | Val R@20 | R@10 | N@10 | R@20 | N@20 |
|---|---|---|---|---|---|---|---|---|
| Baby | L3AE | 2024 | 1.0 | 0.1027 | 0.0692 | 0.0393 | 0.0999 | 0.0473 |
| Baby | L3AE | 2025 | 0.6 | 0.1048 | 0.0682 | 0.0390 | 0.1017 | 0.0476 |
| Baby | L3AE | 2026 | 0.6 | 0.1036 | 0.0696 | 0.0396 | 0.1031 | 0.0482 |
| Baby | L3AE | mean (pstd) | | | 0.0690 | 0.0393 | 0.1016 (0.0013) | 0.0477 (0.0004) |
| Baby | L3AE | kernel alone | | 0.0980 | 0.0675 | 0.0388 | 0.0950 | 0.0459 |
| Sports | Add-EASE-emb | 2024 | 1.0 | 0.1197 | 0.0838 | 0.0477 | 0.1218 | 0.0575 |
| Sports | Add-EASE-emb | 2025 | 0.6 | 0.1196 | 0.0842 | 0.0479 | 0.1220 | 0.0577 |
| Sports | Add-EASE-emb | 2026 | 0.6 | 0.1190 | 0.0847 | 0.0479 | 0.1222 | 0.0576 |
| Sports | Add-EASE-emb | mean (pstd) | | | 0.0842 | 0.0479 | 0.1220 (0.0002) | 0.0576 (0.0001) |
| Sports | Add-EASE-emb | kernel alone | | 0.1137 | 0.0799 | 0.0459 | 0.1152 | 0.0551 |
| Clothing | Add-EASE-tags | 2024 | 1.5 | 0.1036 | 0.0712 | 0.0392 | 0.1029 | 0.0472 |
| Clothing | Add-EASE-tags | 2025 | 1.0 | 0.1041 | 0.0718 | 0.0394 | 0.1019 | 0.0471 |
| Clothing | Add-EASE-tags | 2026 | 1.0 | 0.1035 | 0.0714 | 0.0393 | 0.1034 | 0.0475 |
| Clothing | Add-EASE-tags | mean (pstd) | | | 0.0715 | 0.0393 | 0.1028 (0.0006) | 0.0473 (0.0002) |
| Clothing | Add-EASE-tags | kernel alone | | 0.1013 | 0.0693 | 0.0383 | 0.0990 | 0.0458 |
| MicroLens | Add-EASE-tags | 2024 | 0.3 | 0.1365 | 0.0964 | 0.0530 | 0.1358 | 0.0631 |
| MicroLens | Add-EASE-tags | 2025 | 0.3 | 0.1355 | 0.0961 | 0.0531 | 0.1353 | 0.0632 |
| MicroLens | Add-EASE-tags | 2026 | 0.3 | 0.1360 | 0.0962 | 0.0529 | 0.1344 | 0.0628 |
| MicroLens | Add-EASE-tags | mean (pstd) | | | 0.0962 | 0.0530 | 0.1352 (0.0006) | 0.0630 (0.0002) |
| MicroLens | Add-EASE-tags | kernel alone | | 0.1316 | 0.0942 | 0.0519 | 0.1294 | 0.0611 |

Table C9. Head over kernel: fused minus kernel alone, paired user bootstrap (B = 10,000). Holm over the 24 per-seed tests (per-seed rows) and over the 8 seed-mean tests (seed-mean rows); seed-mean tests average each user's metric over the three seeds before resampling.

| Dataset | Seed, metric | Delta | 95% CI | Relative % [CI] | p | Holm p |
|---|---|---|---|---|---|---|
| Baby | 2024, R@20 | +0.0049 | [0.0030, 0.0067] | 5.1 [3.1, 7.1] | 0.0002 | 0.0048 |
| Baby | 2024, N@20 | +0.0014 | [0.0007, 0.0020] | 2.9 [1.6, 4.3] | 0.0004 | 0.0048 |
| Baby | 2025, R@20 | +0.0067 | [0.0044, 0.0090] | 7.1 [4.7, 9.5] | 0.0002 | 0.0048 |
| Baby | 2025, N@20 | +0.0017 | [0.0009, 0.0025] | 3.7 [2.0, 5.4] | 0.0002 | 0.0048 |
| Baby | 2026, R@20 | +0.0080 | [0.0057, 0.0104] | 8.5 [6.0, 11.0] | 0.0002 | 0.0048 |
| Baby | 2026, N@20 | +0.0023 | [0.0015, 0.0031] | 4.9 [3.2, 6.7] | 0.0002 | 0.0048 |
| Baby | seed mean, R@20 | +0.0065 | [0.0047, 0.0084] | 6.9 [5.0, 8.9] | 0.0002 | 0.0016 |
| Baby | seed mean, N@20 | +0.0018 | [0.0011, 0.0024] | 3.9 [2.5, 5.3] | 0.0002 | 0.0016 |
| Sports | 2024, R@20 | +0.0065 | [0.0050, 0.0081] | 5.7 [4.3, 7.0] | 0.0002 | 0.0048 |
| Sports | 2024, N@20 | +0.0024 | [0.0019, 0.0029] | 4.4 [3.5, 5.3] | 0.0002 | 0.0048 |
| Sports | 2025, R@20 | +0.0067 | [0.0050, 0.0085] | 5.8 [4.3, 7.4] | 0.0002 | 0.0048 |
| Sports | 2025, N@20 | +0.0026 | [0.0020, 0.0032] | 4.7 [3.6, 5.8] | 0.0002 | 0.0048 |
| Sports | 2026, R@20 | +0.0070 | [0.0052, 0.0088] | 6.0 [4.5, 7.6] | 0.0002 | 0.0048 |
| Sports | 2026, N@20 | +0.0025 | [0.0019, 0.0032] | 4.6 [3.5, 5.8] | 0.0002 | 0.0048 |
| Sports | seed mean, R@20 | +0.0067 | [0.0053, 0.0083] | 5.9 [4.6, 7.2] | 0.0002 | 0.0016 |
| Sports | seed mean, N@20 | +0.0025 | [0.0020, 0.0030] | 4.6 [3.6, 5.5] | 0.0002 | 0.0016 |
| Clothing | 2024, R@20 | +0.0039 | [0.0028, 0.0049] | 3.9 [2.9, 5.0] | 0.0002 | 0.0048 |
| Clothing | 2024, N@20 | +0.0014 | [0.0011, 0.0017] | 3.1 [2.4, 3.8] | 0.0002 | 0.0048 |
| Clothing | 2025, R@20 | +0.0029 | [0.0017, 0.0041] | 2.9 [1.7, 4.1] | 0.0002 | 0.0048 |
| Clothing | 2025, N@20 | +0.0013 | [0.0009, 0.0017] | 2.8 [2.0, 3.6] | 0.0002 | 0.0048 |
| Clothing | 2026, R@20 | +0.0044 | [0.0032, 0.0056] | 4.5 [3.2, 5.7] | 0.0002 | 0.0048 |
| Clothing | 2026, N@20 | +0.0017 | [0.0013, 0.0021] | 3.6 [2.7, 4.5] | 0.0002 | 0.0048 |
| Clothing | seed mean, R@20 | +0.0037 | [0.0027, 0.0047] | 3.8 [2.7, 4.8] | 0.0002 | 0.0016 |
| Clothing | seed mean, N@20 | +0.0015 | [0.0011, 0.0018] | 3.2 [2.5, 3.9] | 0.0002 | 0.0016 |
| MicroLens | 2024, R@20 | +0.0064 | [0.0054, 0.0074] | 4.9 [4.2, 5.7] | 0.0002 | 0.0048 |
| MicroLens | 2024, N@20 | +0.0020 | [0.0017, 0.0024] | 3.4 [2.8, 3.9] | 0.0002 | 0.0048 |
| MicroLens | 2025, R@20 | +0.0058 | [0.0049, 0.0068] | 4.5 [3.8, 5.2] | 0.0002 | 0.0048 |
| MicroLens | 2025, N@20 | +0.0021 | [0.0018, 0.0025] | 3.5 [2.9, 4.1] | 0.0002 | 0.0048 |
| MicroLens | 2026, R@20 | +0.0050 | [0.0041, 0.0060] | 3.9 [3.1, 4.6] | 0.0002 | 0.0048 |
| MicroLens | 2026, N@20 | +0.0017 | [0.0013, 0.0021] | 2.8 [2.2, 3.4] | 0.0002 | 0.0048 |
| MicroLens | seed mean, R@20 | +0.0057 | [0.0049, 0.0066] | 4.4 [3.8, 5.1] | 0.0002 | 0.0016 |
| MicroLens | seed mean, N@20 | +0.0020 | [0.0017, 0.0023] | 3.2 [2.7, 3.7] | 0.0002 | 0.0016 |

Table C10. Pipeline anchor: the same fusion code applied to the base of the paper reproduces the deployed SCOPE-v1 values of the paper exactly (absolute difference 0).

| Dataset | Seed | gamma | R@20 | N@20 | Reproduces the stored value |
|---|---|---|---|---|---|
| Baby | 2024 / 2025 / 2026 | 0.3 / 0.3 / 0.3 | 0.1017 / 0.1008 / 0.1017 | 0.0473 / 0.0467 / 0.0477 | yes / yes / yes |
| Sports | 2024 / 2025 / 2026 | 0.3 / 0.3 / 0.3 | 0.1160 / 0.1172 / 0.1170 | 0.0554 / 0.0556 / 0.0555 | yes / yes / yes |
| Clothing | 2024 / 2025 / 2026 | 0.6 / 1.0 / 0.6 | 0.0943 / 0.0945 / 0.0943 | 0.0440 / 0.0439 / 0.0439 | yes / yes / yes |
| MicroLens | 2024 / 2025 / 2026 | 0.3 / 0.3 / 0.3 | 0.1341 / 0.1333 / 0.1331 | 0.0621 / 0.0620 / 0.0619 | yes / yes / yes |

Table C11. Cost of fusing the head with the selected kernel (fit of the kernel, scoring and the gamma search over three seeds): wall-clock seconds and peak GPU memory.

| Dataset | Wall (s) | Peak GB |
|---|---|---|
| Baby | 7.36 | 2.485 |
| Sports | 17.85 | 10.61 |
| Clothing | 21.96 | 12.528 |
| MicroLens | 40.39 | 13.781 |

Interpretation. The literature kernels are at least as strong as the base of the paper on every dataset: six of eight members meet or exceed it on Baby, seven on Sports, eight on Clothing and eight on MicroLens, and on test R@20 the validation-selected kernel scores 0.0950 against the base's 0.0925 on Baby, 0.1152 against 0.1077 on Sports, 0.0990 against 0.0910 on Clothing and 0.1294 against 0.1276 on MicroLens. The kernel is prior art and is not a contribution of the paper; the base of the paper is kept as its reference only because every dependent analysis was run on it. The head's gain survives the swap to the strongest kernel: fused minus kernel alone is positive with Holm p = 0.0048 on all three seeds and both metrics on all four datasets, and the seed-averaged relative lift in R@20 is 6.9% (Baby), 5.9% (Sports), 3.8% (Clothing) and 4.4% (MicroLens), with per-seed lifts between 2.9% and 8.5%. Against the base of the paper the same head adds 10.1 / 7.8 / 3.6% on Baby / Sports / Clothing, so a stronger kernel absorbs part of the head's gain but not most of it. The re-tune of the base on the shared grid changes nothing on Baby and Sports and moves Clothing and MicroLens by at most 0.0008 in one metric.

## D. Matched neighbour comparison in full

This section gives every number of the matched comparison between the SCOPE set-completion head and its nearest learned relatives on Baby and Sports: the protocol, the validation grids, the selected configurations, the per-seed runs, every seed-averaged test, the CBOW gate, a summary of the per-seed tests, the reproduction check of the head, and the history of the earlier version of this comparison.

### D1. Protocol

| Element | Setting |
|---|---|
| Arms | item2vec/CBOW mean-pool head with a random item table ("CBOW random"); the same with the content (text) seed of the SCOPE head ("CBOW content"); a bidirectional set Transformer read out from a classification token (BERT4Rec-style); a causal set Transformer read out after a start token (SASRec-style); Mult-VAE |
| Shared trainer (all masked-set arms and the SCOPE head) | masked-set full-catalog softmax, L2-cosine scoring with a learnable temperature, d = 256, batch 8192 users, weight decay 1e-6, at most 400 epochs, validation every 4 epochs, patience 20 checks, full training histories as context (list cap = the maximum training degree: Baby 100, Sports 237, Clothing 109), SIGReg(E) weight in the grid, content-seeded item table except for CBOW random |
| Set Transformers | no positional embeddings; not the published BERT4Rec or SASRec |
| Mult-VAE | its own multinomial ELBO, KL annealed over 20,000 update steps, batch 500, epoch cap 1000 |
| Grids | CBOW random 4, CBOW content 4, BERT4Rec-style 8, SASRec-style 8, Mult-VAE 4 configurations (Tables D2 and D3) |
| Selection | per dataset and arm, the configuration with the highest standalone validation R@20 at its early-stopped checkpoint, on seed 2024 only; ties keep the earlier grid point; test is never consulted. The fused-selected panel (C) instead takes the configuration with the highest fused validation R@20 |
| Fusion weight gamma | tuned on validation per model and seed over {0, 0.3, 0.6, 1, 1.5, 2, 3, 5}, extended by {8, 12} when 5 is selected; the deployed SCOPE rows use the deployed per-seed gamma; the G7 CBOW gate uses the grid without extension |
| SCOPE rows | "deployed": the checkpoints of the paper, trained with 60-item training lists (users with more than 60 training items: Baby 9, Sports 26); "re-run, 60-item lists": the same head retrained in the shared trainer with the 60-item lists (reproduction check); "re-run, full lists": the same head retrained with full lists (the setting of the arms) |
| Tests | paired user-level bootstrap, B = 10,000, two-sided; seed-averaged families average each user's metric over the three seeds of each side before resampling, so the training-seed spread is not in their CI or p; per-seed families test each seed separately; Holm step-down within each family over its full pre-stated size, with missing or incomplete tests entered at p = 1 |
| Pre-registered MDE for the CBOW gate (delta R@20) | Baby 0.0026, Sports 0.0022, Clothing 0.0017 |

### D2. Validation grid, Baby (seed 2024)

Every configuration of the grid: best standalone validation R@20 (the early-stopping criterion) and N@20, fused validation R@20 at its own validation-tuned gamma and the fused N@20, best and last epoch, whether the epoch cap was hit, and whether it was selected for the standalone (A) or fused (F) rows.

| Arm | Configuration | Val R@20 | Val N@20 | Fused val R@20 (gamma) | Fused val N@20 | Best ep | Last ep | Cap hit | Selected |
|---|---|---|---|---|---|---|---|---|---|
| CBOW random | le0_lr0.001 | 0.0701 | 0.0310 | 0.0966 (1) | 0.0457 | 48 | 128 | no | |
| CBOW random | le1_lr0.001 | 0.0701 | 0.0310 | 0.0966 (1) | 0.0457 | 48 | 128 | no | A |
| CBOW random | le0_lr0.003 | 0.0688 | 0.0304 | 0.0966 (1) | 0.0459 | 24 | 104 | no | F |
| CBOW random | le1_lr0.003 | 0.0687 | 0.0304 | 0.0966 (1) | 0.0459 | 24 | 104 | no | |
| CBOW content | le0_lr0.001 | 0.0830 | 0.0364 | 0.1023 (0.3) | 0.0476 | 4 | 84 | no | A, F |
| CBOW content | le1_lr0.001 | 0.0830 | 0.0364 | 0.1022 (0.3) | 0.0476 | 4 | 84 | no | |
| CBOW content | le0_lr0.003 | 0.0774 | 0.0332 | 0.1001 (0.6) | 0.0472 | 8 | 88 | no | |
| CBOW content | le1_lr0.003 | 0.0775 | 0.0333 | 0.1000 (0.6) | 0.0472 | 8 | 88 | no | |
| BERT4Rec-style | dropout0.1_le0_lr0.001 | 0.0767 | 0.0323 | 0.0999 (0.3) | 0.0458 | 16 | 96 | no | |
| BERT4Rec-style | dropout0.1_le1_lr0.001 | 0.0769 | 0.0323 | 0.0998 (0.3) | 0.0457 | 16 | 96 | no | |
| BERT4Rec-style | dropout0.2_le0_lr0.001 | 0.0775 | 0.0330 | 0.0995 (0.3) | 0.0456 | 16 | 96 | no | A |
| BERT4Rec-style | dropout0.2_le1_lr0.001 | 0.0775 | 0.0330 | 0.0995 (0.3) | 0.0456 | 16 | 96 | no | |
| BERT4Rec-style | dropout0.1_le0_lr0.003 | 0.0741 | 0.0320 | 0.0985 (0.3) | 0.0453 | 56 | 136 | no | |
| BERT4Rec-style | dropout0.1_le1_lr0.003 | 0.0742 | 0.0320 | 0.1000 (0.3) | 0.0455 | 56 | 136 | no | F |
| BERT4Rec-style | dropout0.2_le0_lr0.003 | 0.0762 | 0.0333 | 0.0999 (0.3) | 0.0463 | 96 | 176 | no | |
| BERT4Rec-style | dropout0.2_le1_lr0.003 | 0.0765 | 0.0328 | 0.0968 (0.3) | 0.0445 | 56 | 136 | no | |
| SASRec-style | dropout0.1_le0_lr0.001 | 0.0609 | 0.0250 | 0.0960 (1) | 0.0455 | 4 | 84 | no | |
| SASRec-style | dropout0.1_le1_lr0.001 | 0.0609 | 0.0250 | 0.0960 (0.6) | 0.0454 | 4 | 84 | no | |
| SASRec-style | dropout0.2_le0_lr0.001 | 0.0637 | 0.0263 | 0.0963 (1) | 0.0456 | 4 | 84 | no | A |
| SASRec-style | dropout0.2_le1_lr0.001 | 0.0637 | 0.0263 | 0.0963 (0.6) | 0.0455 | 4 | 84 | no | |
| SASRec-style | dropout0.1_le0_lr0.003 | 0.0592 | 0.0235 | 0.0977 (1) | 0.0461 | 24 | 104 | no | |
| SASRec-style | dropout0.1_le1_lr0.003 | 0.0593 | 0.0236 | 0.0978 (1) | 0.0461 | 24 | 104 | no | F |
| SASRec-style | dropout0.2_le0_lr0.003 | 0.0626 | 0.0251 | 0.0977 (0.6) | 0.0455 | 8 | 88 | no | |
| SASRec-style | dropout0.2_le1_lr0.003 | 0.0625 | 0.0251 | 0.0977 (0.6) | 0.0456 | 8 | 88 | no | |
| Mult-VAE | beta_cap0.2_lr0.001 | 0.0766 | 0.0328 | 0.0980 (0.3) | 0.0454 | 144 | 224 | no | |
| Mult-VAE | beta_cap0.5_lr0.001 | 0.0840 | 0.0364 | 0.0984 (0.3) | 0.0453 | 392 | 472 | no | F |
| Mult-VAE | beta_cap0.2_lr0.003 | 0.0797 | 0.0341 | 0.0966 (0.3) | 0.0451 | 128 | 208 | no | |
| Mult-VAE | beta_cap0.5_lr0.003 | 0.0867 | 0.0378 | 0.0971 (0.3) | 0.0452 | 264 | 344 | no | A |

### D3. Validation grid, Sports (seed 2024)

| Arm | Configuration | Val R@20 | Val N@20 | Fused val R@20 (gamma) | Fused val N@20 | Best ep | Last ep | Cap hit | Selected |
|---|---|---|---|---|---|---|---|---|---|
| CBOW random | le0_lr0.001 | 0.0787 | 0.0359 | 0.1102 (0.6) | 0.0530 | 44 | 124 | no | A |
| CBOW random | le1_lr0.001 | 0.0787 | 0.0359 | 0.1102 (0.6) | 0.0530 | 44 | 124 | no | |
| CBOW random | le0_lr0.003 | 0.0768 | 0.0350 | 0.1104 (1) | 0.0530 | 16 | 96 | no | F |
| CBOW random | le1_lr0.003 | 0.0767 | 0.0350 | 0.1104 (1) | 0.0530 | 16 | 96 | no | |
| CBOW content | le0_lr0.001 | 0.0924 | 0.0407 | 0.1146 (0.3) | 0.0542 | 0 | 80 | no | A, F |
| CBOW content | le1_lr0.001 | 0.0924 | 0.0407 | 0.1146 (0.3) | 0.0542 | 0 | 80 | no | |
| CBOW content | le0_lr0.003 | 0.0877 | 0.0394 | 0.1134 (0.3) | 0.0542 | 4 | 84 | no | |
| CBOW content | le1_lr0.003 | 0.0877 | 0.0393 | 0.1134 (0.3) | 0.0542 | 4 | 84 | no | |
| BERT4Rec-style | dropout0.1_le0_lr0.001 | 0.0790 | 0.0352 | 0.1140 (0.3) | 0.0543 | 24 | 104 | no | |
| BERT4Rec-style | dropout0.1_le1_lr0.001 | 0.0790 | 0.0352 | 0.1140 (0.3) | 0.0542 | 24 | 104 | no | F |
| BERT4Rec-style | dropout0.2_le0_lr0.001 | 0.0803 | 0.0358 | 0.1123 (0.3) | 0.0538 | 20 | 100 | no | |
| BERT4Rec-style | dropout0.2_le1_lr0.001 | 0.0803 | 0.0358 | 0.1124 (0.3) | 0.0538 | 20 | 100 | no | A |
| BERT4Rec-style | dropout0.1_le0_lr0.003 | 0.0759 | 0.0332 | 0.1114 (0.3) | 0.0532 | 48 | 128 | no | |
| BERT4Rec-style | dropout0.1_le1_lr0.003 | 0.0748 | 0.0331 | 0.1116 (0.3) | 0.0532 | 48 | 128 | no | |
| BERT4Rec-style | dropout0.2_le0_lr0.003 | 0.0756 | 0.0337 | 0.1119 (0.3) | 0.0536 | 48 | 128 | no | |
| BERT4Rec-style | dropout0.2_le1_lr0.003 | 0.0759 | 0.0337 | 0.1119 (0.3) | 0.0536 | 48 | 128 | no | |
| SASRec-style | dropout0.1_le0_lr0.001 | 0.0759 | 0.0316 | 0.1121 (0.6) | 0.0539 | 28 | 108 | no | |
| SASRec-style | dropout0.1_le1_lr0.001 | 0.0758 | 0.0315 | 0.1121 (0.6) | 0.0539 | 28 | 108 | no | |
| SASRec-style | dropout0.2_le0_lr0.001 | 0.0781 | 0.0332 | 0.1129 (0.6) | 0.0537 | 8 | 88 | no | A, F |
| SASRec-style | dropout0.2_le1_lr0.001 | 0.0780 | 0.0332 | 0.1129 (0.6) | 0.0537 | 8 | 88 | no | |
| SASRec-style | dropout0.1_le0_lr0.003 | 0.0748 | 0.0306 | 0.1119 (0.6) | 0.0535 | 12 | 92 | no | |
| SASRec-style | dropout0.1_le1_lr0.003 | 0.0747 | 0.0306 | 0.1119 (0.6) | 0.0535 | 12 | 92 | no | |
| SASRec-style | dropout0.2_le0_lr0.003 | 0.0755 | 0.0313 | 0.1120 (0.6) | 0.0534 | 12 | 92 | no | |
| SASRec-style | dropout0.2_le1_lr0.003 | 0.0754 | 0.0313 | 0.1119 (0.6) | 0.0534 | 12 | 92 | no | |
| Mult-VAE | beta_cap0.2_lr0.001 | 0.0815 | 0.0369 | 0.1109 (0.3) | 0.0532 | 92 | 172 | no | |
| Mult-VAE | beta_cap0.5_lr0.001 | 0.0894 | 0.0406 | 0.1119 (0.3) | 0.0533 | 296 | 376 | no | F |
| Mult-VAE | beta_cap0.2_lr0.003 | 0.0859 | 0.0386 | 0.1113 (0.3) | 0.0531 | 212 | 292 | no | |
| Mult-VAE | beta_cap0.5_lr0.003 | 0.0933 | 0.0421 | 0.1117 (0.3) | 0.0534 | 320 | 400 | no | A |

### D4. Seed-mean panels

Test R@20 / N@20, mean and sample standard deviation over seeds 2024, 2025 and 2026. A star marks a cell where SCOPE (the deployed head in panel A, SCOPE-v1 in panels B and C) is significantly better than the arm in the seed-averaged test after Holm (panels A and B: one family of 40 tests; panel C: 20 tests); no arm is significantly better than SCOPE in any cell. Standalone-selected configurations: Baby CBOW random le1_lr0.001, CBOW content le0_lr0.001, BERT4Rec-style dropout0.2_le0_lr0.001, SASRec-style dropout0.2_le0_lr0.001, Mult-VAE beta_cap0.5_lr0.003; Sports CBOW random le0_lr0.001, CBOW content le0_lr0.001, BERT4Rec-style dropout0.2_le1_lr0.001, SASRec-style dropout0.2_le0_lr0.001, Mult-VAE beta_cap0.5_lr0.003. Fused-selected configurations that differ: Baby CBOW random le0_lr0.003, BERT4Rec-style dropout0.1_le1_lr0.003, SASRec-style dropout0.1_le1_lr0.003, Mult-VAE beta_cap0.5_lr0.001; Sports CBOW random le0_lr0.003, BERT4Rec-style dropout0.1_le1_lr0.001, Mult-VAE beta_cap0.5_lr0.001.

(A) Standalone

| Model | Baby R@20 | Baby N@20 | Sports R@20 | Sports N@20 |
|---|---|---|---|---|
| SCOPE head (deployed) | 0.0836 ± 0.0022 | 0.0370 ± 0.0014 | 0.0958 ± 0.0011 | 0.0434 ± 0.0003 |
| SCOPE head re-run, 60-item lists | 0.0851 ± 0.0022 | 0.0380 ± 0.0007 | 0.0956 ± 0.0022 | 0.0433 ± 0.0012 |
| SCOPE head re-run, full lists | 0.0844 ± 0.0007 | 0.0376 ± 0.0007 | 0.0941 ± 0.0008 | 0.0426 ± 0.0004 |
| CBOW random | 0.0695 ± 0.0001 * | 0.0314 ± 0.0003 * | 0.0789 ± 0.0003 * | 0.0367 ± 0.0001 * |
| CBOW content | 0.0841 ± 0.0009 | 0.0377 ± 0.0005 | 0.0940 ± 0.0015 | 0.0424 ± 0.0010 * |
| BERT4Rec-style set Transformer | 0.0776 ± 0.0010 * | 0.0341 ± 0.0003 * | 0.0777 ± 0.0010 * | 0.0348 ± 0.0007 * |
| SASRec-style set Transformer | 0.0640 ± 0.0008 * | 0.0261 ± 0.0002 * | 0.0797 ± 0.0003 * | 0.0337 ± 0.0004 * |
| Mult-VAE | 0.0850 ± 0.0016 | 0.0378 ± 0.0003 | 0.0911 ± 0.0005 * | 0.0425 ± 0.0001 |

(B) Fused with the base, standalone-selected configurations

| Model | Baby R@20 | Baby N@20 | Sports R@20 | Sports N@20 |
|---|---|---|---|---|
| SCOPE-v1 (deployed head + gamma base) | 0.1014 ± 0.0006 | 0.0472 ± 0.0005 | 0.1168 ± 0.0006 | 0.0555 ± 0.0001 |
| SCOPE head re-run, 60-item lists, fused | 0.1019 ± 0.0015 | 0.0475 ± 0.0005 | 0.1166 ± 0.0016 | 0.0554 ± 0.0004 |
| SCOPE head re-run, full lists, fused | 0.1004 ± 0.0011 | 0.0471 ± 0.0001 | 0.1169 ± 0.0008 | 0.0554 ± 0.0002 |
| CBOW random | 0.0951 ± 0.0005 * | 0.0453 ± 0.0001 * | 0.1115 ± 0.0003 * | 0.0534 ± 0.0001 * |
| CBOW content | 0.1009 ± 0.0006 | 0.0473 ± 0.0001 | 0.1167 ± 0.0013 | 0.0554 ± 0.0007 |
| BERT4Rec-style set Transformer | 0.0990 ± 0.0014 * | 0.0460 ± 0.0005 * | 0.1156 ± 0.0001 | 0.0545 ± 0.0001 * |
| SASRec-style set Transformer | 0.0947 ± 0.0004 * | 0.0450 ± 0.0002 * | 0.1134 ± 0.0002 * | 0.0539 ± 0.0001 * |
| Mult-VAE | 0.0978 ± 0.0006 * | 0.0459 ± 0.0002 * | 0.1129 ± 0.0002 * | 0.0538 ± 0.0001 * |

(C) Fused with the base, fused-selected configurations

| Model | Baby R@20 | Baby N@20 | Sports R@20 | Sports N@20 |
|---|---|---|---|---|
| SCOPE-v1 (deployed head + gamma base) | 0.1014 ± 0.0006 | 0.0472 ± 0.0005 | 0.1168 ± 0.0006 | 0.0555 ± 0.0001 |
| CBOW random | 0.0948 ± 0.0005 * | 0.0452 ± 0.0002 * | 0.1110 ± 0.0008 * | 0.0534 ± 0.0003 * |
| CBOW content | 0.1009 ± 0.0006 | 0.0473 ± 0.0001 | 0.1167 ± 0.0013 | 0.0554 ± 0.0007 |
| BERT4Rec-style set Transformer | 0.0975 ± 0.0001 * | 0.0456 ± 0.0001 * | 0.1162 ± 0.0006 | 0.0548 ± 0.0002 * |
| SASRec-style set Transformer | 0.0955 ± 0.0004 * | 0.0453 ± 0.0003 * | 0.1134 ± 0.0002 * | 0.0539 ± 0.0001 * |
| Mult-VAE | 0.0980 ± 0.0004 * | 0.0458 ± 0.0001 * | 0.1128 ± 0.0005 * | 0.0536 ± 0.0002 * |

Won / tied / lost by SCOPE: panels (A) + (B) 29 / 11 / 0 (family of 40); panel (C) 15 / 5 / 0 (family of 20). Every tie is with CBOW content (Baby: all four cells; Sports: standalone R@20 and both fused cells) or with Mult-VAE (Baby: both standalone cells; Sports: standalone N@20), plus the BERT4Rec-style fused R@20 on Sports in both panels.

### D5. Per-seed runs, Baby

Test metrics of every run of a selected configuration, one row per seed. "Last ep" is the epoch at which early stopping ended the run (best epoch plus 80); no run hit the epoch cap. Grid-only runs that were trained but not selected are in Table D2; the G7 rows are the CBOW-gate runs of Section D11 (pre-registered gamma grid without extension).

| Model | Configuration | Seed | Best ep | Last ep | gamma | R@20 | N@20 | R@20 fused | N@20 fused |
|---|---|---|---|---|---|---|---|---|---|
| SCOPE head (deployed) | -- | 2024 | 24 | -- | 0.3 | 0.0857 | 0.0377 | 0.1017 | 0.0473 |
| SCOPE head (deployed) | -- | 2025 | 24 | -- | 0.3 | 0.0813 | 0.0354 | 0.1008 | 0.0467 |
| SCOPE head (deployed) | -- | 2026 | 28 | -- | 0.3 | 0.0838 | 0.0381 | 0.1017 | 0.0477 |
| G7 CBOW random, lambda_E = 0 | le0_lr0.003 | 2024 | 24 | 104 | 1 | 0.0701 | 0.0308 | 0.0953 | 0.0454 |
| G7 CBOW random, lambda_E = 0 | le0_lr0.003 | 2025 | 16 | 96 | 1 | 0.0700 | 0.0314 | 0.0949 | 0.0453 |
| G7 CBOW random, lambda_E = 0 | le0_lr0.003 | 2026 | 16 | 96 | 1.5 | 0.0702 | 0.0320 | 0.0943 | 0.0451 |
| G7 CBOW random, lambda_E = 1 | le1_lr0.003 | 2024 | 24 | 104 | 1 | 0.0701 | 0.0308 | 0.0953 | 0.0454 |
| G7 CBOW random, lambda_E = 1 | le1_lr0.003 | 2025 | 16 | 96 | 1 | 0.0700 | 0.0314 | 0.0949 | 0.0453 |
| G7 CBOW random, lambda_E = 1 | le1_lr0.003 | 2026 | 16 | 96 | 1.5 | 0.0702 | 0.0320 | 0.0943 | 0.0451 |
| G7 CBOW content, lambda_E = 0 | le0_lr0.003 | 2024 | 8 | 88 | 0.6 | 0.0759 | 0.0336 | 0.0985 | 0.0468 |
| G7 CBOW content, lambda_E = 0 | le0_lr0.003 | 2025 | 8 | 88 | 0.6 | 0.0767 | 0.0337 | 0.0973 | 0.0463 |
| G7 CBOW content, lambda_E = 0 | le0_lr0.003 | 2026 | 4 | 84 | 0.3 | 0.0753 | 0.0338 | 0.0985 | 0.0464 |
| G7 CBOW content, lambda_E = 1 | le1_lr0.003 | 2024 | 8 | 88 | 0.6 | 0.0759 | 0.0337 | 0.0985 | 0.0469 |
| G7 CBOW content, lambda_E = 1 | le1_lr0.003 | 2025 | 8 | 88 | 0.6 | 0.0766 | 0.0337 | 0.0972 | 0.0463 |
| G7 CBOW content, lambda_E = 1 | le1_lr0.003 | 2026 | 4 | 84 | 0.6 | 0.0756 | 0.0339 | 0.0983 | 0.0464 |
| CBOW random (standalone-selected) | le1_lr0.001 | 2024 | 48 | 128 | 1 | 0.0694 | 0.0311 | 0.0951 | 0.0452 |
| CBOW random (standalone-selected) | le1_lr0.001 | 2025 | 52 | 132 | 1 | 0.0696 | 0.0316 | 0.0955 | 0.0455 |
| CBOW random (standalone-selected) | le1_lr0.001 | 2026 | 56 | 136 | 1 | 0.0695 | 0.0315 | 0.0946 | 0.0453 |
| CBOW content (standalone- and fused-selected) | le0_lr0.001 | 2024 | 4 | 84 | 0.3 | 0.0849 | 0.0381 | 0.1014 | 0.0474 |
| CBOW content (standalone- and fused-selected) | le0_lr0.001 | 2025 | 4 | 84 | 0.3 | 0.0831 | 0.0371 | 0.1011 | 0.0473 |
| CBOW content (standalone- and fused-selected) | le0_lr0.001 | 2026 | 4 | 84 | 0.3 | 0.0842 | 0.0379 | 0.1003 | 0.0471 |
| BERT4Rec-style (standalone-selected) | dropout0.2_le0_lr0.001 | 2024 | 16 | 96 | 0.3 | 0.0774 | 0.0340 | 0.0981 | 0.0456 |
| BERT4Rec-style (standalone-selected) | dropout0.2_le0_lr0.001 | 2025 | 24 | 104 | 0.3 | 0.0768 | 0.0338 | 0.0983 | 0.0460 |
| BERT4Rec-style (standalone-selected) | dropout0.2_le0_lr0.001 | 2026 | 20 | 100 | 0.3 | 0.0787 | 0.0345 | 0.1007 | 0.0466 |
| SASRec-style (standalone-selected) | dropout0.2_le0_lr0.001 | 2024 | 4 | 84 | 1 | 0.0632 | 0.0260 | 0.0942 | 0.0449 |
| SASRec-style (standalone-selected) | dropout0.2_le0_lr0.001 | 2025 | 12 | 92 | 0.6 | 0.0638 | 0.0260 | 0.0948 | 0.0453 |
| SASRec-style (standalone-selected) | dropout0.2_le0_lr0.001 | 2026 | 4 | 84 | 0.6 | 0.0649 | 0.0264 | 0.0951 | 0.0448 |
| Mult-VAE (standalone-selected) | beta_cap0.5_lr0.003 | 2024 | 264 | 344 | 0.3 | 0.0869 | 0.0380 | 0.0973 | 0.0456 |
| Mult-VAE (standalone-selected) | beta_cap0.5_lr0.003 | 2025 | 384 | 464 | 0.3 | 0.0843 | 0.0375 | 0.0976 | 0.0461 |
| Mult-VAE (standalone-selected) | beta_cap0.5_lr0.003 | 2026 | 272 | 352 | 0.3 | 0.0839 | 0.0380 | 0.0985 | 0.0459 |
| CBOW random (fused-selected) | le0_lr0.003 | 2024 | 24 | 104 | 1 | 0.0701 | 0.0308 | 0.0953 | 0.0454 |
| CBOW random (fused-selected) | le0_lr0.003 | 2025 | 16 | 96 | 1 | 0.0700 | 0.0314 | 0.0949 | 0.0453 |
| CBOW random (fused-selected) | le0_lr0.003 | 2026 | 16 | 96 | 1.5 | 0.0702 | 0.0320 | 0.0943 | 0.0451 |
| BERT4Rec-style (fused-selected) | dropout0.1_le1_lr0.003 | 2024 | 56 | 136 | 0.3 | 0.0747 | 0.0328 | 0.0976 | 0.0455 |
| BERT4Rec-style (fused-selected) | dropout0.1_le1_lr0.003 | 2025 | 132 | 212 | 0.3 | 0.0732 | 0.0318 | 0.0975 | 0.0457 |
| BERT4Rec-style (fused-selected) | dropout0.1_le1_lr0.003 | 2026 | 44 | 124 | 0.3 | 0.0736 | 0.0321 | 0.0974 | 0.0457 |
| SASRec-style (fused-selected) | dropout0.1_le1_lr0.003 | 2024 | 24 | 104 | 1 | 0.0616 | 0.0253 | 0.0958 | 0.0455 |
| SASRec-style (fused-selected) | dropout0.1_le1_lr0.003 | 2025 | 16 | 96 | 0.6 | 0.0638 | 0.0252 | 0.0956 | 0.0456 |
| SASRec-style (fused-selected) | dropout0.1_le1_lr0.003 | 2026 | 4 | 84 | 1.5 | 0.0618 | 0.0252 | 0.0950 | 0.0449 |
| Mult-VAE (fused-selected) | beta_cap0.5_lr0.001 | 2024 | 392 | 472 | 0.3 | 0.0828 | 0.0370 | 0.0976 | 0.0458 |
| Mult-VAE (fused-selected) | beta_cap0.5_lr0.001 | 2025 | 284 | 364 | 0.3 | 0.0833 | 0.0374 | 0.0981 | 0.0458 |
| Mult-VAE (fused-selected) | beta_cap0.5_lr0.001 | 2026 | 256 | 336 | 0.3 | 0.0844 | 0.0371 | 0.0984 | 0.0459 |
| SCOPE head re-run, 60-item lists | le1_lr0.003_cap60 | 2024 | 24 | 104 | 0.3 | 0.0868 | 0.0384 | 0.1030 | 0.0477 |
| SCOPE head re-run, 60-item lists | le1_lr0.003_cap60 | 2025 | 28 | 108 | 0.3 | 0.0859 | 0.0384 | 0.1001 | 0.0470 |
| SCOPE head re-run, 60-item lists | le1_lr0.003_cap60 | 2026 | 28 | 108 | 0.3 | 0.0826 | 0.0372 | 0.1025 | 0.0478 |
| SCOPE head re-run, full lists | le1_lr0.003_fulllists | 2024 | 28 | 108 | 0.3 | 0.0852 | 0.0384 | 0.1010 | 0.0472 |
| SCOPE head re-run, full lists | le1_lr0.003_fulllists | 2025 | 32 | 112 | 0.3 | 0.0838 | 0.0371 | 0.0991 | 0.0470 |
| SCOPE head re-run, full lists | le1_lr0.003_fulllists | 2026 | 28 | 108 | 0.3 | 0.0843 | 0.0375 | 0.1011 | 0.0472 |

### D6. Per-seed runs, Sports

| Model | Configuration | Seed | Best ep | Last ep | gamma | R@20 | N@20 | R@20 fused | N@20 fused |
|---|---|---|---|---|---|---|---|---|---|
| SCOPE head (deployed) | -- | 2024 | 16 | -- | 0.3 | 0.0946 | 0.0432 | 0.1160 | 0.0554 |
| SCOPE head (deployed) | -- | 2025 | 16 | -- | 0.3 | 0.0962 | 0.0434 | 0.1172 | 0.0556 |
| SCOPE head (deployed) | -- | 2026 | 20 | -- | 0.3 | 0.0967 | 0.0438 | 0.1170 | 0.0555 |
| G7 CBOW random, lambda_E = 0 | le0_lr0.003 | 2024 | 16 | 96 | 1 | 0.0777 | 0.0357 | 0.1100 | 0.0531 |
| G7 CBOW random, lambda_E = 0 | le0_lr0.003 | 2025 | 16 | 96 | 0.6 | 0.0782 | 0.0362 | 0.1116 | 0.0535 |
| G7 CBOW random, lambda_E = 0 | le0_lr0.003 | 2026 | 16 | 96 | 0.6 | 0.0775 | 0.0362 | 0.1113 | 0.0535 |
| G7 CBOW random, lambda_E = 1 | le1_lr0.003 | 2024 | 16 | 96 | 1 | 0.0777 | 0.0357 | 0.1100 | 0.0531 |
| G7 CBOW random, lambda_E = 1 | le1_lr0.003 | 2025 | 16 | 96 | 0.6 | 0.0781 | 0.0362 | 0.1116 | 0.0535 |
| G7 CBOW random, lambda_E = 1 | le1_lr0.003 | 2026 | 16 | 96 | 0.6 | 0.0775 | 0.0362 | 0.1113 | 0.0535 |
| G7 CBOW content, lambda_E = 0 | le0_lr0.003 | 2024 | 4 | 84 | 0.3 | 0.0882 | 0.0403 | 0.1143 | 0.0548 |
| G7 CBOW content, lambda_E = 0 | le0_lr0.003 | 2025 | 4 | 84 | 0.3 | 0.0888 | 0.0407 | 0.1155 | 0.0550 |
| G7 CBOW content, lambda_E = 0 | le0_lr0.003 | 2026 | 4 | 84 | 0.3 | 0.0895 | 0.0407 | 0.1161 | 0.0551 |
| G7 CBOW content, lambda_E = 1 | le1_lr0.003 | 2024 | 4 | 84 | 0.3 | 0.0883 | 0.0403 | 0.1143 | 0.0548 |
| G7 CBOW content, lambda_E = 1 | le1_lr0.003 | 2025 | 4 | 84 | 0.3 | 0.0888 | 0.0407 | 0.1155 | 0.0551 |
| G7 CBOW content, lambda_E = 1 | le1_lr0.003 | 2026 | 4 | 84 | 0.3 | 0.0893 | 0.0406 | 0.1161 | 0.0551 |
| CBOW random (standalone-selected) | le0_lr0.001 | 2024 | 44 | 124 | 0.6 | 0.0785 | 0.0367 | 0.1112 | 0.0534 |
| CBOW random (standalone-selected) | le0_lr0.001 | 2025 | 44 | 124 | 1 | 0.0791 | 0.0366 | 0.1117 | 0.0535 |
| CBOW random (standalone-selected) | le0_lr0.001 | 2026 | 40 | 120 | 0.6 | 0.0789 | 0.0368 | 0.1115 | 0.0535 |
| CBOW content (standalone- and fused-selected) | le0_lr0.001 | 2024 | 0 | 80 | 0.3 | 0.0925 | 0.0412 | 0.1153 | 0.0546 |
| CBOW content (standalone- and fused-selected) | le0_lr0.001 | 2025 | 4 | 84 | 0.3 | 0.0941 | 0.0429 | 0.1173 | 0.0558 |
| CBOW content (standalone- and fused-selected) | le0_lr0.001 | 2026 | 4 | 84 | 0.3 | 0.0954 | 0.0431 | 0.1176 | 0.0559 |
| BERT4Rec-style (standalone-selected) | dropout0.2_le1_lr0.001 | 2024 | 20 | 100 | 0.3 | 0.0774 | 0.0345 | 0.1156 | 0.0545 |
| BERT4Rec-style (standalone-selected) | dropout0.2_le1_lr0.001 | 2025 | 24 | 104 | 0.3 | 0.0787 | 0.0356 | 0.1157 | 0.0546 |
| BERT4Rec-style (standalone-selected) | dropout0.2_le1_lr0.001 | 2026 | 20 | 100 | 0.3 | 0.0768 | 0.0343 | 0.1154 | 0.0544 |
| SASRec-style (standalone- and fused-selected) | dropout0.2_le0_lr0.001 | 2024 | 8 | 88 | 0.6 | 0.0801 | 0.0339 | 0.1134 | 0.0538 |
| SASRec-style (standalone- and fused-selected) | dropout0.2_le0_lr0.001 | 2025 | 8 | 88 | 0.6 | 0.0796 | 0.0339 | 0.1136 | 0.0538 |
| SASRec-style (standalone- and fused-selected) | dropout0.2_le0_lr0.001 | 2026 | 16 | 96 | 0.6 | 0.0795 | 0.0332 | 0.1132 | 0.0540 |
| Mult-VAE (standalone-selected) | beta_cap0.5_lr0.003 | 2024 | 320 | 400 | 0.3 | 0.0913 | 0.0425 | 0.1127 | 0.0537 |
| Mult-VAE (standalone-selected) | beta_cap0.5_lr0.003 | 2025 | 300 | 380 | 0.3 | 0.0905 | 0.0424 | 0.1132 | 0.0539 |
| Mult-VAE (standalone-selected) | beta_cap0.5_lr0.003 | 2026 | 296 | 376 | 0.3 | 0.0914 | 0.0425 | 0.1129 | 0.0538 |
| CBOW random (fused-selected) | le0_lr0.003 | 2024 | 16 | 96 | 1 | 0.0777 | 0.0357 | 0.1100 | 0.0531 |
| CBOW random (fused-selected) | le0_lr0.003 | 2025 | 16 | 96 | 0.6 | 0.0782 | 0.0362 | 0.1116 | 0.0535 |
| CBOW random (fused-selected) | le0_lr0.003 | 2026 | 16 | 96 | 0.6 | 0.0775 | 0.0362 | 0.1113 | 0.0535 |
| BERT4Rec-style (fused-selected) | dropout0.1_le1_lr0.001 | 2024 | 24 | 104 | 0.3 | 0.0789 | 0.0356 | 0.1165 | 0.0550 |
| BERT4Rec-style (fused-selected) | dropout0.1_le1_lr0.001 | 2025 | 28 | 108 | 0.3 | 0.0782 | 0.0353 | 0.1166 | 0.0550 |
| BERT4Rec-style (fused-selected) | dropout0.1_le1_lr0.001 | 2026 | 20 | 100 | 0.3 | 0.0776 | 0.0346 | 0.1156 | 0.0546 |
| Mult-VAE (fused-selected) | beta_cap0.5_lr0.001 | 2024 | 296 | 376 | 0.3 | 0.0893 | 0.0406 | 0.1122 | 0.0535 |
| Mult-VAE (fused-selected) | beta_cap0.5_lr0.001 | 2025 | 300 | 380 | 0.3 | 0.0882 | 0.0407 | 0.1131 | 0.0538 |
| Mult-VAE (fused-selected) | beta_cap0.5_lr0.001 | 2026 | 224 | 304 | 0.3 | 0.0880 | 0.0406 | 0.1131 | 0.0535 |
| SCOPE head re-run, 60-item lists | le1_lr0.003_cap60 | 2024 | 20 | 100 | 0.3 | 0.0950 | 0.0432 | 0.1160 | 0.0554 |
| SCOPE head re-run, 60-item lists | le1_lr0.003_cap60 | 2025 | 20 | 100 | 0.3 | 0.0937 | 0.0422 | 0.1154 | 0.0551 |
| SCOPE head re-run, 60-item lists | le1_lr0.003_cap60 | 2026 | 16 | 96 | 0.3 | 0.0981 | 0.0446 | 0.1185 | 0.0558 |
| SCOPE head re-run, full lists | le1_lr0.003_fulllists | 2024 | 16 | 96 | 0.3 | 0.0933 | 0.0425 | 0.1171 | 0.0555 |
| SCOPE head re-run, full lists | le1_lr0.003_fulllists | 2025 | 16 | 96 | 0.3 | 0.0941 | 0.0423 | 0.1176 | 0.0555 |
| SCOPE head re-run, full lists | le1_lr0.003_fulllists | 2026 | 20 | 100 | 0.3 | 0.0950 | 0.0431 | 0.1161 | 0.0552 |

### D7. Seed-averaged tests, primary family (deployed SCOPE minus the standalone-selected arm; Holm over 40 tests)

Delta = SCOPE minus arm; "alone" compares the head alone with the arm alone, "fused" compares SCOPE-v1 with the arm fused with the base. "a better" means SCOPE is significantly better after Holm.

| Dataset | Arm | Setting | Metric | SCOPE | Arm | Delta | 95% CI | p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Baby | CBOW random | alone | R@20 | 0.0836 | 0.0695 | +0.0141 | [+0.0115, +0.0168] | < 0.0002 | < 0.0002 | a better |
| Baby | CBOW random | alone | N@20 | 0.0370 | 0.0314 | +0.0056 | [+0.0044, +0.0069] | < 0.0002 | < 0.0002 | a better |
| Baby | CBOW random | fused | R@20 | 0.1014 | 0.0951 | +0.0064 | [+0.0046, +0.0081] | < 0.0002 | < 0.0002 | a better |
| Baby | CBOW random | fused | N@20 | 0.0472 | 0.0453 | +0.0019 | [+0.0013, +0.0025] | < 0.0002 | < 0.0002 | a better |
| Baby | CBOW content | alone | R@20 | 0.0836 | 0.0841 | -0.0005 | [-0.0027, +0.0018] | 0.677 | 1 | n.s. |
| Baby | CBOW content | alone | N@20 | 0.0370 | 0.0377 | -0.0007 | [-0.0018, +0.0004] | 0.223 | 1 | n.s. |
| Baby | CBOW content | fused | R@20 | 0.1014 | 0.1009 | +0.0005 | [-0.0010, +0.0020] | 0.524 | 1 | n.s. |
| Baby | CBOW content | fused | N@20 | 0.0472 | 0.0473 | -0.0000 | [-0.0005, +0.0005] | 0.927 | 1 | n.s. |
| Baby | BERT4Rec-style | alone | R@20 | 0.0836 | 0.0776 | +0.0060 | [+0.0038, +0.0081] | < 0.0002 | < 0.0002 | a better |
| Baby | BERT4Rec-style | alone | N@20 | 0.0370 | 0.0341 | +0.0030 | [+0.0020, +0.0039] | < 0.0002 | < 0.0002 | a better |
| Baby | BERT4Rec-style | fused | R@20 | 0.1014 | 0.0990 | +0.0024 | [+0.0011, +0.0037] | < 0.0002 | < 0.0002 | a better |
| Baby | BERT4Rec-style | fused | N@20 | 0.0472 | 0.0460 | +0.0012 | [+0.0007, +0.0017] | < 0.0002 | < 0.0002 | a better |
| Baby | SASRec-style | alone | R@20 | 0.0836 | 0.0640 | +0.0196 | [+0.0169, +0.0223] | < 0.0002 | < 0.0002 | a better |
| Baby | SASRec-style | alone | N@20 | 0.0370 | 0.0261 | +0.0109 | [+0.0096, +0.0123] | < 0.0002 | < 0.0002 | a better |
| Baby | SASRec-style | fused | R@20 | 0.1014 | 0.0947 | +0.0067 | [+0.0051, +0.0084] | < 0.0002 | < 0.0002 | a better |
| Baby | SASRec-style | fused | N@20 | 0.0472 | 0.0450 | +0.0022 | [+0.0017, +0.0028] | < 0.0002 | < 0.0002 | a better |
| Baby | Mult-VAE | alone | R@20 | 0.0836 | 0.0850 | -0.0014 | [-0.0042, +0.0013] | 0.321 | 1 | n.s. |
| Baby | Mult-VAE | alone | N@20 | 0.0370 | 0.0378 | -0.0008 | [-0.0021, +0.0006] | 0.245 | 1 | n.s. |
| Baby | Mult-VAE | fused | R@20 | 0.1014 | 0.0978 | +0.0036 | [+0.0019, +0.0053] | < 0.0002 | < 0.0002 | a better |
| Baby | Mult-VAE | fused | N@20 | 0.0472 | 0.0459 | +0.0014 | [+0.0008, +0.0020] | < 0.0002 | < 0.0002 | a better |
| Sports | CBOW random | alone | R@20 | 0.0958 | 0.0789 | +0.0170 | [+0.0153, +0.0187] | < 0.0002 | < 0.0002 | a better |
| Sports | CBOW random | alone | N@20 | 0.0434 | 0.0367 | +0.0067 | [+0.0059, +0.0076] | < 0.0002 | < 0.0002 | a better |
| Sports | CBOW random | fused | R@20 | 0.1168 | 0.1115 | +0.0053 | [+0.0043, +0.0063] | < 0.0002 | < 0.0002 | a better |
| Sports | CBOW random | fused | N@20 | 0.0555 | 0.0534 | +0.0020 | [+0.0017, +0.0024] | < 0.0002 | < 0.0002 | a better |
| Sports | CBOW content | alone | R@20 | 0.0958 | 0.0940 | +0.0019 | [+0.0004, +0.0033] | 0.014 | 0.142 | n.s. |
| Sports | CBOW content | alone | N@20 | 0.0434 | 0.0424 | +0.0010 | [+0.0003, +0.0018] | 0.004 | 0.048 | a better |
| Sports | CBOW content | fused | R@20 | 0.1168 | 0.1167 | +0.0000 | [-0.0008, +0.0009] | 0.983 | 1 | n.s. |
| Sports | CBOW content | fused | N@20 | 0.0555 | 0.0554 | +0.0001 | [-0.0002, +0.0003] | 0.735 | 1 | n.s. |
| Sports | BERT4Rec-style | alone | R@20 | 0.0958 | 0.0777 | +0.0182 | [+0.0164, +0.0201] | < 0.0002 | < 0.0002 | a better |
| Sports | BERT4Rec-style | alone | N@20 | 0.0434 | 0.0348 | +0.0086 | [+0.0077, +0.0096] | < 0.0002 | < 0.0002 | a better |
| Sports | BERT4Rec-style | fused | R@20 | 0.1168 | 0.1156 | +0.0012 | [+0.0003, +0.0021] | 0.011 | 0.125 | n.s. |
| Sports | BERT4Rec-style | fused | N@20 | 0.0555 | 0.0545 | +0.0009 | [+0.0006, +0.0013] | < 0.0002 | < 0.0002 | a better |
| Sports | SASRec-style | alone | R@20 | 0.0958 | 0.0797 | +0.0161 | [+0.0143, +0.0180] | < 0.0002 | < 0.0002 | a better |
| Sports | SASRec-style | alone | N@20 | 0.0434 | 0.0337 | +0.0098 | [+0.0088, +0.0107] | < 0.0002 | < 0.0002 | a better |
| Sports | SASRec-style | fused | R@20 | 0.1168 | 0.1134 | +0.0034 | [+0.0024, +0.0044] | < 0.0002 | < 0.0002 | a better |
| Sports | SASRec-style | fused | N@20 | 0.0555 | 0.0539 | +0.0016 | [+0.0012, +0.0019] | < 0.0002 | < 0.0002 | a better |
| Sports | Mult-VAE | alone | R@20 | 0.0958 | 0.0911 | +0.0048 | [+0.0028, +0.0068] | < 0.0002 | < 0.0002 | a better |
| Sports | Mult-VAE | alone | N@20 | 0.0434 | 0.0425 | +0.0010 | [-0.0001, +0.0020] | 0.071 | 0.637 | n.s. |
| Sports | Mult-VAE | fused | R@20 | 0.1168 | 0.1129 | +0.0038 | [+0.0027, +0.0049] | < 0.0002 | < 0.0002 | a better |
| Sports | Mult-VAE | fused | N@20 | 0.0555 | 0.0538 | +0.0017 | [+0.0013, +0.0021] | < 0.0002 | < 0.0002 | a better |

### D8. Seed-averaged tests, fused-selected panel (SCOPE-v1 minus the fused-selected arm; Holm over 20 tests)

| Dataset | Arm | Metric | SCOPE-v1 | Arm | Delta | 95% CI | p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Baby | CBOW random | R@20 | 0.1014 | 0.0948 | +0.0066 | [+0.0048, +0.0083] | < 0.0002 | < 0.0002 | a better |
| Baby | CBOW random | N@20 | 0.0472 | 0.0452 | +0.0020 | [+0.0014, +0.0026] | < 0.0002 | < 0.0002 | a better |
| Baby | CBOW content | R@20 | 0.1014 | 0.1009 | +0.0005 | [-0.0010, +0.0020] | 0.524 | 1 | n.s. |
| Baby | CBOW content | N@20 | 0.0472 | 0.0473 | -0.0000 | [-0.0005, +0.0005] | 0.927 | 1 | n.s. |
| Baby | BERT4Rec-style | R@20 | 0.1014 | 0.0975 | +0.0039 | [+0.0025, +0.0052] | < 0.0002 | < 0.0002 | a better |
| Baby | BERT4Rec-style | N@20 | 0.0472 | 0.0456 | +0.0016 | [+0.0011, +0.0021] | < 0.0002 | < 0.0002 | a better |
| Baby | SASRec-style | R@20 | 0.1014 | 0.0955 | +0.0059 | [+0.0043, +0.0076] | < 0.0002 | < 0.0002 | a better |
| Baby | SASRec-style | N@20 | 0.0472 | 0.0453 | +0.0019 | [+0.0013, +0.0025] | < 0.0002 | < 0.0002 | a better |
| Baby | Mult-VAE | R@20 | 0.1014 | 0.0980 | +0.0034 | [+0.0017, +0.0051] | < 0.0002 | < 0.0002 | a better |
| Baby | Mult-VAE | N@20 | 0.0472 | 0.0458 | +0.0014 | [+0.0008, +0.0020] | < 0.0002 | < 0.0002 | a better |
| Sports | CBOW random | R@20 | 0.1168 | 0.1110 | +0.0058 | [+0.0048, +0.0068] | < 0.0002 | < 0.0002 | a better |
| Sports | CBOW random | N@20 | 0.0555 | 0.0534 | +0.0021 | [+0.0017, +0.0025] | < 0.0002 | < 0.0002 | a better |
| Sports | CBOW content | R@20 | 0.1168 | 0.1167 | +0.0000 | [-0.0008, +0.0009] | 0.983 | 1 | n.s. |
| Sports | CBOW content | N@20 | 0.0555 | 0.0554 | +0.0001 | [-0.0002, +0.0003] | 0.735 | 1 | n.s. |
| Sports | BERT4Rec-style | R@20 | 0.1168 | 0.1162 | +0.0005 | [-0.0004, +0.0015] | 0.268 | 1 | n.s. |
| Sports | BERT4Rec-style | N@20 | 0.0555 | 0.0548 | +0.0006 | [+0.0003, +0.0010] | 0.0006 | 0.004 | a better |
| Sports | SASRec-style | R@20 | 0.1168 | 0.1134 | +0.0034 | [+0.0024, +0.0044] | < 0.0002 | < 0.0002 | a better |
| Sports | SASRec-style | N@20 | 0.0555 | 0.0539 | +0.0016 | [+0.0012, +0.0019] | < 0.0002 | < 0.0002 | a better |
| Sports | Mult-VAE | R@20 | 0.1168 | 0.1128 | +0.0040 | [+0.0029, +0.0050] | < 0.0002 | < 0.0002 | a better |
| Sports | Mult-VAE | N@20 | 0.0555 | 0.0536 | +0.0019 | [+0.0015, +0.0023] | < 0.0002 | < 0.0002 | a better |

### D9. Seed-averaged tests, encoder-only family (shared-trainer SCOPE head with full lists minus the standalone-selected arm; Holm over 40 tests)

This family removes the two training differences between the deployed head and the arms (60-item lists and the trainer instance), so that SCOPE and each arm differ only in the encoder.

| Dataset | Arm | Setting | Metric | SCOPE re-run | Arm | Delta | 95% CI | p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Baby | CBOW random | alone | R@20 | 0.0844 | 0.0695 | +0.0149 | [+0.0125, +0.0174] | < 0.0002 | < 0.0002 | a better |
| Baby | CBOW random | alone | N@20 | 0.0376 | 0.0314 | +0.0062 | [+0.0051, +0.0074] | < 0.0002 | < 0.0002 | a better |
| Baby | CBOW random | fused | R@20 | 0.1004 | 0.0951 | +0.0054 | [+0.0037, +0.0071] | < 0.0002 | < 0.0002 | a better |
| Baby | CBOW random | fused | N@20 | 0.0471 | 0.0453 | +0.0018 | [+0.0012, +0.0024] | < 0.0002 | < 0.0002 | a better |
| Baby | CBOW content | alone | R@20 | 0.0844 | 0.0841 | +0.0003 | [-0.0017, +0.0024] | 0.744 | 1 | n.s. |
| Baby | CBOW content | alone | N@20 | 0.0376 | 0.0377 | -0.0001 | [-0.0010, +0.0009] | 0.867 | 1 | n.s. |
| Baby | CBOW content | fused | R@20 | 0.1004 | 0.1009 | -0.0005 | [-0.0019, +0.0009] | 0.474 | 1 | n.s. |
| Baby | CBOW content | fused | N@20 | 0.0471 | 0.0473 | -0.0001 | [-0.0006, +0.0003] | 0.557 | 1 | n.s. |
| Baby | BERT4Rec-style | alone | R@20 | 0.0844 | 0.0776 | +0.0068 | [+0.0045, +0.0091] | < 0.0002 | < 0.0002 | a better |
| Baby | BERT4Rec-style | alone | N@20 | 0.0376 | 0.0341 | +0.0036 | [+0.0025, +0.0046] | < 0.0002 | < 0.0002 | a better |
| Baby | BERT4Rec-style | fused | R@20 | 0.1004 | 0.0990 | +0.0014 | [+0.0000, +0.0028] | 0.041 | 0.494 | n.s. |
| Baby | BERT4Rec-style | fused | N@20 | 0.0471 | 0.0460 | +0.0011 | [+0.0006, +0.0016] | < 0.0002 | < 0.0002 | a better |
| Baby | SASRec-style | alone | R@20 | 0.0844 | 0.0640 | +0.0204 | [+0.0178, +0.0231] | < 0.0002 | < 0.0002 | a better |
| Baby | SASRec-style | alone | N@20 | 0.0376 | 0.0261 | +0.0115 | [+0.0102, +0.0129] | < 0.0002 | < 0.0002 | a better |
| Baby | SASRec-style | fused | R@20 | 0.1004 | 0.0947 | +0.0057 | [+0.0041, +0.0074] | < 0.0002 | < 0.0002 | a better |
| Baby | SASRec-style | fused | N@20 | 0.0471 | 0.0450 | +0.0021 | [+0.0016, +0.0027] | < 0.0002 | < 0.0002 | a better |
| Baby | Mult-VAE | alone | R@20 | 0.0844 | 0.0850 | -0.0006 | [-0.0033, +0.0021] | 0.684 | 1 | n.s. |
| Baby | Mult-VAE | alone | N@20 | 0.0376 | 0.0378 | -0.0002 | [-0.0015, +0.0011] | 0.760 | 1 | n.s. |
| Baby | Mult-VAE | fused | R@20 | 0.1004 | 0.0978 | +0.0026 | [+0.0009, +0.0043] | 0.001 | 0.021 | a better |
| Baby | Mult-VAE | fused | N@20 | 0.0471 | 0.0459 | +0.0013 | [+0.0007, +0.0019] | 0.0002 | 0.003 | a better |
| Sports | CBOW random | alone | R@20 | 0.0941 | 0.0789 | +0.0153 | [+0.0136, +0.0170] | < 0.0002 | < 0.0002 | a better |
| Sports | CBOW random | alone | N@20 | 0.0426 | 0.0367 | +0.0059 | [+0.0051, +0.0067] | < 0.0002 | < 0.0002 | a better |
| Sports | CBOW random | fused | R@20 | 0.1169 | 0.1115 | +0.0055 | [+0.0044, +0.0065] | < 0.0002 | < 0.0002 | a better |
| Sports | CBOW random | fused | N@20 | 0.0554 | 0.0534 | +0.0020 | [+0.0016, +0.0023] | < 0.0002 | < 0.0002 | a better |
| Sports | CBOW content | alone | R@20 | 0.0941 | 0.0940 | +0.0001 | [-0.0013, +0.0016] | 0.856 | 1 | n.s. |
| Sports | CBOW content | alone | N@20 | 0.0426 | 0.0424 | +0.0002 | [-0.0005, +0.0009] | 0.558 | 1 | n.s. |
| Sports | CBOW content | fused | R@20 | 0.1169 | 0.1167 | +0.0002 | [-0.0006, +0.0010] | 0.667 | 1 | n.s. |
| Sports | CBOW content | fused | N@20 | 0.0554 | 0.0554 | -0.0000 | [-0.0003, +0.0003] | 0.903 | 1 | n.s. |
| Sports | BERT4Rec-style | alone | R@20 | 0.0941 | 0.0777 | +0.0165 | [+0.0147, +0.0184] | < 0.0002 | < 0.0002 | a better |
| Sports | BERT4Rec-style | alone | N@20 | 0.0426 | 0.0348 | +0.0078 | [+0.0068, +0.0088] | < 0.0002 | < 0.0002 | a better |
| Sports | BERT4Rec-style | fused | R@20 | 0.1169 | 0.1156 | +0.0014 | [+0.0004, +0.0023] | 0.004 | 0.052 | n.s. |
| Sports | BERT4Rec-style | fused | N@20 | 0.0554 | 0.0545 | +0.0009 | [+0.0005, +0.0012] | < 0.0002 | < 0.0002 | a better |
| Sports | SASRec-style | alone | R@20 | 0.0941 | 0.0797 | +0.0144 | [+0.0126, +0.0163] | < 0.0002 | < 0.0002 | a better |
| Sports | SASRec-style | alone | N@20 | 0.0426 | 0.0337 | +0.0089 | [+0.0080, +0.0099] | < 0.0002 | < 0.0002 | a better |
| Sports | SASRec-style | fused | R@20 | 0.1169 | 0.1134 | +0.0036 | [+0.0026, +0.0046] | < 0.0002 | < 0.0002 | a better |
| Sports | SASRec-style | fused | N@20 | 0.0554 | 0.0539 | +0.0015 | [+0.0012, +0.0019] | < 0.0002 | < 0.0002 | a better |
| Sports | Mult-VAE | alone | R@20 | 0.0941 | 0.0911 | +0.0031 | [+0.0011, +0.0051] | 0.003 | 0.039 | a better |
| Sports | Mult-VAE | alone | N@20 | 0.0426 | 0.0425 | +0.0001 | [-0.0009, +0.0012] | 0.794 | 1 | n.s. |
| Sports | Mult-VAE | fused | R@20 | 0.1169 | 0.1129 | +0.0040 | [+0.0029, +0.0051] | < 0.0002 | < 0.0002 | a better |
| Sports | Mult-VAE | fused | N@20 | 0.0554 | 0.0538 | +0.0016 | [+0.0012, +0.0020] | < 0.0002 | < 0.0002 | a better |

Won / tied / lost by the re-run head: 27 / 13 / 0. Compared with the primary family, the BERT4Rec-style fused R@20 on Baby (Holm p 0.494) and on Sports (0.052) and the Mult-VAE standalone R@20 on Sports (0.039, a win here) change side of the threshold; both families are reported.

### D10. Reproduction check of the head (shared-trainer re-runs minus the deployed head; Holm over 16 tests)

| Dataset | Re-run | Setting | Metric | Re-run | Deployed | Delta | 95% CI | p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Baby | 60-item lists | alone | R@20 | 0.0851 | 0.0836 | +0.0015 | [-0.0002, +0.0032] | 0.083 | 1 | n.s. |
| Baby | 60-item lists | alone | N@20 | 0.0380 | 0.0370 | +0.0010 | [+0.0003, +0.0016] | 0.008 | 0.123 | n.s. |
| Baby | 60-item lists | fused | R@20 | 0.1019 | 0.1014 | +0.0005 | [-0.0006, +0.0015] | 0.402 | 1 | n.s. |
| Baby | 60-item lists | fused | N@20 | 0.0475 | 0.0472 | +0.0002 | [-0.0001, +0.0006] | 0.171 | 1 | n.s. |
| Baby | full lists | alone | R@20 | 0.0844 | 0.0836 | +0.0008 | [-0.0009, +0.0026] | 0.357 | 1 | n.s. |
| Baby | full lists | alone | N@20 | 0.0376 | 0.0370 | +0.0006 | [-0.0002, +0.0013] | 0.121 | 1 | n.s. |
| Baby | full lists | fused | R@20 | 0.1004 | 0.1014 | -0.0010 | [-0.0021, +0.0002] | 0.093 | 1 | n.s. |
| Baby | full lists | fused | N@20 | 0.0471 | 0.0472 | -0.0001 | [-0.0005, +0.0003] | 0.557 | 1 | n.s. |
| Sports | 60-item lists | alone | R@20 | 0.0956 | 0.0958 | -0.0003 | [-0.0014, +0.0010] | 0.696 | 1 | n.s. |
| Sports | 60-item lists | alone | N@20 | 0.0433 | 0.0434 | -0.0001 | [-0.0007, +0.0005] | 0.697 | 1 | n.s. |
| Sports | 60-item lists | fused | R@20 | 0.1166 | 0.1168 | -0.0001 | [-0.0008, +0.0006] | 0.737 | 1 | n.s. |
| Sports | 60-item lists | fused | N@20 | 0.0554 | 0.0555 | -0.0000 | [-0.0003, +0.0002] | 0.784 | 1 | n.s. |
| Sports | full lists | alone | R@20 | 0.0941 | 0.0958 | -0.0017 | [-0.0030, -0.0004] | 0.009 | 0.132 | n.s. |
| Sports | full lists | alone | N@20 | 0.0426 | 0.0434 | -0.0008 | [-0.0014, -0.0003] | 0.005 | 0.086 | n.s. |
| Sports | full lists | fused | R@20 | 0.1169 | 0.1168 | +0.0002 | [-0.0005, +0.0009] | 0.629 | 1 | n.s. |
| Sports | full lists | fused | N@20 | 0.0554 | 0.0555 | -0.0001 | [-0.0003, +0.0002] | 0.538 | 1 | n.s. |

All 16 cells are ties: the shared trainer reproduces the deployed head within its seed spread (on Baby, every re-run mean is within two standard errors of the deployed mean).

### D11. CBOW gate (G7)

Pre-registered rule: for each fixed CBOW configuration, if the fused R@20 margin of SCOPE-v1 over that configuration (seed means, conservative worst case over configurations) is at or below the pre-registered MDE (Baby 0.0026, Sports 0.0022), the head's advantage over a content-seeded mean pool is stated as an increment with its test, and the non-pairwise form of the score rests on construction only. The CBOW heads here use L2-cosine scoring, SCOPE's training budget (learning rate 3e-3, batch 8192, at most 400 epochs, patience 20 checks), the content-seeded configurations use exactly SCOPE's seed, and gamma is tuned on validation over the grid without extension. Tests: SCOPE minus CBOW, seed-averaged, Holm over 32 tests. Won / tied / lost by SCOPE: 32 / 0 / 0.

| Dataset | CBOW configuration | Setting | Metric | SCOPE | CBOW | Delta | 95% CI | p | Holm p | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Baby | random seed, lambda_E = 0 | alone | R@20 | 0.0836 | 0.0701 | +0.0135 | [+0.0109, +0.0162] | < 0.0002 | < 0.0002 | a better |
| Baby | random seed, lambda_E = 0 | alone | N@20 | 0.0370 | 0.0314 | +0.0056 | [+0.0044, +0.0069] | < 0.0002 | < 0.0002 | a better |
| Baby | random seed, lambda_E = 0 | fused | R@20 | 0.1014 | 0.0948 | +0.0066 | [+0.0048, +0.0083] | < 0.0002 | < 0.0002 | a better |
| Baby | random seed, lambda_E = 0 | fused | N@20 | 0.0472 | 0.0452 | +0.0020 | [+0.0014, +0.0026] | < 0.0002 | < 0.0002 | a better |
| Baby | random seed, lambda_E = 1 | alone | R@20 | 0.0836 | 0.0701 | +0.0135 | [+0.0109, +0.0162] | < 0.0002 | < 0.0002 | a better |
| Baby | random seed, lambda_E = 1 | alone | N@20 | 0.0370 | 0.0314 | +0.0056 | [+0.0044, +0.0069] | < 0.0002 | < 0.0002 | a better |
| Baby | random seed, lambda_E = 1 | fused | R@20 | 0.1014 | 0.0948 | +0.0066 | [+0.0048, +0.0083] | < 0.0002 | < 0.0002 | a better |
| Baby | random seed, lambda_E = 1 | fused | N@20 | 0.0472 | 0.0452 | +0.0020 | [+0.0014, +0.0026] | < 0.0002 | < 0.0002 | a better |
| Baby | content seed, lambda_E = 0 | alone | R@20 | 0.0836 | 0.0760 | +0.0076 | [+0.0052, +0.0101] | < 0.0002 | < 0.0002 | a better |
| Baby | content seed, lambda_E = 0 | alone | N@20 | 0.0370 | 0.0337 | +0.0033 | [+0.0021, +0.0046] | < 0.0002 | < 0.0002 | a better |
| Baby | content seed, lambda_E = 0 | fused | R@20 | 0.1014 | 0.0981 | +0.0033 | [+0.0017, +0.0049] | < 0.0002 | < 0.0002 | a better |
| Baby | content seed, lambda_E = 0 | fused | N@20 | 0.0472 | 0.0465 | +0.0007 | [+0.0002, +0.0013] | 0.013 | 0.022 | a better |
| Baby | content seed, lambda_E = 1 | alone | R@20 | 0.0836 | 0.0760 | +0.0076 | [+0.0051, +0.0100] | < 0.0002 | < 0.0002 | a better |
| Baby | content seed, lambda_E = 1 | alone | N@20 | 0.0370 | 0.0337 | +0.0033 | [+0.0021, +0.0045] | < 0.0002 | < 0.0002 | a better |
| Baby | content seed, lambda_E = 1 | fused | R@20 | 0.1014 | 0.0980 | +0.0034 | [+0.0017, +0.0051] | < 0.0002 | < 0.0002 | a better |
| Baby | content seed, lambda_E = 1 | fused | N@20 | 0.0472 | 0.0465 | +0.0007 | [+0.0002, +0.0013] | 0.011 | 0.022 | a better |
| Sports | random seed, lambda_E = 0 | alone | R@20 | 0.0958 | 0.0778 | +0.0181 | [+0.0163, +0.0198] | < 0.0002 | < 0.0002 | a better |
| Sports | random seed, lambda_E = 0 | alone | N@20 | 0.0434 | 0.0361 | +0.0074 | [+0.0065, +0.0082] | < 0.0002 | < 0.0002 | a better |
| Sports | random seed, lambda_E = 0 | fused | R@20 | 0.1168 | 0.1110 | +0.0058 | [+0.0048, +0.0068] | < 0.0002 | < 0.0002 | a better |
| Sports | random seed, lambda_E = 0 | fused | N@20 | 0.0555 | 0.0534 | +0.0021 | [+0.0017, +0.0025] | < 0.0002 | < 0.0002 | a better |
| Sports | random seed, lambda_E = 1 | alone | R@20 | 0.0958 | 0.0778 | +0.0181 | [+0.0163, +0.0198] | < 0.0002 | < 0.0002 | a better |
| Sports | random seed, lambda_E = 1 | alone | N@20 | 0.0434 | 0.0361 | +0.0074 | [+0.0065, +0.0082] | < 0.0002 | < 0.0002 | a better |
| Sports | random seed, lambda_E = 1 | fused | R@20 | 0.1168 | 0.1110 | +0.0058 | [+0.0048, +0.0068] | < 0.0002 | < 0.0002 | a better |
| Sports | random seed, lambda_E = 1 | fused | N@20 | 0.0555 | 0.0534 | +0.0021 | [+0.0017, +0.0025] | < 0.0002 | < 0.0002 | a better |
| Sports | content seed, lambda_E = 0 | alone | R@20 | 0.0958 | 0.0888 | +0.0070 | [+0.0055, +0.0085] | < 0.0002 | < 0.0002 | a better |
| Sports | content seed, lambda_E = 0 | alone | N@20 | 0.0434 | 0.0406 | +0.0029 | [+0.0022, +0.0036] | < 0.0002 | < 0.0002 | a better |
| Sports | content seed, lambda_E = 0 | fused | R@20 | 0.1168 | 0.1153 | +0.0015 | [+0.0006, +0.0024] | 0.002 | 0.010 | a better |
| Sports | content seed, lambda_E = 0 | fused | N@20 | 0.0555 | 0.0550 | +0.0005 | [+0.0002, +0.0008] | 0.003 | 0.013 | a better |
| Sports | content seed, lambda_E = 1 | alone | R@20 | 0.0958 | 0.0888 | +0.0070 | [+0.0056, +0.0085] | < 0.0002 | < 0.0002 | a better |
| Sports | content seed, lambda_E = 1 | alone | N@20 | 0.0434 | 0.0406 | +0.0029 | [+0.0022, +0.0036] | < 0.0002 | < 0.0002 | a better |
| Sports | content seed, lambda_E = 1 | fused | R@20 | 0.1168 | 0.1153 | +0.0014 | [+0.0006, +0.0024] | 0.002 | 0.010 | a better |
| Sports | content seed, lambda_E = 1 | fused | N@20 | 0.0555 | 0.0550 | +0.0005 | [+0.0002, +0.0008] | 0.004 | 0.013 | a better |

Gate outcome (fused R@20 margin of SCOPE-v1 over each configuration, seed means, against the pre-registered MDE):

| Dataset | CBOW configuration | Margin | MDE | Minimum over configurations | Gate |
|---|---|---|---|---|---|
| Baby | random seed, lambda_E = 0 | +0.0066 | 0.0026 | | |
| Baby | random seed, lambda_E = 1 | +0.0066 | 0.0026 | | |
| Baby | content seed, lambda_E = 0 | +0.0033 | 0.0026 | yes | not triggered |
| Baby | content seed, lambda_E = 1 | +0.0034 | 0.0026 | | |
| Sports | random seed, lambda_E = 0 | +0.0058 | 0.0022 | | |
| Sports | random seed, lambda_E = 1 | +0.0058 | 0.0022 | | |
| Sports | content seed, lambda_E = 0 | +0.0015 | 0.0022 | | |
| Sports | content seed, lambda_E = 1 | +0.0014 | 0.0022 | yes | triggered |

The lambda_E = 0 and lambda_E = 1 variants of CBOW random give the same test R@20 to four decimals on both datasets, and the two content-seeded variants differ by at most 0.0004: the SIGReg term on the item table has no measurable effect on a mean-pool arm at this learning rate. The pre-registered family nevertheless counts all four configurations.

### D12. Per-seed tests, summary

Each seed is tested separately (G13 per-seed family: 120 tests, Holm over 120; G7 per-seed family: 96 tests, Holm over 96). Entries are wins / ties by SCOPE out of the four tests (alone and fused, R@20 and N@20) per arm and seed; no arm wins any test.

| Family | Dataset | Arm | Seed 2024 | Seed 2025 | Seed 2026 |
|---|---|---|---|---|---|
| G13 per-seed | Baby | CBOW random | 4 / 0 | 3 / 1 | 4 / 0 |
| G13 per-seed | Baby | CBOW content | 0 / 4 | 0 / 4 | 0 / 4 |
| G13 per-seed | Baby | BERT4Rec-style | 4 / 0 | 0 / 4 | 1 / 3 |
| G13 per-seed | Baby | SASRec-style | 4 / 0 | 4 / 0 | 4 / 0 |
| G13 per-seed | Baby | Mult-VAE | 2 / 2 | 0 / 4 | 1 / 3 |
| G13 per-seed | Sports | CBOW random | 4 / 0 | 4 / 0 | 4 / 0 |
| G13 per-seed | Sports | CBOW content | 0 / 4 | 0 / 4 | 0 / 4 |
| G13 per-seed | Sports | BERT4Rec-style | 3 / 1 | 3 / 1 | 3 / 1 |
| G13 per-seed | Sports | SASRec-style | 4 / 0 | 4 / 0 | 4 / 0 |
| G13 per-seed | Sports | Mult-VAE | 2 / 2 | 3 / 1 | 3 / 1 |
| G7 per-seed | Baby | CBOW random, lambda_E = 0 | 4 / 0 | 4 / 0 | 4 / 0 |
| G7 per-seed | Baby | CBOW random, lambda_E = 1 | 4 / 0 | 4 / 0 | 4 / 0 |
| G7 per-seed | Baby | CBOW content, lambda_E = 0 | 2 / 2 | 0 / 4 | 3 / 1 |
| G7 per-seed | Baby | CBOW content, lambda_E = 1 | 2 / 2 | 1 / 3 | 4 / 0 |
| G7 per-seed | Sports | CBOW random, lambda_E = 0 | 4 / 0 | 4 / 0 | 4 / 0 |
| G7 per-seed | Sports | CBOW random, lambda_E = 1 | 4 / 0 | 4 / 0 | 4 / 0 |
| G7 per-seed | Sports | CBOW content, lambda_E = 0 | 2 / 2 | 2 / 2 | 2 / 2 |
| G7 per-seed | Sports | CBOW content, lambda_E = 1 | 2 / 2 | 2 / 2 | 2 / 2 |

Totals: G13 per-seed 72 wins / 48 ties / 0 losses; G7 per-seed 72 / 24 / 0. The per-seed tests carry the training-seed spread that the seed-averaged tests do not, and with Holm over 120 tests they call fewer wins: on Baby the BERT4Rec-style and Mult-VAE arms tie the head on seed 2025 in every cell, and the content-seeded CBOW ties it in every cell on every seed of both datasets.

### D13. History of the earlier comparison

An earlier version of this comparison had competitor-side implementation errors: CBOW was trained and scored with L1-normalized vectors instead of the cosine used by the head, the competitors' content seed was down-scaled relative to SCOPE's, and Mult-VAE was stopped by a binding epoch cap. Each arm also ran in one configuration with a fusion grid that lacked SCOPE's largest weight, and CBOW used SCOPE's fusion weight without selection. The rerun above corrects all of these (one shared trainer, one budget, validation-selected configurations, three seeds, every configuration reported). Every recorded run of the earlier version is listed here as history; none of them is evidence for the paper.

| Run | Dataset | Arm | Alone R@20 | Fused R@20 | Note |
|---|---|---|---|---|---|
| A | Baby | CBOW, random init | 0.0646 | 0.0917 | gamma fixed at 0.3 |
| A | Baby | CBOW, text seed | 0.0615 | 0.0924 | gamma fixed at 0.3 |
| A | Sports | CBOW, random init | 0.0712 | 0.1109 | gamma fixed at 0.3 |
| A | Sports | CBOW, text seed | 0.0651 | 0.1082 | gamma fixed at 0.3 |
| A | Clothing | CBOW, random init | 0.0398 | 0.0867 | gamma fixed at 0.6 |
| A | Clothing | CBOW, text seed | 0.0353 | 0.0886 | gamma fixed at 0.6 |
| B | Baby | BERT4Rec-style | 0.0745 | 0.0968 | |
| B | Baby | SASRec-style | 0.0039 | 0.0039 | failed numerically (fully masked causal attention rows) |
| B | Baby | Mult-VAE | 0.0763 | 0.0969 | |
| B | Sports | BERT4Rec-style | 0.0696 | 0.1124 | |
| B | Sports | SASRec-style | 0.0013 | 0.0013 | failed numerically |
| B | Sports | Mult-VAE | 0.0821 | 0.1115 | |
| B | Clothing | BERT4Rec-style | 0.0356 | 0.0922 | |
| B | Clothing | SASRec-style | 0.0014 | 0.0014 | failed numerically |
| B | Clothing | Mult-VAE | 0.0483 | 0.0922 | |
| C | Baby | BERT4Rec-style | 0.0705 | 0.0963 | |
| C | Baby | SASRec-style (start token added) | 0.0610 | 0.0958 | |
| C | Baby | Mult-VAE | 0.0763 | 0.0969 | |
| C | Sports | BERT4Rec-style | 0.0696 | 0.1124 | |
| C | Sports | SASRec-style (start token added) | 0.0656 | 0.1120 | |
| C | Sports | Mult-VAE | 0.0821 | 0.1115 | |
| C | Clothing | BERT4Rec-style | 0.0356 | 0.0922 | |
| C | Clothing | SASRec-style (start token added) | 0.0366 | 0.0911 | |
| C | Clothing | Mult-VAE | 0.0483 | 0.0922 | |
| reference | Baby | SCOPE head / SCOPE-v1 (seed 2024) | 0.0857 | 0.1017 | |
| reference | Sports | SCOPE head / SCOPE-v1 (seed 2024) | 0.0946 | 0.1160 | |
| reference | Clothing | SCOPE head / SCOPE-v1 (seed 2024) | 0.0682 | 0.0943 | |

Interpretation. Under one matched protocol the SCOPE head matches or exceeds its nearest learned relatives and loses to none: it exceeds the randomly initialized item2vec/CBOW, the BERT4Rec-style and SASRec-style set Transformers and Mult-VAE in 29 of the 40 primary tests and ties the rest, and the fused-selected panel gives the same picture (15 wins, 5 ties). The ties are informative. A mean-pool head with the same content seed, trained with the same masked-set softmax, is statistically indistinguishable from the SCOPE head on Baby in every cell (fused R@20 +0.0005, p = 0.524) and on Sports when fused (+0.0000, p = 0.983) and on standalone R@20 (Holm p 0.142), with the head ahead only on standalone N@20 on Sports (+0.0010, Holm p 0.048); the CBOW gate finds the residual encoder's fused increment over the best fixed content-seeded mean pool below the pre-registered MDE on Sports (+0.0014 against 0.0022) while on Baby it is above the MDE (+0.0033 against 0.0026) but not significant in the standalone-selected comparison. The value of the head therefore lies in the content-seeded item table trained with the masked-set full-catalog softmax, not in the residual encoder, and the paper states the head's standing in these terms. The encoder-only family, which retrains the head in the arms' own trainer, gives the same conclusion (27 wins, 13 ties, 0 losses), and the reproduction check shows that the shared trainer reproduces the deployed head (16 ties of 16).

## E. Coverage, breadth, user-activity strata and popularity concentration

This section characterizes where SCOPE-U's margin over GUME comes from on Baby, Sports and Clothing (seed 2024): which test pairs it covers, how many users it changes, which activity strata gain, and how concentrated its recommendations are.

Table E1. Coverage of test pairs (share of test (user, item) pairs whose item is in the user's top-20, in %). The union oracle counts a pair as covered if any of the three views (base, set view, GUME) covers it; it chooses a view after seeing the test item and is not an achievable target. "Captured share" is SCOPE-U's coverage as a share of the oracle's; "incremental share" is (SCOPE-U minus GUME) / (oracle minus GUME).

| Dataset | GUME | SCOPE-U | Union oracle | Captured share | Incremental share of headroom |
|---|---|---|---|---|---|
| Baby | 10.1 | 10.9 | 15.8 | 68.8 | 14.1 |
| Sports | 11.7 | 12.7 | 17.4 | 72.9 | 17.1 |
| Clothing | 9.9 | 10.9 | 14.5 | 74.9 | 21.0 |

Table E2. Per-pair top-20 miss rate (one minus coverage) of the four scorers.

| Dataset | SCOPE-U | GUME | Base | Set view |
|---|---|---|---|---|
| Baby | 0.8910 | 0.8991 | 0.9069 | 0.9147 |
| Sports | 0.8730 | 0.8827 | 0.8930 | 0.9055 |
| Clothing | 0.8911 | 0.9009 | 0.9093 | 0.9317 |

Table E3. Breadth: adding the set view to the set-free composition base + GUME, share of users (%) whose R@20 rises, falls or is unchanged, and share of users whose top-20 list changes at all.

| Dataset | Helped | Hurt | Unchanged R@20 | Helped or unchanged | Top-20 list changed |
|---|---|---|---|---|---|
| Baby | 0.8 | 0.8 | 98.4 | 99.2 | 95.9 |
| Sports | 1.3 | 1.2 | 97.6 | 98.8 | 98.0 |
| Clothing | 0.5 | 0.5 | 99.0 | 99.5 | 92.7 |

Table E4. Popularity concentration: coverage of the tail of the item-popularity distribution (%), Gini of the recommendation counts over items, and share of the catalog recommended at least once.

| Dataset | Tail coverage SCOPE-U | Tail coverage GUME | Gini SCOPE-U | Gini GUME | Catalog coverage SCOPE-U | Catalog coverage GUME |
|---|---|---|---|---|---|---|
| Baby | 1.06 | 0.97 | 0.914 | 0.903 | 0.692 | 0.632 |
| Sports | 1.89 | 1.32 | 0.901 | 0.914 | 0.697 | 0.542 |
| Clothing | 3.11 | 2.04 | 0.735 | 0.792 | 0.850 | 0.755 |

Table E5. User-activity quintiles (Q1 = least active fifth, mean training-set size 3.0 on every dataset): SCOPE-U minus GUME and the set view's margin over the base (SCOPE-v1 minus base), delta R@20 per quintile, and the Q1 values of SCOPE-U and GUME.

| Dataset | Comparison | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|---|
| Baby | SCOPE-U minus GUME | +0.0049 | +0.0080 | +0.0111 | +0.0069 | +0.0078 |
| Baby | SCOPE-v1 minus base | +0.0162 | +0.0059 | +0.0077 | +0.0121 | +0.0055 |
| Sports | SCOPE-U minus GUME | +0.0073 | +0.0103 | +0.0101 | +0.0108 | +0.0101 |
| Sports | SCOPE-v1 minus base | +0.0091 | +0.0077 | +0.0100 | +0.0108 | +0.0083 |
| Clothing | SCOPE-U minus GUME | +0.0058 | +0.0103 | +0.0103 | +0.0109 | +0.0118 |
| Clothing | SCOPE-v1 minus base | +0.0005 | +0.0004 | +0.0030 | +0.0034 | +0.0059 |

Q1 R@20, SCOPE-U vs GUME: Baby 0.1085 vs 0.1036, Sports 0.1285 vs 0.1212, Clothing 0.0916 vs 0.0858.

Interpretation. SCOPE-U covers 0.8 to 1.0 percentage points more test pairs than GUME, which is 14 to 21% of the headroom that a per-pair choice among the three views would give, and it has the lowest miss rate of the four scorers on all three datasets. Adding the set view to base + GUME changes the top-20 list of 93 to 98% of users but the R@20 of only 1 to 2.5% of them, in equal shares upward and downward; the set view's contribution inside the composition is broad in the lists and narrow in the metric. Both margins are positive in every activity quintile; on Clothing the set view's margin over the base is smallest for the two least active fifths (+0.0005, +0.0004) and largest for the most active fifth (+0.0059), while SCOPE-U's margin over GUME grows with activity on Clothing and is flat from the second quintile on Sports. SCOPE-U recommends a larger share of the catalog and more tail items than GUME on all three datasets, but its recommendation counts are more concentrated than GUME's on Baby (Gini 0.914 vs 0.903) and less concentrated on Sports and Clothing.

## F. Further analyses

This section collects the remaining analyses of the head, the base and their fusion: hyper-parameter sweeps, the modality of the head's seed, item-level content signals, further signals over base + GUME, zero-shot transfer of the set operator between datasets, the pruning study with its controls, and user-conditioned fusion.

### F1. Head dimension

Seed 2024; fused R@20 / N@20 of SCOPE-v1 and R@20 of the head alone; d = 256 is the deployed setting. Clothing d = 512 was not run.

| Dataset | d = 32 | d = 64 | d = 128 | d = 256 | d = 512 |
|---|---|---|---|---|---|
| Baby, fused R@20 / N@20 | 0.0969 / 0.0460 | 0.1004 / 0.0466 | 0.1001 / 0.0468 | 0.1017 / 0.0473 | 0.1013 / 0.0474 |
| Baby, head alone R@20 | 0.0714 | 0.0799 | 0.0818 | 0.0857 | 0.0843 |
| Sports, fused R@20 / N@20 | 0.1146 / 0.0544 | 0.1166 / 0.0549 | 0.1171 / 0.0554 | 0.1160 / 0.0554 | 0.1166 / 0.0552 |
| Sports, head alone R@20 | 0.0826 | 0.0870 | 0.0919 | 0.0946 | 0.0917 |
| Clothing, fused R@20 / N@20 | 0.0944 / 0.0441 | 0.0947 / 0.0441 | 0.0950 / 0.0443 | 0.0943 / 0.0440 | -- |
| Clothing, head alone R@20 | 0.0497 | 0.0573 | 0.0639 | 0.0682 | -- |

The fusion weight gamma selected on validation is 0.3 in every cell except Clothing d = 256 (0.6).

### F2. Masking ratio

Fused R@20 of SCOPE-v1 when the share of each training set that is hidden is fixed instead of drawn uniformly (the deployed setting).

| Dataset | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | Uniform (deployed) | Spread over the five ratios |
|---|---|---|---|---|---|---|---|
| Baby | 0.1003 | 0.0993 | 0.0993 | 0.1017 | 0.1007 | 0.1017 | 0.0024 |
| Sports | 0.1167 | 0.1170 | 0.1176 | 0.1160 | 0.1157 | 0.1160 | 0.0019 |
| Clothing | 0.0934 | 0.0946 | 0.0935 | 0.0944 | 0.0954 | 0.0943 | 0.0019 |

### F3. Modality of the head's seed

The item table of the head is initialized from the text features (deployed), from the image features, or from the mean of the two seeds, with everything else unchanged; R@20 standalone and fused, and the selected gamma. The text-seed row is a separate run from the deployed head (fused 0.0993 / 0.1168 / 0.0945 against the deployed 0.1017 / 0.1160 / 0.0943 on Baby / Sports / Clothing); the standalone values do not depend on the base.

| Dataset | Text: standalone / fused (gamma) | Image: standalone / fused (gamma) | Both: standalone / fused (gamma) |
|---|---|---|---|
| Baby | 0.0826 / 0.0993 (0.3) | 0.0727 / 0.0959 (0.6) | 0.0775 / 0.0979 (0.3) |
| Sports | 0.0950 / 0.1168 (0.3) | 0.0808 / 0.1124 (0.6) | 0.0899 / 0.1157 (0.3) |
| Clothing | 0.0674 / 0.0945 (0.6) | 0.0500 / 0.0929 (1.5) | 0.0610 / 0.0935 (1.5) |

### F4. Item-level content signal against ranking-level use

AUC of a small metric learned on held-out item pairs for predicting co-purchase from the image or text features, the same for predicting a catalog attribute that collaborative filtering cannot see (brand on Baby, category on Sports and Clothing), the AUC of the raw cosine of the features (the similarity used by frozen kNN item graphs), and the validation-selected weight that an image-kNN or text-kNN ranking view receives when fused with the base alone.

| Dataset | Co-purchase AUC, learned image / text | Co-purchase AUC, raw cosine image / text | Attribute | Attribute AUC, learned image / text | Ranking-view weight image / text |
|---|---|---|---|---|---|
| Baby | 0.753 / 0.722 | 0.536 / 0.541 | brand | 0.842 / 0.855 | 0.0 / 0.0 |
| Sports | 0.766 / 0.751 | 0.540 / 0.652 | category | 0.877 / 0.952 | 0.0 / 0.0 |
| Clothing | 0.728 / 0.710 | 0.588 / 0.617 | category | 0.895 / 0.942 | 0.3 / 0.3 |

In the same pairwise fusion with the base, the set view receives weight 3.0 / 3.0 / 1.0 on Baby / Sports / Clothing (3.0 is the largest weight of the grid).

### F5. Further signals over base + GUME

Delta R@20 of adding one signal to the set-free composition base + GUME (seed 2024 unless stated), against the MDE of the equivalence analysis of the paper. The static set view is the set view of SCOPE-U. The orthogonalized set view removes, per user, the component of the set scores linearly explained by GUME's scores. The user-kNN view scores an item by how often the user's nearest neighbours by interaction rows interacted with it. The residual-trained head adds GUME's frozen scores scaled by beta inside the training softmax (at beta = 0 the head is unconditional but early-stopped on the fused validation score); "significant seeds" counts, of three seeds, those with a raw-significant increment. The joint-set re-ranker builds each top-20 greedily with a co-occurrence bonus (attraction) or penalty (repulsion) between a candidate and the items already selected.

| Signal added to base + GUME | Baby | Sports | Clothing |
|---|---|---|---|
| Static set view (SCOPE-U) | +0.00096 | +0.00099 | +0.00079 |
| Orthogonalized set view | +0.0002 | +0.0003 | +0.0008 |
| User-kNN view | 0.0 | +0.0007 | +0.0003 |
| Residual-trained head, beta = 0.5 (significant seeds of 3) | +0.0017 (3) | +0.0011 (1) | +0.0003 (0) |
| Residual-trained head, beta = 0, fused early stopping (significant seeds of 3) | +0.0012 (1) | +0.0016 (2) | +0.0007 (1) |
| Joint-set re-ranker, attraction | -0.0025 | -0.0009 | +0.0003 |
| Joint-set re-ranker, repulsion | +0.0003 | +0.0001 | 0.0 |
| MDE (80% power) | 0.00255 | 0.00216 | 0.00174 |
| base + GUME R@20 | 0.1080 | 0.1263 | 0.1065 |
| Mean per-user Spearman correlation, set view vs GUME | 0.57 | 0.48 | 0.31 |

No addition reaches the MDE on any dataset.

### F6. Zero-shot transfer of the set operator between datasets

The item embeddings are frozen at one text seed shared by the three datasets; only the residual encoder and the temperature are trained, on one dataset, and applied to another. Transfer efficiency is (transferred minus untrained mean pool) / (in-domain trained minus untrained mean pool) on the target dataset; head-only R@20, independent of the base.

| Trained on \ applied to | Baby | Sports | Clothing |
|---|---|---|---|
| Baby | 1.0 | -1.877 | -2.929 |
| Sports | -0.96 | 1.0 | -1.889 |
| Clothing | -0.8 | -1.091 | 1.0 |

| Target dataset | Untrained mean pool R@20 | In-domain trained R@20 | In-domain gain |
|---|---|---|---|
| Baby | 0.0271 | 0.0474 | +0.0202 |
| Sports | 0.0362 | 0.0524 | +0.0162 |
| Clothing | 0.0442 | 0.0579 | +0.0138 |

Every operator applied to another dataset falls below the untrained mean pool: the operator learns structure specific to the catalog it is trained on, not a generic transform of the text space.

### F7. Pruned components and attribution controls on the base

These runs decided which components stay in the head and the base, and control for what a fused view could add for reasons other than its content. The base of these runs is the base of the paper at a fixed configuration where stated.

Table F7a. Attribution ladder at a fixed configuration (lambda = 800, a = 0.5, pre-pruning head): R@20 / N@20 and the marginal R@20 of each step.

| Step | Baby | Sports | Clothing |
|---|---|---|---|
| EASE, one hop | 0.0832 / 0.0392 | 0.0932 / 0.0463 | 0.0573 / 0.0294 |
| + text-kNN term (marginal R@20) | 0.0925 / 0.0443 (+0.0092) | 0.1082 / 0.0525 (+0.0150) | 0.0871 / 0.0416 (+0.0298) |
| + two-hop term, the full base (marginal) | 0.0925 / 0.0442 (+0.0001) | 0.1082 / 0.0524 (+0.0000) | 0.0873 / 0.0417 (+0.0001) |
| + set head, fused (marginal) | 0.1010 / 0.0470 (+0.0085) | 0.1153 / 0.0553 (+0.0071) | 0.0926 / 0.0435 (+0.0054) |

Table F7b. Removed components: marginal R@20 of each component when removed or replaced.

| Component | Baby | Sports | Clothing |
|---|---|---|---|
| Two-hop term of the base, on the validation-tuned base | +0.0003 | +0.0004 | +0.0001 |
| Two-hop term, fixed ladder configuration | +0.0001 | +0.0000 | +0.0001 |
| SIGReg on E and on z versus neither | +0.0004 | -0.0013 | -0.0014 |
| SCOPE-G with K = 1 vs K = 3 propagation steps (R@20) | 0.1025 vs 0.1002 | 0.1174 vs 0.1161 | 0.0956 vs 0.0951 |
| Text initialization of the item table, marginal fused R@20 (pre-pruning head) | +0.0058 (6.1%) | +0.0044 (4.0%) | +0.0013 (1.4%) |

Head architecture on Baby, fused R@20: residual encoder 0.0970, mean pool 0.0945, encoder with a predictor 0.0973. This comparison was produced by an early trainer that used the L1-normalized scoring; the pruning decisions were re-checked with the main training script, and the matched comparison of Section D is the evidence on the encoder.

Table F7c. Null views and placebo (seed 2024): validation-selected weight of each view when fused with the tuned base, the set view's lift, and the paired test of base + set against base + random view.

| Control | Baby | Sports | Clothing |
|---|---|---|---|
| Random-score view, weight | 0.0 | 0.0 | 0.0 |
| Co-occurrence-kNN view, weight | 0.0 | 0.0 | 0.0 |
| Set view, weight | 3.0 | 3.0 | 1.5 |
| Set view lift over the tuned base (R@20) | +0.0095 | +0.0071 | +0.0014 |
| Tuned base R@20 | 0.0915 | 0.1081 | 0.0912 |
| p, base + set vs base + random view (Holm) | 0 (0) | 0 (0) | 0.027 (0.054) |
| Placebo view, weight (pre-pruning head) | 0.0 | 0.0 | 0.0 |
| Set view, weight and marginal R@20 in the placebo run | 2.0, +0.0069 | 2.0, +0.0014 | 1.5, +0.0015 |

Table F7d. Gamma sweep and decorrelation: selected fusion weight, whether the optimum is interior to the grid, fused minus the better single view, and the share of users for which the fused list is at least as good as, or strictly better than, the better of the two single views.

| Quantity | Baby | Sports | Clothing |
|---|---|---|---|
| Selected gamma (interior optimum) | 0.2 (yes) | 0.3 (yes) | 0.7 (yes) |
| Fused minus best single view, R@20 | +0.0097 | +0.0076 | +0.0015 |
| Users with fused >= max(base, set), % | 96.2 | 96.6 | 97.2 |
| Users with fused > max(base, set), % | 0.7 | 0.6 | 0.6 |
| Users tied on both views, % | 90.5 | 90.6 | 91.9 |

Table F7e. Frozen content views fused with the base alone: validation-selected weights of a text-kNN view, an image-kNN view and the set view, and the Spearman correlation between the base's and the set view's scores.

| Quantity | Baby | Sports | Clothing |
|---|---|---|---|
| Text-kNN view weight | 0.0 | 0.0 | 0.3 |
| Image-kNN view weight | 0.0 | 0.0 | 0.3 |
| Set view weight | 3.0 | 3.0 | 1.0 |
| Spearman(base, set) | 0.48 | 0.53 | 0.48 |

Table F7f. Trained controls for the seed and the objective (three seeds, fused minus base R@20): the head with its text seed and masked-set objective, the same head seeded from co-occurrence or at random, and a text-seeded head trained with a BPR objective; the share is each control's gain as a percentage of the deployed head's gain.

| Control | Baby | Sports | Clothing |
|---|---|---|---|
| Text seed, masked-set objective (deployed) | +0.0086 | +0.0094 | +0.0035 |
| Co-occurrence seed, masked-set objective (share) | +0.0038 (44%) | +0.0044 (46%) | +0.0013 (37%) |
| Random seed, masked-set objective (share) | +0.0031 (36%) | +0.0037 (40%) | +0.0009 (25%) |
| Text seed, BPR objective (share) | +0.0080 (93%) | +0.0071 (75%) | +0.0028 (80%) |

### F8. User-conditioned fusion

Two attempts to replace the global fusion weights by user-dependent ones. A logistic-regression router picks, per user, the base or FREEDOM from user features; a mixture-of-experts gate weights the base, FREEDOM and GUME per user and item and is trained on the training interactions.

| Quantity | Baby | Sports | Clothing |
|---|---|---|---|
| Router validation AUC | 0.538 | 0.552 | 0.561 |
| Routed minus global fusion of the two views, R@20 | -0.0093 | -0.0151 | -0.0122 |
| Mixture-of-experts minus global linear fusion, R@20 (95% CI) | -0.0101 [-0.0130, -0.0072] | -- | -- |

Neither was cross-fitted, and neither covers all three views of SCOPE-U; both fall below the global weights.

### F9. Fusion normalization

SCOPE-v1 with the two views standardized per user (deployed), converted to ranks, or min-max scaled; validation-selected gamma, fused R@20 and the complementarity (fused minus the better single view).

| Dataset | z-score: gamma / R@20 / complementarity | Rank: gamma / R@20 / complementarity | Min-max: gamma / R@20 / complementarity |
|---|---|---|---|
| Baby | 0.3 / 0.1019 / +0.0094 | 2.0 / 0.0975 / +0.0051 | 0.7 / 0.1027 / +0.0103 |
| Sports | 0.3 / 0.1164 / +0.0088 | 3.0 / 0.1130 / +0.0053 | 0.7 / 0.1176 / +0.0099 |
| Clothing | 1.0 / 0.0936 / +0.0026 | 3.0 / 0.0866 / -0.0044 | 3.0 / 0.0949 / +0.0039 |

### F10. SCOPE-G graph ablation

SCOPE-G propagates over the mean of the co-occurrence graph and the text-kNN graph (deployed, "both"); the two single graphs are ablations. R@20 / N@20, seed 2024.

| Dataset | Both graphs (deployed) | Co-occurrence only | Text only |
|---|---|---|---|
| Baby | 0.1010 / 0.0478 | 0.1011 / 0.0476 | 0.1023 / 0.0480 |
| Sports | 0.1179 / 0.0555 | 0.1165 / 0.0554 | 0.1191 / 0.0563 |
| Clothing | 0.0970 / 0.0450 | 0.0932 / 0.0436 | 0.0981 / 0.0453 |

### F11. Set-free fusion of more backbones

The base fused with every re-run collaborative backbone whose score matrix was stored, standardized per user, with weights chosen by coordinate ascent on validation R@20 and no set view; compared with SCOPE-U (seed 2024). No significance test was run.

| Dataset | Experts | R@20 / N@20 | R@20 minus SCOPE-U |
|---|---|---|---|
| Baby | base, FREEDOM, GUME, LGMRec, MGCN, LightGCN | 0.1115 / 0.0505 | +0.0017 |
| Sports | base, FREEDOM, GUME, LGMRec, MGCN | 0.1299 / 0.0596 | +0.0009 |
| Clothing | base, FREEDOM, GUME, LGMRec, MGCN | 0.1105 / 0.0510 | +0.0013 |
| MicroLens | base, FREEDOM, GUME, LGMRec | 0.1396 / 0.0638 | +0.0001 |

On MicroLens the fusion uses the validation-selected GUME and the completed LGMRec run of the main table (FREEDOM receives weight 0). The base composes with several backbones at once, and SCOPE-U's lead in the paper is over each baseline taken alone.

### F12. Set completion on training sets

Each user's training set is split at random into two halves; every scorer is conditioned on one half and R@20 of the other half is measured with the conditioning half masked. Both the head and the co-occurrence graph were fitted on training data that contains the hidden halves, so this is an in-sample diagnostic of the training task. The text-kNN of this diagnostic normalizes the text vectors by their L1 norm.

| Scorer | Baby | Sports | Clothing |
|---|---|---|---|
| SCOPE head | 0.1610 | 0.2880 | 0.4270 |
| Top-20 co-occurrence kNN | 0.4610 | 0.6340 | 0.8780 |
| Text kNN | 0.0340 | 0.0450 | 0.0600 |
| Popularity | 0.0500 | 0.0290 | 0.0160 |

### F13. Pairwise reproduction of the head

An item-item matrix fitted by ridge regression to reproduce the trained head's score matrix, scored as a sum of pairwise terms over each user's items; ridge weight selected on validation from {10, 100, 500, 1500} (100 on all three datasets). Head-only, independent of the base.

| Dataset | Set head R@20 | Best pairwise operator R@20 | Recovered | Head minus operator (p) |
|---|---|---|---|---|
| Baby | 0.0857 | 0.0856 | 99.9% | +0.0001 (0.927) |
| Sports | 0.0946 | 0.0962 | 101.6% | -0.0015 (0.1178) |
| Clothing | 0.0682 | 0.0691 | 101.3% | -0.0009 (0.285) |

Interpretation. The head is robust to its own hyper-parameters (a spread of 0.0019 to 0.0024 R@20 over masking ratios, and fused values within 0.0048 of the deployed one over d = 32 to 512), the text seed is the best of the three seeds on every dataset, the two-hop term of the base contributes at most 0.0004 R@20 and SIGReg on both the item table and the set vector changes R@20 by +0.0004 to -0.0014, which is why the deployed base is one-hop and the deployed head keeps SIGReg on the item table only. The controls locate the head's gain over the base: random and co-occurrence-kNN views receive weight zero where the set view receives 3.0, a randomly seeded head keeps 25 to 40% of the text-seeded head's gain and a text-seeded BPR head 75 to 93% of it, so the seed matters more than the masked-set objective on these benchmarks. The pairwise operator fitted to the head's scores reproduces its standalone accuracy within 0.0015 R@20 (all n.s.), so the non-pairwise form of the score is a property of the score's form, not a measured top-20 advantage. Over base + GUME no further content- or set-derived signal reaches the MDE, and two user-conditioned fusions fall below the global weights; the operator does not transfer between catalogs.

## G. Published versus reproduced baselines, and baseline configurations

This section places every re-run baseline that has a published number next to that number, states the best value of any compared method from either source, and lists how the baselines were configured.

Published values are hand-copied from the primary table named in the last column; "indep." marks values from an independent reproduction (EGRA, arXiv 2508.16170, Table I), which is the only reference available on MicroLens. Delta % = 100 (ours minus published) / published on the four-decimal values. DA-MRS is our port of the released code with its user-item graph built by coordinate-list assignment: the released code fills a SciPy dok_matrix through a private method that SciPy 1.13 removed, and the usual replacement, dict.update on the matrix, silently leaves the graph empty on current SciPy.

Table G1. Recall@20, reproduced (ours) versus published, Baby / Sports / Clothing / MicroLens.

| Method | Baby ours | Baby pub. | Delta % | Sports ours | Sports pub. | Delta % | Clothing ours | Clothing pub. | Delta % | MicroLens ours | MicroLens indep. | Delta % | Published source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GUME | 0.1024 | 0.1042 | -1.7 | 0.1190 | 0.1165 | +2.1 | 0.0993 | 0.1024 | -3.0 | 0.1216 | 0.1207 | +0.7 | arXiv 2407.12338 Table 2 |
| SMORE | 0.0995 | 0.1035 | -3.9 | 0.1128 | 0.1142 | -1.2 | 0.0953 | 0.0987 | -3.4 | 0.1118 | -- | -- | arXiv 2412.14978 Table 2 |
| COHESION | 0.0992 | 0.1052 | -5.7 | 0.1081 | 0.1137 | -4.9 | 0.0849 | 0.0983 | -13.6 | 0.1098 | -- | -- | arXiv 2504.04452 Table 3 |
| DRAGON | 0.0990 | 0.1021 | -3.0 | 0.1105 | 0.1124 | -1.7 | 0.0932 | 0.0957 | -2.6 | 0.1171 | -- | -- | arXiv 2301.12097 Table 3 |
| DA-MRS | 0.0932 | 0.0994 | -6.2 | 0.1075 | 0.1125 | -4.4 | 0.0900 | 0.0963 | -6.5 | 0.1169 | 0.1221 | -4.3 | arXiv 2406.12501 Table 2 (LightGCN backbone); MicroLens EGRA |
| LGMRec | 0.0965 | 0.1002 | -3.7 | 0.1063 | 0.1068 | -0.5 | 0.0791 | 0.0828 | -4.5 | 0.1048 | 0.1132 | -7.4 | arXiv 2312.16400 Table 2; MicroLens EGRA |
| FREEDOM | 0.0928 | 0.0992 | -6.5 | 0.1067 | 0.1089 | -2.0 | 0.0914 | 0.0941 | -2.9 | 0.0978 | 0.1032 | -5.2 | arXiv 2211.06924 Table 4; MicroLens EGRA |
| MENTOR | 0.0919 | 0.1048 | -12.3 | 0.0585 | 0.1139 | -48.6 | 0.0495 | 0.0989 | -49.9 | 0.0491 | -- | -- | arXiv 2402.19407 Table 2 |
| MGCN | 0.0753 | 0.0964 | -21.9 | 0.0735 | 0.1106 | -33.5 | 0.0817 | 0.0945 | -13.5 | 0.1141 | 0.1134 | +0.6 | arXiv 2308.03588 Table 2; MicroLens EGRA |
| BM3 | 0.0747 | 0.0883 | -15.4 | 0.0864 | 0.0980 | -11.8 | 0.0526 | 0.0621 | -15.3 | 0.0875 | 0.0981 | -10.8 | arXiv 2207.05969 Table 3; Clothing from MENTOR Table 2; MicroLens EGRA |
| LATTICE | 0.0855 | 0.0850 | +0.6 | 0.0979 | 0.0953 | +2.7 | 0.0616 | 0.0733 | -16.0 | 0.1128 | 0.1089 | +3.6 | FREEDOM, arXiv 2211.06924 Table 4; MicroLens EGRA |
| GRCN | 0.0845 | 0.0824 | +2.5 | 0.0883 | 0.0919 | -3.9 | 0.0662 | 0.0657 | +0.8 | 0.1110 | -- | -- | FREEDOM, arXiv 2211.06924 Table 4 |
| LightGCN | 0.0737 | 0.0754 | -2.3 | 0.0871 | 0.0864 | +0.8 | 0.0532 | 0.0544 | -2.2 | 0.1100 | 0.1075 | +2.3 | FREEDOM, arXiv 2211.06924 Table 4; MicroLens EGRA |
| VBPR | 0.0591 | 0.0663 | -10.9 | 0.0530 | 0.0856 | -38.1 | 0.0355 | 0.0415 | -14.5 | 0.0492 | 0.1026 | -52.0 | FREEDOM, arXiv 2211.06924 Table 4; MicroLens EGRA |
| MMGCN | 0.0583 | 0.0660 | -11.7 | 0.0528 | 0.0636 | -17.0 | 0.0302 | 0.0361 | -16.3 | 0.0715 | -- | -- | FREEDOM, arXiv 2211.06924 Table 4 |

Table G2. NDCG@20, reproduced versus published.

| Method | Baby ours | Baby pub. | Delta % | Sports ours | Sports pub. | Delta % | Clothing ours | Clothing pub. | Delta % | MicroLens ours | MicroLens indep. | Delta % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GUME | 0.0448 | 0.0460 | -2.6 | 0.0530 | 0.0527 | +0.6 | 0.0447 | 0.0466 | -4.1 | 0.0531 | 0.0525 | +1.1 |
| SMORE | 0.0436 | 0.0457 | -4.6 | 0.0505 | 0.0506 | -0.2 | 0.0432 | 0.0443 | -2.5 | 0.0471 | -- | -- |
| COHESION | 0.0438 | 0.0454 | -3.5 | 0.0484 | 0.0503 | -3.8 | 0.0384 | 0.0438 | -12.3 | 0.0474 | -- | -- |
| DRAGON | 0.0432 | 0.0435 | -0.7 | 0.0498 | 0.0500 | -0.4 | 0.0421 | 0.0435 | -3.2 | 0.0518 | -- | -- |
| DA-MRS | 0.0415 | 0.0435 | -4.6 | 0.0480 | 0.0498 | -3.6 | 0.0403 | 0.0433 | -6.9 | 0.0500 | 0.0536 | -6.7 |
| LGMRec | 0.0422 | 0.0440 | -4.1 | 0.0470 | 0.0480 | -2.1 | 0.0353 | 0.0371 | -4.9 | 0.0449 | 0.0489 | -8.2 |
| FREEDOM | 0.0407 | 0.0424 | -4.0 | 0.0464 | 0.0481 | -3.5 | 0.0406 | 0.0420 | -3.3 | 0.0406 | 0.0437 | -7.1 |
| MENTOR | 0.0404 | 0.0450 | -10.2 | 0.0258 | 0.0511 | -49.5 | 0.0214 | 0.0441 | -51.5 | 0.0200 | -- | -- |
| MGCN | 0.0331 | 0.0427 | -22.5 | 0.0335 | 0.0496 | -32.5 | 0.0368 | 0.0428 | -14.0 | 0.0485 | 0.0484 | +0.2 |
| BM3 | 0.0326 | 0.0383 | -14.9 | 0.0388 | 0.0438 | -11.4 | 0.0235 | 0.0281 | -16.4 | 0.0356 | 0.0400 | -11.0 |
| LATTICE | 0.0373 | 0.0370 | +0.8 | 0.0435 | 0.0421 | +3.3 | 0.0267 | 0.0330 | -19.1 | 0.0490 | 0.0473 | +3.6 |
| GRCN | 0.0372 | 0.0358 | +3.9 | 0.0395 | 0.0413 | -4.4 | 0.0289 | 0.0284 | +1.8 | 0.0476 | -- | -- |
| LightGCN | 0.0328 | 0.0328 | +0.0 | 0.0393 | 0.0387 | +1.6 | 0.0239 | 0.0243 | -1.6 | 0.0476 | 0.0467 | +1.9 |
| VBPR | 0.0241 | 0.0284 | -15.1 | 0.0221 | 0.0384 | -42.4 | 0.0144 | 0.0192 | -25.0 | 0.0195 | 0.0441 | -55.8 |
| MMGCN | 0.0250 | 0.0282 | -11.3 | 0.0233 | 0.0270 | -13.7 | 0.0124 | 0.0154 | -19.5 | 0.0289 | -- | -- |

Table G3. Best value of any compared method from either source, among the twenty re-run baselines' own and published numbers, and SCOPE-U's margin over it. The last row gives the values that enter with the additional rows of the main table (the content closed forms of Section C).

| Dataset | Best R@20 (source) | Best N@20 (source) | SCOPE-U R@20 / N@20 | Margin % R@20 / N@20 |
|---|---|---|---|---|
| Baby | 0.1052 (COHESION, published) | 0.0460 (GUME, published) | 0.1098 / 0.0498 | +4.4 / +8.3 |
| Sports | 0.1190 (GUME, ours) | 0.0530 (GUME, ours) | 0.1290 / 0.0603 | +8.4 / +13.8 |
| Clothing | 0.1024 (GUME, published) | 0.0466 (GUME, published) | 0.1092 / 0.0501 | +6.6 / +7.5 |
| MicroLens | 0.1245 (EASE, ours) | 0.0593 (EASE, ours) | 0.1395 / 0.0640 | +12.0 / +7.9 |
| MicroLens, with the content closed forms | 0.1300 (FEASE-full-tfidf, ours) | 0.0612 (FEASE-full-tfidf, ours) | 0.1395 / 0.0640 | -- |

Baseline configurations. The twenty re-run baselines are three collaborative models (LightGCN, EASE, ADMM-SLIM), fourteen multimodal recommenders (VBPR, MMGCN, GRCN, LATTICE, BM3, MGCN, MENTOR, FREEDOM, LGMRec, DA-MRS, DRAGON, COHESION, SMORE, GUME) and three diffusion- or language-model-augmented recommenders (DiffMM, LLMRec, RLMRec). Six are ports of released code (GRCN, DRAGON, DA-MRS, SMORE, COHESION, GUME); the others are re-implemented from their papers, and three of the re-implementations are simplified (DiffMM without its diffusion stage, LLMRec and RLMRec without their language-model augmentations or profiles, which the released text features replace). Every learned baseline is trained once, on seed 2024, from one configuration: the released values for the ports (per dataset for GUME, one setting across datasets for the others) and one setting across datasets for the re-implementations; the released configurations of GUME, SMORE and MGCN specify a learning-rate decay, which we did not apply. The only validation-based choices for learned baselines are between two learning rates for DRAGON on Baby, between two depths for COHESION on Clothing, over four DA-MRS settings (kl_weight 0.1 or 1.0 by neighbor_weight 0.001 or 0.01, knn_k 10) on Baby, Sports and Clothing (the released setting is kept on Sports and Clothing; kl_weight 0.1, neighbor_weight 0.001 is selected on Baby; MicroLens uses the released setting), and over a nine-setting grid for GUME on MicroLens (Table G4). EASE and ADMM-SLIM select their regularization weights on validation. All learned baselines share the framework's training settings: Adam, embedding size 64, batch size 2048, at most 1000 epochs, validation after every epoch, early stopping after 20 evaluations without improvement (157 of the 158 run configurations of the initial sweep share these values, and so does every later run: LGMRec on MicroLens and the DA-MRS settings above). LGMRec's best validation epochs are 128 (Baby), 230 (Sports), 165 (Clothing), 325 (MicroLens) and 122 (Electronics).

Table G4. GUME on MicroLens: best validation R@20 of each setting of the nine-setting grid, learning rate by number of layers.

| Learning rate | 1 layer | 2 layers | 3 layers |
|---|---|---|---|
| 5e-4 | 0.1235 | 0.1206 | 0.1177 |
| 1e-3 | 0.1223 | 0.1196 | 0.1168 |
| 2e-3 | 0.1205 | 0.1181 | 0.1161 |

The validation-selected setting (5e-4, 1 layer) gives the GUME row of the main table on MicroLens (0.0816 / 0.0428 / 0.1216 / 0.0531). SCOPE-U and every GUME composition on MicroLens use this run's score matrix (seeds 2024 to 2026 of the head); SCOPE-v2 does not involve GUME. With the initial GUME run (1e-3, 2 layers; 0.0771 / 0.0400 / 0.1181 / 0.0506) SCOPE-U was 0.0980 / 0.0532 / 0.1393 / 0.0639 on seed 2024; with the validation-selected run it is 0.0980 / 0.0533 / 0.1395 / 0.0640, and 0.1398 and 0.1396 R@20 on seeds 2025 and 2026.

Interpretation. Ten of the fifteen comparable baselines are reproduced within 7% of their published R@20 on Baby, and GUME, the strongest re-run baseline on the three Amazon datasets, is within 3% on every dataset (+2.1% on Sports). Seven methods are more than 10% below a published value on at least one dataset (MENTOR, MGCN, BM3, VBPR and MMGCN on several datasets; COHESION and LATTICE on Clothing), which is why the paper credits each baseline with the better of its re-run and published numbers when it states margins: on that pool SCOPE-U leads by 4.4 to 12.0% R@20 and 7.5 to 13.8% N@20. One addition narrows the picture: on MicroLens the content closed forms of Section C (0.1300, FEASE-full-tfidf) replace EASE as the strongest baseline, and SCOPE-U stays 0.0095 above them.

## H. Feature provenance

This section reports the checks on item-feature views that the paper's provenance screen rests on.

Table H1. Sign rule on item-side views (seed 2024): for each user, the standardized score that a view assigns to the user's training items minus the score of the user's test items, and the score of the test items minus that of the validation items, averaged over users. A view fitted on the training interactions separates training from test items; a view built only from released item features should not.

| Dataset | View | Train minus test | Test minus validation |
|---|---|---|---|
| Baby | text affinity | -0.233 | 0.0045 |
| Baby | image affinity | -0.102 | 0.0161 |
| Baby | interaction EASE (lambda 800, no text) | 6.53 | 0.0091 |
| Sports | text affinity | -0.469 | 0.0077 |
| Sports | image affinity | -0.149 | -0.002 |
| Sports | interaction EASE (lambda 800, no text) | 13.421 | 0.0754 |
| Clothing | text affinity | -0.707 | -0.0366 |
| Clothing | image affinity | -0.278 | 0.0013 |
| Clothing | interaction EASE (lambda 800, no text) | 24.354 | -0.0441 |

Table H2. An earlier version of the same test on the fused base and on its text term alone (train minus test z-gap), superseded by Table H1 and listed for completeness.

| Dataset | Base (EASE + text term) | Text term alone |
|---|---|---|
| Baby | 7.3524 | 6.291 |
| Sports | 13.1055 | 12.9431 |
| Clothing | 23.5526 | 23.6344 |

Interpretation. The item-feature views separate training from test items in the wrong direction (train minus test between -0.102 and -0.707) and test from validation items by at most 0.0754 in standardized units, while the interaction-fitted view separates them by 6.53 to 24.354; released text and image features therefore behave like features that carry no split information.

## I. Implementation details

This section lists the protocol constants, the SCOPE training settings, the grids and the selected weights of every variant, and the footprint of the method.

Table I1. Protocol constants.

| Item | Value |
|---|---|
| Split | 8:1:1 random split of the MMRec release (per-interaction split label) |
| Metrics | Recall@K and NDCG@K, K in {10, 20}, full-catalog ranking with the user's training items masked |
| Seeds | main table seed 2024; three-seed values 2024, 2025, 2026 |
| Feature dimensions | Amazon: image 4096, text 384; MicroLens: image 1024, text 1024, video 768 |
| Paired bootstrap | B = 10,000 user-level resamples, two-sided; in the family study and the head-on-kernel test p = (count + 1) / (B + 1), so the smallest reportable p is 0.0002; in earlier runs a p of 0 means p < 0.0002 |
| Hardware | the baselines of the main table ran on one 24 GB GPU (RTX 4090); the closed-form family, the head-on-kernel test and the neighbour comparison ran on one RTX 3090 |

Table I2. SCOPE training settings (shared across datasets unless stated).

| Item | Value |
|---|---|
| Head dimension d | 256 |
| Head optimizer | Adam, learning rate 0.003, weight decay 1e-6, batches of 8192 users |
| Head schedule | at most 400 epochs, validation R@20 of the head alone every 4 epochs, early stopping after 20 checks without improvement, best checkpoint restored |
| Head regularizer | SIGReg on the item table with weight 1 (256 slices, 17 points, T = 5) |
| Temperature | initialized at 0.1, floor 0.001 |
| Training lists | at most 60 items per user (list cap 60) |
| SCOPE-G | K = 1 propagation step over the mean of the co-occurrence graph and the text-kNN graph (k = 20), symmetrically normalized; at most 220 epochs, patience 18 checks; gamma grid {0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0} |
| Base grid | lambda in {400, 800, 1500}, a in {0, 0.3, 0.5, 0.7}, text-kNN with k = 20 |
| Base selected (lambda, a) | Baby 800, 0.5; Sports 1500, 0.5; Clothing 1500, 0.7; MicroLens 400, 0.3 |
| SCOPE-v1 gamma grid | {0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0} |
| Composition grid (SCOPE-v2, SCOPE-U) | each of the three view weights in {0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0}, all-zero excluded: 342 combinations; the four Electronics weights in {0, 0.3, 0.6, 1.0, 2.0} |

Table I3. Selected weights of the SCOPE variants (validation R@20, seed 2024).

| Variant | Weights | Baby | Sports | Clothing | MicroLens |
|---|---|---|---|---|---|
| SCOPE-v1 | gamma (best epoch of the head) | 0.3 (24) | 0.3 (16) | 0.6 (16) | 0.3 (12) |
| SCOPE-G | gamma, K | 0.3, 1 | 0.3, 1 | 0.3, 1 | 0.3, 1 |
| SCOPE-v2 | (FREEDOM, base, set) | 1.0, 0.6, 2.0 | 1.0, 0.3, 1.5 | 3.0, 0.6, 1.5 | 1.0, 0.6, 2.0 |
| SCOPE-U | (GUME, base, set) | 1.0, 0.3, 0.6 | 1.5, 0.3, 1.5 | 3.0, 0.3, 0.3 | 1.0, 0.3, 0.6 |
| base + GUME, no set (control) | (GUME, base) | 1.5, 0.3 | 1.5, 0.3 | 2.0, 0.3 | 1.0, 0.3 |
| SCOPE-v1 on the strongest published kernel | gamma per seed 2024 / 2025 / 2026 | 1.0 / 0.6 / 0.6 | 1.0 / 0.6 / 0.6 | 1.5 / 1.0 / 1.0 | 0.3 / 0.3 / 0.3 |

Table I4. Footprint. Item table = the head's item embeddings; encoder + tau = the residual encoder and the temperature; dense B = the base's item-item matrix; "GUME user tables" = GUME's three per-user embedding tables of width 64. Solve = one base fit at lambda 800, a 0.5 (median of 3 after 1 warm-up); Infer = the head alone scoring the full catalog (median of 7 after 3 warm-ups); peak GB = peak GPU memory of the head's inference.

| Dataset | Item table (M params) | Encoder + tau (K params) | Per-user params | Dense B (M entries) | Dense B (GB, fp32) | GUME user tables (M params) | Solve (s) | Infer (s) | Peak GB |
|---|---|---|---|---|---|---|---|---|---|
| Baby | 1.80 | 263 | 0 | 49.7 | 0.20 | 3.73 | 0.21 | 0.004 | 1.14 |
| Sports | 4.70 | 263 | 0 | 337.0 | 1.35 | 6.83 | 1.45 | 0.014 | 5.31 |
| Clothing | 5.90 | 263 | 0 | 530.5 | 2.12 | 7.56 | 2.55 | 0.019 | 7.34 |
| MicroLens | 4.41 | 263 | 0 | 296.8 | 1.19 | 18.84 | 2.35 | 0.020 | 6.98 |

A dense item-item matrix on Electronics would take 15.9 GB. The wall-clock time and peak memory of fusing the head with the strongest published kernel are in Table C11.

Known discrepancies between evaluations. Several quantities exist in two evaluations of the same model, which agree to within the last digit:

- The re-exported scores used by the significance tests differ from the main-table cells by at most 0.0002 (Baby GUME 0.1022 against 0.1024; Baby SCOPE-U 0.1100 against 0.1098; Clothing SCOPE-U 0.1093 against 0.1092).
- The three-seed SCOPE-v1 values from the deployed per-seed gamma (0.1014 / 0.1168 / 0.0943 on Baby / Sports / Clothing) and from a re-fusion with the gamma re-selected on the composition grid (0.1015 / 0.1171 / 0.0944) are two fusions of the same heads.
- On MicroLens, SCOPE-v1's R@10 and N@10 are 0.0943 / 0.0518 from the deployed fusion and 0.0944 / 0.0519 from the composition re-fusion.
- The three-seed SCOPE-U values of seeds 2025 and 2026 were evaluated with 16-bit views while seed 2024 used 32-bit; a 32-bit re-evaluation gives a Baby SCOPE-U three-seed N@20 of 0.0500 against the printed 0.0499 (R@20 0.1098 unchanged).
- GUME's per-user parameters are counted as three tables of width 64 (3.73 M on Baby); the footprint measurement counts one table (1.244 M).
- The Clothing SCOPE-U miss rate is 0.8911 from the coverage computation and 0.8912 as one minus the coverage of the headroom computation (16-bit summation order); Table E2 prints the coverage computation.
- On datasets with more than 20K items or 50K users (Clothing, MicroLens, Electronics) score matrices are stored in half precision and converted to single precision before ranking.
