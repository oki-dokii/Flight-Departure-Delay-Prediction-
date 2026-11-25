# Development drift analysis

Evidence label: **Full-development validation (population prevalence/decomposition); Pilot (feature PSI and rehearsal performance)**. **Held-out test** is closed; score cells remain blank.

Population rates and composition use the full paired eligible Phase A development population (5,749,040 targets). Feature PSI and model comparisons use the regenerated paired 20,000-flight pilot, not full-population matrices. The unmasked Phase A file and SHA-256 are recorded in p2_evidence.json. No raw or holdout data were read.

| Fold / validation | Train late rate | Validation late rate | Validation n | Composition change (pp) | Within-cell change (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / Jul | 20.98% | 28.44% | 616,011 | -0.040 | +7.507 |
| 2 / Aug | 22.12% | 22.55% | 596,812 | -0.142 | +0.575 |
| 3 / Sep | 22.24% | 16.56% | 559,894 | -0.227 | -5.457 |
| 4 / Oct | 21.65% | 19.35% | 585,051 | -0.030 | -2.275 |

The pattern is not monotone: late rates fall through September and rebound in October. Each validation month is compared with its own expanding training window, whose late fraction stays near 21–22%. The largest shifts occur within the selected cells, rather than from their changing flight shares.

Kitagawa decomposition uses carrier × origin-local scheduled six-hour bands. Composition = Σ(wV−wT)(rV+rT)/2; within = Σ(rV−rT)(wV+wT)/2. These sum exactly to the validation-minus-training prevalence change. For cells absent in one population, the observed rate is carried across, allocating their contribution to composition; entering/exiting masses and all cell supports are exported. This is descriptive, not causal: route mix, airport mix and shared weather shocks remain inside cells.

PSI numerics use deciles fitted only on the entire eligible training fold, with infinite outer bounds and a separate missing bin. Categorical levels with training share >=1% get separate bins; an explicit shared rare/unseen bucket and separate missing bucket retain coverage. Shares are floored at 1e-6 then renormalized. Edge/category definitions and raw counts are exported. KS is the maximum empirical-CDF distance on finite numeric values, an effect size only; no p-values are calculated. PSI 0.1/0.25 are conventions, not significance cutoffs.

Month and cyclic calendar fields are time-deterministic and flagged/excluded from the ranking; month is especially drifting by construction under forward chaining. Top non-calendar fold/checkpoint PSI values:
- origin_late_rate_3h, fold 1, 60 min: PSI 0.1387; KS 0.1685.
- origin_late_rate_3h, fold 1, 360 min: PSI 0.1156; KS 0.1686.
- origin_late_rate_3h, fold 3, 60 min: PSI 0.1135; KS 0.1305.

| LightGBM pilot / fold | Raw AP, 6h | Normalized AP, 6h | AP/prevalence, 6h | Raw AP, 1h | Normalized AP, 1h | AP/prevalence, 1h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4857 | 0.2798 | 1.698 | 0.5099 | 0.3136 | 1.783 |
| 2 | 0.4038 | 0.2078 | 1.632 | 0.4437 | 0.2607 | 1.793 |
| 3 | 0.3072 | 0.1748 | 1.914 | 0.3664 | 0.2452 | 2.283 |
| 4 | 0.3552 | 0.2141 | 1.979 | 0.3615 | 0.2218 | 2.014 |

Normalized AP = (AP − validation prevalence)/(1 − validation prevalence); lift = AP/prevalence. All models and paired model-minus-logistic differences at identical checkpoint/fold/seed are in drift_performance*.csv and drift_paired_differences.csv. The table shows means over training seeds, not overlapping standard-deviation error bars. These are rehearsal defaults, not selected tuned models.
At 60 min, LightGBM pooled AP std is 0.0612, within-fold seed RMS std 0.0030; fold means explain 99.76% of pooled variance.
At 360 min, LightGBM pooled AP std is 0.0663, within-fold seed RMS std 0.0064; fold means explain 99.07% of pooled variance.

Figures: drift_prevalence.png, drift_psi.png and drift_normalized_ap_{60,360}.png. Full P3 feature PSI is pending. Winter holidays and conditions may shift the November–December regime; no winter prevalence, feature distribution or performance is asserted.
