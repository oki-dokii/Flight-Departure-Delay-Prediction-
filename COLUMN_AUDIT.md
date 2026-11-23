# Column audit

`COLUMN_AUDIT.csv` inventories the actual development parquet schema. Known schedule/geographic fields are eligible under the schedule-as-known assumption. Target-flight outcomes are forbidden predictors even though they are valid descriptive EDA inputs. Unknown columns default to unavailable.

Conditional entries remain blocked by `src.eda.approved_features`: final tail assignment, its missingness indicator, and reconstructed schedule timestamps require P1/P3 review. Do not pass all numeric columns to the evaluation harness. Derived history/weather features require their own registry of source columns and availability timestamps. A source audit cannot establish that an aggregate is safe.

Prior-flight events require event and assumed feed-availability times no later than the target checkpoint. Final BTS data is retrospective, not a live feed. See the implementation plan for unresolved DST, arrival validation and aircraft continuity decisions.

Dictionary: https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ
