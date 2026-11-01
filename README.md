Model Card: Trained on FHWA NBI 1983-2025, labeled rows 2000-2020, target = Poor (min condition <=4) within 5 years for bridges not already Poor. 
Temporal holdout only, no random split. On 2016-2018 validation: XGBoost AUC 0.900 vs Markov baseline 0.816, PR-AUC 0.331, Recall@0.3 0.751. 
On 2019-2020 test, performance degrades to AUC 0.794 vs Markov 0.806, indicating temporal drift - reported transparently. 
Final production model retrained on all 2000-2020 labeled rows.
