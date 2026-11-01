# train_mybridge.py - myBridge
# Input:  data_NBI.csv (1983-2025, local only, NOT committed to GitHub)
# Output: model/xgb_poor5yr.pkl, model/features.json, data/app_data.parquet
# v2: valid split + early stopping, rare-grouping, no county_code, unseen-category fix
# v3: scour kept as-is, 1200 trees/lr 0.03, full valid+test report, final retrain on 2000-2020 (FROZEN)

import pandas as pd, numpy as np, joblib, json, os
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier

INPUT = "data_NBI.csv"

# ---------------- 1. LOAD ----------------
print("1. Loading data_NBI.csv...")
df_raw = pd.read_csv(INPUT, dtype=str, low_memory=False)
print(f"   Raw rows: {len(df_raw):,}")

rename = {
 "Year":"year", "1 - State Code":"state_code", "1 - State Name":"state_name",
 "8 - Structure Number":"structure_number", "2025 NBI Structure Number":"structure_number_2025",
 "22 - Owner Agency":"owner", "27 - Year Built":"year_built",
 "29 - Average Daily Traffic":"adt", "43A - Main Span Material":"material",
 "43B - Main Span Design":"design", "45 - Number of Spans in Main Unit":"num_spans",
 "49 - Structure Length (ft.)":"structure_length_ft", "CAT10 - Bridge Condition":"cat10",
 "Bridge Age (yr)":"age_reported", "CAT29 - Deck Area (sq. ft.)":"deck_area_sqft",
 "16 - Latitude (decimal)":"lat", "17 - Longitude (decimal)":"long",
 "106 - Year Reconstructed":"year_reconstructed", "48 - Length of Maximum Span (ft.)":"max_span_ft",
 "34 - Skew Angle (degrees)":"skew", "58 - Deck Condition Rating":"deck_cond",
 "59 - Superstructure Condition Rating":"super_cond", "60 - Substructure Condition Rating":"sub_cond",
 "61 - Channel and Channel Protection Condition Rating":"channel_cond",
 "113 - Scour Critical Bridge Value":"scour", "64 - Operating Rating (US tons)":"operating_tons",
 "66 - Inventory Rating (US tons)":"inventory_tons", "109 - Average Daily Truck Traffic (Percent ADT)":"pct_truck",
 "96 - Total Project Cost":"total_project_cost", "3 - County Code":"county_code",
 "3 - County Name":"county_name", "6A - Features Intersected":"features_intersected",
 "7 - Facility Carried By Structure":"facility_carried",
}
df = df_raw.rename(columns=rename)

num_cols = ["year","year_built","year_reconstructed","adt","num_spans","structure_length_ft",
 "max_span_ft","deck_area_sqft","skew","lat","long","pct_truck","operating_tons","inventory_tons",
 "deck_cond","super_cond","sub_cond","channel_cond"]
for c in num_cols:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

print("   Year counts (tail):")
print(df["year"].value_counts().sort_index().tail(10))

# ---------------- 2. CLEAN + ID ----------------
print("2. Cleaning and building IDs...")
df["min_cond"] = df[["deck_cond","super_cond","sub_cond"]].min(axis=1)
df = df[df["min_cond"].between(0,9)].copy()

df["structure_number_2025"] = df["structure_number_2025"].fillna(df["structure_number"])
df["stable_id"] = df["state_code"].astype(str).str.strip() + "_" + df["structure_number_2025"].astype(str).str.strip()
# Segment: a new Year Built = replacement = new bridge
df["segment"] = df["stable_id"] + "_" + df["year_built"].fillna(0).astype(int).astype(str)

# 1 row per segment+year, keep worst condition
df = df.sort_values(["segment","year","min_cond"]).drop_duplicates(["segment","year"], keep="first")
df = df.sort_values(["segment","year"]).reset_index(drop=True)

# ---------------- 3. FEATURES AT TIME t ----------------
print("3. Building features...")
# year_reconstructed <= year_built = never reconstructed
df["recon_year"] = df["year_reconstructed"].where(df["year_reconstructed"] > df["year_built"])
df["age_effective"] = df["year"] - df["recon_year"].fillna(df["year_built"])
df["years_since_recon"] = df["year"] - df["recon_year"]
df["log_adt"] = np.log1p(df["adt"].clip(lower=0))

# Clip extreme data-entry errors
for c in ["deck_area_sqft","structure_length_ft","max_span_ft","adt"]:
    if c in df.columns:
        df[c] = df[c].clip(upper=df[c].quantile(0.995))

g = df.groupby("segment", sort=False)
df["min_cond_lag"] = g["min_cond"].shift(1)
df["years_since_last"] = g["year"].diff()
df["cond_drop_rate"] = (df["min_cond_lag"] - df["min_cond"]) / df["years_since_last"]
df["cond_drop_rate"] = df["cond_drop_rate"].replace([np.inf,-np.inf], np.nan).fillna(0)

# ---------------- 4. BUILD TARGET: POOR WITHIN 5 YEARS ----------------
print("4. Building IS_POOR_5YR target (this takes 1-2 min)...")
base = df[["segment","year","min_cond","recon_year"]].copy()
fut = df[["segment","year","min_cond","recon_year"]].rename(
    columns={"year":"fut_year","min_cond":"fut_min_cond","recon_year":"fut_recon_year"})

m = base.merge(fut, on="segment")
m["gap"] = m["fut_year"] - m["year"]
m = m[(m["gap"] >= 1) & (m["gap"] <= 6)].copy()

# Censor: reconstruction happened between t and the future inspection
m["was_recon"] = (m["fut_recon_year"] > m["year"]) & (m["fut_recon_year"] <= m["fut_year"])
# Censor: condition magically jumped +2 or more (undocumented rehab) - only check 4-6yr snapshot
snapshot = m[m["gap"] >= 4].sort_values("gap").groupby(["segment","year"]).first().reset_index()
censored_keys = set(zip(snapshot.loc[(snapshot["was_recon"]) | ((snapshot["fut_min_cond"]-snapshot["min_cond"])>=2), "segment"],
                        snapshot.loc[(snapshot["was_recon"]) | ((snapshot["fut_min_cond"]-snapshot["min_cond"])>=2), "year"]))

# Aggregate within-5yr window
win = m[m["gap"] <= 5]
agg = win.groupby(["segment","year"]).agg(
    went_poor=("fut_min_cond", lambda s: int((s <= 4).any())),
    had_recon_win=("was_recon", "any")
).reset_index()

# Need a follow-up inspection at 4-6 years to confirm a 0 label
has_followup = snapshot[["segment","year"]].assign(has_followup=1)

labels = agg.merge(has_followup, on=["segment","year"], how="inner")
labels = labels[~labels["had_recon_win"]]
labels["IS_POOR_5YR"] = labels["went_poor"]
labels = labels[~labels.apply(lambda r: (r["segment"], r["year"]) in censored_keys, axis=1)]

train_df = df.merge(labels[["segment","year","IS_POOR_5YR"]], on=["segment","year"], how="inner")

# *** KEY FIX: Only predict crossing to Poor from Good/Fair ***
train_df = train_df[train_df["min_cond"] > 4].copy()
print(f"   Labeled non-Poor rows: {len(train_df):,} | Poor-within-5yr rate: {train_df['IS_POOR_5YR'].mean():.1%}")

# ---------------- 5. TEMPORAL SPLIT (v2: +valid, rare-grouping, drop county_code) ----------------
print("5. Temporal split (no random split)...")
features = ["min_cond","deck_cond","super_cond","sub_cond","channel_cond",
 "min_cond_lag","cond_drop_rate","age_effective","years_since_recon",
 "log_adt","pct_truck","num_spans","structure_length_ft","max_span_ft",
 "deck_area_sqft","skew","state_code","owner","material","design","scour"]
features = [f for f in features if f in train_df.columns]
cats = ["state_code","owner","material","design","scour"]

# v2: group rare categories (<200 rows in train_df) into "Other"
# v3: scour keeps known codes as-is; only values <50 rows go to Other
SCOUR_KNOWN = set(list("0123456789") + ["A", "B", "C", "U", "T"])
kept_values = {}
for c in ["material","design","owner","scour","state_code","county_code"]:
    if c in train_df.columns:
        vc = train_df[c].astype(str).value_counts()
        if c == "scour":
            kept = set(vc[vc >= 50].index) | (SCOUR_KNOWN & set(vc.index))
        else:
            kept = set(vc[vc >= 200].index)
        kept_values[c] = kept
        s = train_df[c].astype(str)
        n_rare = int((~s.isin(kept)).sum())
        train_df[c] = s.where(s.isin(kept), "Other")
        print(f"   {c}: {len(kept)} kept, {n_rare:,} rows -> Other")

for c in cats:
    if c in features: train_df[c] = train_df[c].astype(str).astype("category")

train = train_df[(train_df["year"] >= 2000) & (train_df["year"] <= 2015)]
valid = train_df[(train_df["year"] >= 2016) & (train_df["year"] <= 2018)]
test  = train_df[(train_df["year"] >= 2019) & (train_df["year"] <= 2020)]
print(f"   Train 2000-2015: {len(train):,} | Valid 2016-2018: {len(valid):,} | Test 2019-2020: {len(test):,}")

X_train, y_train = train[features], train["IS_POOR_5YR"]
X_valid, y_valid = valid[features], valid["IS_POOR_5YR"]
X_test, y_test = test[features], test["IS_POOR_5YR"]

# ---------------- 6. MARKOV BASELINE ----------------
print("6. Markov baseline (empirical P(Poor in 5yr) by current min_cond)...")
markov_map = train.groupby("min_cond")["IS_POOR_5YR"].mean()
valid_markov_score = valid["min_cond"].map(markov_map).fillna(markov_map.mean())
test_markov_score = test["min_cond"].map(markov_map).fillna(markov_map.mean())
markov_valid_auc = roc_auc_score(y_valid, valid_markov_score)
markov_auc = roc_auc_score(y_test, test_markov_score)
print(f"   Markov Valid AUC: {markov_valid_auc:.3f} | Test AUC: {markov_auc:.3f} | Map: {markov_map.to_dict()}")

# ---------------- 7. TRAIN XGBOOST (v3: 1200 trees, lr 0.03, ES 80) ----------------
print("7. Training XGBoost (v3)...")
neg, pos = (y_train==0).sum(), (y_train==1).sum()
xgb_params = dict(
    n_estimators=1200, max_depth=5, learning_rate=0.03,
    subsample=0.9, colsample_bytree=0.9,
    scale_pos_weight=neg/max(pos,1),
    enable_categorical=True, eval_metric="logloss", random_state=42, n_jobs=-1,
    early_stopping_rounds=80,
)
model = XGBClassifier(**xgb_params)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
print(f"   Best iteration: {model.best_iteration} / 1200")

pred_valid = model.predict_proba(X_valid)[:,1]
pred = model.predict_proba(X_test)[:,1]
xgb_valid_auc = roc_auc_score(y_valid, pred_valid)
xgb_test_auc = roc_auc_score(y_test, pred)
xgb_valid_prauc = average_precision_score(y_valid, pred_valid)
xgb_test_prauc = average_precision_score(y_test, pred)
rec_valid_03 = recall_score(y_valid, (pred_valid > 0.3).astype(int))
rec_test_03 = recall_score(y_test, (pred > 0.3).astype(int))
rec_valid_05 = recall_score(y_valid, (pred_valid > 0.5).astype(int))
rec_test_05 = recall_score(y_test, (pred > 0.5).astype(int))
print(f"   XGBoost v3 Valid: AUC={xgb_valid_auc:.3f} PR-AUC={xgb_valid_prauc:.3f} Recall@0.3={rec_valid_03:.3f} Recall@0.5={rec_valid_05:.3f}")
print(f"   XGBoost v3 Test:  AUC={xgb_test_auc:.3f} PR-AUC={xgb_test_prauc:.3f} Recall@0.3={rec_test_03:.3f} Recall@0.5={rec_test_05:.3f}")
print(f"   Markov    Valid: AUC={markov_valid_auc:.3f} | Test AUC={markov_auc:.3f}")
pred_bin = (pred > 0.3).astype(int)
print(f"   Confusion Matrix (test @0.3):\n{confusion_matrix(y_test, pred_bin)}")

# v2 ablation: XGBoost on min_cond only (same setup, isolates feature value)
print("   Ablation: XGBoost on min_cond only...")
model_1f = XGBClassifier(**{**xgb_params, "enable_categorical": False})
model_1f.fit(X_train[["min_cond"]], y_train, eval_set=[(X_valid[["min_cond"]], y_valid)], verbose=False)
pred_1f = model_1f.predict_proba(X_test[["min_cond"]])[:,1]
xgb_mincond_auc = roc_auc_score(y_test, pred_1f)
print(f"   XGBoost min_cond-only Test AUC: {xgb_mincond_auc:.3f} (best iter {model_1f.best_iteration})")

# Feature importance
imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("   Top 15 features:\n", imp.head(15))

# ---------------- 8. SAVE MODEL + PREDICT LATEST (2025) ----------------
print("8. Saving model and predicting latest year...")
os.makedirs("model", exist_ok=True); os.makedirs("data", exist_ok=True)

# v3: FINAL app model retrained on all labeled rows 2000-2020 (train+valid+test),
# using the best iteration count from the holdout run. Reported metrics are from
# the holdout run above, NOT from this retrain.
full = train_df[(train_df["year"] >= 2000) & (train_df["year"] <= 2020)]
X_full, y_full = full[features], full["IS_POOR_5YR"]
n_final = int(model.best_iteration) + 1
neg_f, pos_f = (y_full == 0).sum(), (y_full == 1).sum()
final_params = {**xgb_params, "n_estimators": n_final,
                "scale_pos_weight": neg_f / max(pos_f, 1)}
del final_params["early_stopping_rounds"]
final_model = XGBClassifier(**final_params)
final_model.fit(X_full, y_full, verbose=False)
print(f"   FINAL model: {len(full):,} rows 2000-2020, {n_final} trees")

joblib.dump(final_model, "model/xgb_poor5yr.pkl")
json.dump(features, open("model/features.json","w"))
json.dump({"markov_valid_auc": float(markov_valid_auc), "markov_auc": float(markov_auc),
           "xgb_valid_auc": float(xgb_valid_auc), "xgb_auc": float(xgb_test_auc),
           "xgb_valid_prauc": float(xgb_valid_prauc), "xgb_prauc": float(xgb_test_prauc),
           "xgb_valid_recall_0.3": float(rec_valid_03), "poor_recall_0.3": float(rec_test_03),
           "xgb_valid_recall_0.5": float(rec_valid_05), "xgb_recall_0.5": float(rec_test_05),
           "xgb_mincond_auc": float(xgb_mincond_auc),
           "best_iteration": int(model.best_iteration), "final_trees": n_final,
           "final_rows": int(len(full))}, open("model/metrics.json","w"), indent=2)

latest = df.sort_values("year").drop_duplicates("segment", keep="last").copy()
for c in cats:
    if c in features:
        # v2: same rare->Other mapping as training, then align to training
        # categories so unseen values become NaN instead of crashing
        if c in kept_values:
            s = latest[c].astype(str)
            latest[c] = s.where(s.isin(kept_values[c]), "Other")
        latest[c] = pd.Categorical(latest[c].astype(str), categories=X_train[c].cat.categories)

latest["P_POOR_5YR"] = final_model.predict_proba(latest[features])[:,1].round(3)
latest["IS_ALREADY_POOR"] = (latest["min_cond"] <= 4).astype(int)
latest["RECOMMENDED_ACTION"] = np.select(
    [latest["IS_ALREADY_POOR"]==1, latest["P_POOR_5YR"] > 0.7, latest["P_POOR_5YR"] > 0.3],
    ["Replacement - Engineer Review","Replacement - Engineer Review","Rehabilitation - Engineer Review"],
    default="Preservation")
latest["PROGRAM_YEAR"] = np.select(
    [latest["IS_ALREADY_POOR"]==1, latest["P_POOR_5YR"] > 0.7, latest["P_POOR_5YR"] > 0.3],
    [latest["year"]+1, latest["year"]+1, latest["year"]+2], default=latest["year"]+5).astype(int)

keep_app = ["stable_id","segment","facility_carried","features_intersected","county_name","state_name",
 "lat","long","year","min_cond","deck_cond","super_cond","sub_cond","age_effective",
 "adt","pct_truck","material","design","owner","operating_tons","inventory_tons","deck_area_sqft",
 "P_POOR_5YR","IS_ALREADY_POOR","RECOMMENDED_ACTION","PROGRAM_YEAR"]
keep_app = [c for c in keep_app if c in latest.columns]
app_df = latest[keep_app].dropna(subset=["lat","long"])
app_df.to_parquet("data/app_data.parquet", index=False)

print(f"   Saved model/xgb_poor5yr.pkl")
print(f"   Saved data/app_data.parquet: {len(app_df):,} bridges, {os.path.getsize('data/app_data.parquet')/1024/1024:.1f} MB")
print("   First 5 rows of app_data.parquet:")
check = pd.read_parquet("data/app_data.parquet")
print(check.head(5).to_string())
print(f"   lat null: {int(check['lat'].isna().sum())}, long null: {int(check['long'].isna().sum())} / {len(check):,}")
print("DONE.")
