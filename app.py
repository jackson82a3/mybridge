# app.py - myBridge Streamlit app (frozen v3 model outputs)
# Reads ONLY data/app_data.parquet and model/metrics.json. Never data_NBI.csv.

import json

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="myBridge", layout="wide")


@st.cache_data
def load_bridges():
    return pd.read_parquet("data/app_data.parquet")


@st.cache_data
def load_metrics():
    with open("model/metrics.json") as f:
        return json.load(f)


def clean(v):
    return "" if pd.isna(v) else str(v)


def marker_color(row):
    if row["IS_ALREADY_POOR"] == 1 or row["P_POOR_5YR"] > 0.7:
        return "red"
    if row["P_POOR_5YR"] > 0.3:
        return "orange"
    return "green"


df = load_bridges()
metrics = load_metrics()

if "approved" not in st.session_state:
    st.session_state.approved = []

st.title("myBridge")

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
counties = sorted(df["county_name"].dropna().unique())
sel_counties = st.sidebar.multiselect("County", counties, default=counties)
actions = sorted(df["RECOMMENDED_ACTION"].dropna().unique())
sel_actions = st.sidebar.multiselect("Recommended action", actions, default=actions)
p_min = st.sidebar.slider("Minimum P_POOR_5YR", 0.0, 1.0, 0.0, 0.01)

f = df[
    df["county_name"].isin(sel_counties)
    & df["RECOMMENDED_ACTION"].isin(sel_actions)
    & (df["P_POOR_5YR"] >= p_min)
].copy()

# ---------- Approved list (sidebar) ----------
st.sidebar.subheader(f"Approved ({len(st.session_state.approved)})")
for sid in st.session_state.approved:
    st.sidebar.write(sid)

if f.empty:
    st.warning("No bridges match the current filters.")
    st.stop()

# ---------- Top metrics ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Bridges", f"{len(f):,}")
c2.metric("Already Poor", f"{int((f['IS_ALREADY_POOR'] == 1).sum()):,}")
c3.metric("At-Risk", f"{int(((f['P_POOR_5YR'] > 0.3) & (f['IS_ALREADY_POOR'] == 0)).sum()):,}")
c4.metric("Average P_POOR_5YR", f"{f['P_POOR_5YR'].mean():.3f}")

# ---------- Map ----------
st.subheader("Bridge Map")
m = folium.Map(location=[f["lat"].mean(), f["long"].mean()], zoom_start=7)
for _, r in f.iterrows():
    popup = folium.Popup(
        f"{clean(r['facility_carried'])} over {clean(r['features_intersected'])}"
        f"<br>County: {clean(r['county_name'])}"
        f"<br>min_cond: {r['min_cond']}<br>P_POOR_5YR: {r['P_POOR_5YR']:.3f}",
        max_width=300,
    )
    folium.CircleMarker(
        location=[r["lat"], r["long"]],
        radius=4,
        color=marker_color(r),
        fill=True,
        fill_opacity=0.7,
        popup=popup,
    ).add_to(m)
st_folium(m, width=1100, height=500, returned_objects=[])

# ---------- Priority table ----------
st.subheader("Priority List")
table_cols = ["stable_id", "facility_carried", "features_intersected", "county_name",
              "min_cond", "age_effective", "P_POOR_5YR", "RECOMMENDED_ACTION", "PROGRAM_YEAR"]
table_cols = [c for c in table_cols if c in f.columns]
ordered = f.sort_values("P_POOR_5YR", ascending=False)
st.dataframe(ordered[table_cols], width="stretch", hide_index=True)

# ---------- Bridge Card ----------
st.subheader("Bridge Card")
labels = {
    r["stable_id"]: f"{r['stable_id']} - {clean(r['facility_carried'])} "
                    f"({clean(r['county_name'])}, P={r['P_POOR_5YR']:.3f})"
    for _, r in ordered.iterrows()
}
sid = st.selectbox("Select bridge", options=list(labels.keys()), format_func=labels.get)
row = ordered[ordered["stable_id"] == sid].iloc[0]
st.dataframe(
    pd.DataFrame({"Field": row.index.astype(str), "Value": row.astype(str).values}),
    width="stretch", hide_index=True,
)
t1, t2 = st.columns(2)
t1.metric("Operating tons", f"{row['operating_tons']}")
t2.metric("Inventory tons", f"{row['inventory_tons']}")

# ---------- Engineer Review ----------
st.subheader("Engineer Review")
engineer_name = st.text_input("Engineer name", key="engineer_name")
pe_license = st.text_input("PE license", key="pe_license")
uploaded = st.file_uploader("Load rating CSV", type=["csv"], key="load_rating")
if uploaded is not None:
    rating_df = pd.read_csv(uploaded)
    st.write("PE verification - edit below if needed:")
    st.data_editor(rating_df, key=f"rating_editor_{uploaded.file_id}",
                   num_rows="dynamic", width="stretch")
inventory_rf = st.number_input("Inventory RF (manual)", min_value=0.0, value=0.0,
                               step=0.01, key="inventory_rf")
operating_rf = st.number_input("Operating RF (manual)", min_value=0.0, value=0.0,
                               step=0.01, key="operating_rf")

has_rating = uploaded is not None
has_manual = inventory_rf > 0 and operating_rf > 0
approve_ready = bool(engineer_name.strip()) and bool(pe_license.strip()) and (has_rating or has_manual)

b1, b2 = st.columns(2)
approve = b1.button("Approve", disabled=not approve_ready, key="approve_btn")
reject = b2.button("Reject", key="reject_btn")
if approve:
    if sid not in st.session_state.approved:
        st.session_state.approved.append(sid)
    st.success(f"Bridge {sid} approved by {engineer_name.strip()} (PE {pe_license.strip()}).")
if reject:
    st.info(f"Bridge {sid} rejected - flagged for follow-up.")

# ---------- Model Card ----------
with st.expander("Model Card"):
    def fmt(key):
        v = metrics.get(key)
        return f"{v:.4f}" if isinstance(v, (int, float)) else "-"

    card = pd.DataFrame(
        {
            "Markov": [fmt("markov_valid_auc"), fmt("markov_auc"), "-", "-", "-", "-"],
            "XGBoost": [fmt("xgb_valid_auc"), fmt("xgb_auc"), fmt("xgb_valid_prauc"),
                        fmt("xgb_prauc"), fmt("xgb_valid_recall_0.3"), fmt("poor_recall_0.3")],
        },
        index=["Valid AUC", "Test AUC", "Valid PR-AUC", "Test PR-AUC",
               "Valid Recall@0.3", "Test Recall@0.3"],
    )
    st.table(card)
    st.write("Training on NBI 1983-2025, target Poor within 5 years, temporal split only.")
