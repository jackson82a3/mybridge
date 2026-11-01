# app.py - myBridge, Apple-inspired minimal design (frozen v3 model outputs)
# Reads ONLY data/app_data.parquet and model/metrics.json. Never data_NBI.csv.

import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="myBridge", layout="wide")

COST_PER_SQFT = 300  # assumed rehab cost for program math, stated in the UI


@st.cache_data
def load_bridges():
    return pd.read_parquet("data/app_data.parquet")


@st.cache_data
def load_metrics():
    with open("model/metrics.json") as f:
        return json.load(f)


df = load_bridges().copy()
metrics = load_metrics()
df["risk_band"] = np.select(
    [(df["IS_ALREADY_POOR"] == 1) | (df["P_POOR_5YR"] > 0.7), df["P_POOR_5YR"] > 0.3],
    ["Critical", "High"],
    default="Monitor",
)

if "selected_bridge" not in st.session_state:
    st.session_state.selected_bridge = df.sort_values("P_POOR_5YR", ascending=False).iloc[0]["stable_id"]
if "approved_list" not in st.session_state:
    st.session_state.approved_list = []
if "review_queue" not in st.session_state:
    st.session_state.review_queue = []
if "show_why" not in st.session_state:
    st.session_state.show_why = False
if "show_how" not in st.session_state:
    st.session_state.show_how = False


def clean(v):
    return "" if pd.isna(v) else str(v)


def num(v, fmt="{:,.0f}"):
    return "-" if pd.isna(v) else fmt.format(v)


# ---------------- Design system ----------------
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stApp { font-family:'Inter',-apple-system,'SF Pro Display',sans-serif; }
.stApp { background:#F5F5F7; color:#1D1D1F; }
[data-testid="stHeader"], footer, [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], #MainMenu { display:none !important; }
.block-container { max-width:1180px; padding-top:0 !important; padding-bottom:64px; }
.topnav { position:sticky; top:0; z-index:999; display:flex; justify-content:space-between;
  align-items:center; padding:14px 4px; background:rgba(245,245,247,0.82);
  backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px); }
.brand { font-weight:700; font-size:20px; letter-spacing:-0.02em; }
.badgepill { background:#E9E9EE; color:#6E6E73; font-size:12px; font-weight:600;
  padding:6px 14px; border-radius:980px; }
.hero { text-align:center; padding:56px 0 8px; }
.eyebrow { font-size:12px; font-weight:700; letter-spacing:0.14em; color:#86868B; }
.hero h1 { font-size:52px; font-weight:700; letter-spacing:-0.03em; margin:12px 0; }
.herosub { font-size:17px; color:#6E6E73; max-width:640px; margin:0 auto; }
.mcard { background:#fff; border-radius:18px; box-shadow:0 4px 20px rgba(0,0,0,0.04);
  padding:28px 32px; margin:16px 0; }
div[data-testid="stVerticalBlockBorderWrapper"] > div { background:#fff !important;
  border:none !important; border-radius:18px !important;
  box-shadow:0 4px 20px rgba(0,0,0,0.04) !important; padding:24px 28px !important; }
.statgrid { display:flex; text-align:center; }
.stat { flex:1; padding:4px 8px; }
.stat + .stat { border-left:1px solid #E9E9EE; }
.statnum { font-size:32px; font-weight:700; letter-spacing:-0.02em; }
.statlabel { font-size:14px; color:#86868B; margin-top:4px; }
div[data-testid="stButton"] button[kind="primary"] { background:#1D1D1F !important;
  color:#fff !important; border:none !important; border-radius:980px !important;
  padding:10px 28px !important; font-weight:600 !important; }
div[data-testid="stButton"] button[kind="primary"]:hover { background:#000 !important; }
div[data-testid="stButton"] button[kind="secondary"] { background:transparent !important;
  color:#0066CC !important; border:none !important; font-weight:600 !important;
  box-shadow:none !important; }
div[data-testid="stButton"] button:disabled { background:#E5E5EA !important;
  color:#86868B !important; }
div[data-testid="stTabs"] [data-baseweb="tab-list"] { background:#E9E9EE;
  border-radius:980px; padding:4px; gap:2px; width:fit-content; margin:24px auto 0; }
div[data-testid="stTabs"] [data-baseweb="tab"] { border:none; border-radius:980px;
  color:#1D1D1F; font-weight:500; padding:8px 22px; }
div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] { background:#fff;
  box-shadow:0 1px 4px rgba(0,0,0,0.12); }
div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
div[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
  border-radius:12px !important; }
[data-testid="stWidgetLabel"] p { font-size:13px !important; color:#86868B !important; }
.pill { display:inline-block; font-size:12px; font-weight:600; padding:5px 14px;
  border-radius:980px; color:#fff; }
.pill-critical { background:#D70015; } .pill-high { background:#E85D04; }
.pill-monitor { background:#86868B; } .pill-green { background:#1A9E54; }
.pill-gray { background:#E9E9EE; color:#1D1D1F; }
.legend { display:flex; gap:8px; margin:0 0 12px; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }
.lpill { background:#F5F5F7; font-size:12px; font-weight:600; padding:5px 14px;
  border-radius:980px; color:#1D1D1F; }
.bigname { font-size:28px; font-weight:700; letter-spacing:-0.02em; }
.countyline { font-size:15px; color:#86868B; margin-bottom:12px; }
.bigpct { font-size:48px; font-weight:700; letter-spacing:-0.03em; line-height:1.1; }
.specgrid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-top:20px; }
.speclabel { font-size:12px; color:#86868B; text-transform:uppercase; letter-spacing:0.06em; }
.specval { font-size:17px; font-weight:600; margin-top:2px; }
.bignum { font-size:40px; font-weight:700; letter-spacing:-0.02em; text-align:center; }
.bignumlabel { font-size:14px; color:#86868B; text-align:center; }
.outcome { font-size:19px; font-weight:600; text-align:center; margin:8px 0 20px; }
.caption { font-size:13px; color:#86868B; }
.footer { text-align:center; font-size:13px; color:#86868B; margin-top:64px; }
.vsnum { font-size:40px; font-weight:700; letter-spacing:-0.02em; }
.vslabel { font-size:14px; color:#86868B; }
.queuerow { display:flex; justify-content:space-between; align-items:center;
  padding:10px 0; border-bottom:1px solid #F0F0F2; font-size:15px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Nav + Hero ----------------
st.markdown(
    '<div class="topnav"><div class="brand">myBridge</div>'
    '<div class="badgepill">XGBoost v3 &bull; NBI 1983-2025</div></div>',
    unsafe_allow_html=True,
)

total_n = len(df)
atrisk_n = int(((df["P_POOR_5YR"] > 0.3) & (df["IS_ALREADY_POOR"] == 0)).sum())
st.markdown(
    f'<div class="hero"><div class="eyebrow">FHWA NATIONAL BRIDGE INVENTORY</div>'
    f"<h1>Every bridge. Know when it needs you.</h1>"
    f'<div class="herosub">Monitoring {total_n:,} bridges &mdash; {atrisk_n:,} flagged '
    f"at-risk of turning Poor within 5 years.</div></div>",
    unsafe_allow_html=True,
)
h1, h2, h3, h4 = st.columns([2.5, 1.5, 1.5, 2.5], gap="large")
if h2.button("Find highest-risk bridge", type="primary", width="stretch"):
    top = df.sort_values("P_POOR_5YR", ascending=False).iloc[0]
    st.session_state.selected_bridge = top["stable_id"]
    st.toast(f"Highest risk: {top['stable_id']} at {top['P_POOR_5YR']:.1%} - see Map Explorer.")
if h3.button("How it works", width="stretch"):
    st.session_state.show_how = not st.session_state.show_how
if st.session_state.show_how:
    st.info(
        "myBridge scores every bridge on its chance of turning Poor within 5 years, "
        "ranks a fundable program under your budget, and routes each candidate to a "
        "licensed PE for review before any decision."
    )

# ---------------- Stats row ----------------
poor_pct = 100 * (df["IS_ALREADY_POOR"] == 1).mean()
st.markdown(
    f'<div class="mcard"><div class="statgrid">'
    f'<div class="stat"><div class="statnum">{total_n:,}</div>'
    f'<div class="statlabel">Total Bridges</div></div>'
    f'<div class="stat"><div class="statnum">{poor_pct:.1f}%</div>'
    f'<div class="statlabel">Already Poor</div></div>'
    f'<div class="stat"><div class="statnum">{atrisk_n:,}</div>'
    f'<div class="statlabel">At-Risk P&gt;0.3</div></div>'
    f'<div class="stat"><div class="statnum">{metrics.get("xgb_valid_auc", 0):.3f}</div>'
    f'<div class="statlabel">Valid AUC</div></div>'
    f"</div></div>",
    unsafe_allow_html=True,
)

# ---------------- Tabs ----------------
tab_overview, tab_map, tab_prog, tab_review, tab_model = st.tabs(
    ["Overview", "Map Explorer", "Priority Program", "Engineer Review", "Model Card"]
)

# ===== Overview / Network Overview =====
with tab_overview:
    st.markdown("### Network Overview")
    with st.container(border=True):
        budget = st.slider("Annual program budget", 10, 200, 50, 5, format="$%dM/yr")
        cand = df[(df["P_POOR_5YR"] > 0.3) & (df["IS_ALREADY_POOR"] == 0)].sort_values(
            "P_POOR_5YR", ascending=False
        ).copy()
        med_area = cand["deck_area_sqft"].median()
        cand["est_cost"] = cand["deck_area_sqft"].fillna(med_area) * COST_PER_SQFT
        funded = int((cand["est_cost"].cumsum() <= budget * 1_000_000).sum())
        deferred = len(cand) - funded
        st.markdown(
            f'<div class="outcome">With ${budget}M/yr, fund {funded} bridges '
            f"and defer {deferred} at-risk.</div>",
            unsafe_allow_html=True,
        )
        n1, n2 = st.columns(2, gap="large")
        n1.markdown(
            f'<div class="bignum">{funded}</div><div class="bignumlabel">Funded</div>',
            unsafe_allow_html=True,
        )
        n2.markdown(
            f'<div class="bignum">{deferred}</div><div class="bignumlabel">Deferred</div>',
            unsafe_allow_html=True,
        )
        bar = px.bar(
            pd.DataFrame({"Plan": ["Funded", "Deferred"], "Bridges": [funded, deferred]}),
            x="Plan",
            y="Bridges",
            color="Plan",
            color_discrete_map={"Funded": "#0071E3", "Deferred": "#D2D2D7"},
            height=260,
        )
        bar.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            margin=dict(l=8, r=8, t=8, b=8),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(bar, width="stretch")
        st.markdown(
            f'<div class="caption">Assumes ${COST_PER_SQFT}/sq ft of deck area, '
            f"highest risk funded first.</div>",
            unsafe_allow_html=True,
        )

# ===== Map Explorer =====
with tab_map:
    with st.container(border=True):
        f1, f2, f3 = st.columns([2, 2, 1.4], gap="large")
        q = f1.text_input("Search", placeholder="Facility / Features")
        counties = sorted(df["county_name"].dropna().unique())
        sel_counties = f2.multiselect("County", counties, default=counties)
        p_min = f3.slider("Min risk", 0.0, 1.0, 0.0, 0.01)
    f = df[
        df["county_name"].isin(sel_counties) & (df["P_POOR_5YR"] >= p_min)
    ].copy()
    if q.strip():
        ql = q.strip().lower()
        f = f[
            f["facility_carried"].fillna("").str.lower().str.contains(ql)
            | f["features_intersected"].fillna("").str.lower().str.contains(ql)
        ]
    if f.empty:
        st.warning("No bridges match the current filters.")
    else:
        f = f.reset_index(drop=True)
        f["marker_size"] = f["P_POOR_5YR"] * 18 + 3
        with st.container(border=True):
            st.markdown(
                '<div class="legend">'
                '<span class="lpill"><span class="dot" style="background:#D70015"></span>Critical</span>'
                '<span class="lpill"><span class="dot" style="background:#E85D04"></span>High</span>'
                '<span class="lpill"><span class="dot" style="background:#86868B"></span>Monitor</span>'
                "</div>",
                unsafe_allow_html=True,
            )
            fig = px.scatter_map(
                f,
                lat="lat",
                lon="long",
                color="risk_band",
                size="marker_size",
                hover_name="facility_carried",
                hover_data={
                    "P_POOR_5YR": ":.3f",
                    "county_name": True,
                    "min_cond": True,
                    "lat": False,
                    "long": False,
                    "marker_size": False,
                },
                custom_data=["stable_id"],
                color_discrete_map={
                    "Critical": "#D70015",
                    "High": "#E85D04",
                    "Monitor": "#86868B",
                },
                category_orders={"risk_band": ["Critical", "High", "Monitor"]},
                map_style="carto-positron",
                height=600,
                opacity=0.75,
                size_max=14,
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0), showlegend=False, uirevision="map"
            )
            sel = st.plotly_chart(fig, key="risk_map", on_select="rerun", width="stretch")
            if sel and sel.selection and sel.selection.get("points"):
                cd = sel.selection["points"][0].get("customdata")
                if cd:
                    st.session_state.selected_bridge = cd[0] if isinstance(cd, (list, tuple)) else cd

        sid = st.session_state.selected_bridge
        if sid not in set(df["stable_id"]):
            sid = df.sort_values("P_POOR_5YR", ascending=False).iloc[0]["stable_id"]
            st.session_state.selected_bridge = sid
        row = df[df["stable_id"] == sid].iloc[0]
        band = row["risk_band"]
        pill = {"Critical": "pill-critical", "High": "pill-high"}.get(band, "pill-monitor")
        st.markdown(
            f'<div class="mcard"><div class="bigname">{clean(row["facility_carried"])} '
            f"over {clean(row['features_intersected'])}</div>"
            f'<div class="countyline">{clean(row["county_name"])} &bull; {sid}</div>'
            f'<div class="bigpct">{row["P_POOR_5YR"]:.1%}</div>'
            f'<div class="caption">chance of Poor within 5 years</div>'
            f'<div style="margin-top:10px"><span class="pill {pill}">{band}</span></div>'
            f'<div class="specgrid">'
            f'<div><div class="speclabel">Min Cond</div><div class="specval">{row["min_cond"]:.0f}</div></div>'
            f'<div><div class="speclabel">Age</div><div class="specval">{num(row["age_effective"])} yrs</div></div>'
            f'<div><div class="speclabel">ADT</div><div class="specval">{num(row["adt"])}</div></div>'
            f'<div><div class="speclabel">Operating Tons</div><div class="specval">{num(row["operating_tons"], "{:,.1f}")}</div></div>'
            f'<div><div class="speclabel">Inventory Tons</div><div class="specval">{num(row["inventory_tons"], "{:,.1f}")}</div></div>'
            f'<div><div class="speclabel">Program Year</div><div class="specval">{row["PROGRAM_YEAR"]:.0f}</div></div>'
            f"</div></div>",
            unsafe_allow_html=True,
        )
        a1, a2 = st.columns(2, gap="large")
        if a1.button("Ask myBridge Why", type="primary", width="stretch"):
            st.session_state.show_why = not st.session_state.show_why
        if a2.button("Send to Engineer Review", width="stretch"):
            if sid not in st.session_state.review_queue:
                st.session_state.review_queue.append(sid)
            st.success(f"{sid} added to the Engineer Review queue.")
        if st.session_state.show_why:
            with st.container(border=True):
                st.markdown("**Why this bridge matters**")
                st.write(
                    f"P_POOR_5YR is {row['P_POOR_5YR']:.1%} - band **{band}**. "
                    f"Current worst condition {row['min_cond']:.0f} "
                    f"(deck {row['deck_cond']:.0f} / superstructure {row['super_cond']:.0f} / "
                    f"substructure {row['sub_cond']:.0f}), age {num(row['age_effective'])} years, "
                    f"{num(row['adt'])} vehicles per day."
                )
                st.write(f"Recommended: **{row['RECOMMENDED_ACTION']}** - program {row['PROGRAM_YEAR']:.0f}.")
                st.markdown(
                    '<div class="caption">Screening model only. A licensed PE makes the final call.</div>',
                    unsafe_allow_html=True,
                )

# ===== Priority Program =====
with tab_prog:
    prog = df.sort_values("P_POOR_5YR", ascending=False)
    prog_cols = ["stable_id", "facility_carried", "county_name", "min_cond",
                 "P_POOR_5YR", "RECOMMENDED_ACTION", "PROGRAM_YEAR"]
    prog_cols = [c for c in prog_cols if c in prog.columns]
    st.dataframe(
        prog[prog_cols],
        hide_index=True,
        width="stretch",
        column_config={
            "P_POOR_5YR": st.column_config.ProgressColumn(
                "Risk", min_value=0.0, max_value=1.0, format="%.3f"
            ),
        },
    )

# ===== Engineer Review =====
with tab_review:
    with st.container(border=True):
        st.markdown("**Review queue**")
        if not st.session_state.review_queue:
            st.markdown(
                '<div class="caption">Queue is empty. Send bridges here from Map Explorer.</div>',
                unsafe_allow_html=True,
            )
        for qsid in st.session_state.review_queue:
            qr = df[df["stable_id"] == qsid]
            if qr.empty:
                continue
            qr = qr.iloc[0]
            status = (
                '<span class="pill pill-green">Approved &#10003;</span>'
                if qsid in st.session_state.approved_list
                else '<span class="pill pill-gray">Pending</span>'
            )
            st.markdown(
                f'<div class="queuerow"><span>{qsid} &mdash; '
                f"{clean(qr['facility_carried'])} ({qr['P_POOR_5YR']:.1%})</span>"
                f"<span>{status}</span></div>",
                unsafe_allow_html=True,
            )
    rsid = st.session_state.selected_bridge
    rrow = df[df["stable_id"] == rsid].iloc[0]
    with st.container(border=True):
        st.markdown(f"**Under review: {rsid}**")
        if rsid in st.session_state.approved_list:
            st.markdown('<span class="pill pill-green">Approved &#10003;</span>', unsafe_allow_html=True)
        er_name = st.text_input("Engineer name", key="er_name")
        er_license = st.text_input("PE license", key="er_license")
        er_up = st.file_uploader("Load rating CSV", type=["csv"], key="er_upload")
        if er_up is not None:
            er_df = pd.read_csv(er_up)
            st.data_editor(er_df, key=f"er_editor_{er_up.file_id}",
                           num_rows="dynamic", width="stretch")
        er_inv = st.number_input("Inventory RF (manual)", min_value=0.0, value=0.0,
                                 step=0.01, key="er_inv_rf")
        er_op = st.number_input("Operating RF (manual)", min_value=0.0, value=0.0,
                                step=0.01, key="er_op_rf")
        ready = bool(er_name.strip()) and bool(er_license.strip()) and (
            er_up is not None or (er_inv > 0 and er_op > 0)
        )
        e1, e2 = st.columns(2, gap="large")
        if e1.button("Approve", type="primary", disabled=not ready, key="er_approve"):
            if rsid not in st.session_state.approved_list:
                st.session_state.approved_list.append(rsid)
            st.success(f"Bridge {rsid} approved by {er_name.strip()} (PE {er_license.strip()}).")
        if e2.button("Reject", key="er_reject"):
            st.info(f"Bridge {rsid} rejected - flagged for follow-up.")
    if st.session_state.approved_list:
        with st.container(border=True):
            st.markdown(f"**Approved ({len(st.session_state.approved_list)})**")
            for asid in st.session_state.approved_list:
                st.markdown(f'<div class="caption">{asid}</div>', unsafe_allow_html=True)

# ===== Model Card =====
with tab_model:
    mv = metrics.get("markov_valid_auc", 0)
    xv = metrics.get("xgb_valid_auc", 0)
    mt = metrics.get("markov_auc", 0)
    xt = metrics.get("xgb_auc", 0)
    with st.container(border=True):
        v1, v2 = st.columns(2, gap="large")
        v1.markdown(
            f'<div class="vslabel">Markov baseline</div><div class="vsnum">{mv:.2f}</div>'
            f'<div class="vslabel">Valid AUC</div>',
            unsafe_allow_html=True,
        )
        v2.markdown(
            f'<div class="vslabel">myBridge XGBoost</div><div class="vsnum">{xv:.2f}</div>'
            f'<div class="vslabel">Valid AUC</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="caption">Temporal split only - trained 2000-2015, validated '
            f"2016-2018, tested 2019-2020. Honest caveat: recent years drift, with test AUC "
            f"at Markov {mt:.2f} vs myBridge {xt:.2f}.</div>",
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer">myBridge &bull; Built on FHWA NBI &bull; Llama explains, '
    "XGBoost predicts, a licensed PE decides.</div>",
    unsafe_allow_html=True,
)
