# app.py - myBridge minimal dark v5 (frozen v3 model outputs)
# Reads ONLY data/app_data.parquet and model/metrics.json (internal use only, never shown).
# Plain language everywhere except the engineer technical expander.

import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="myBridge", layout="wide")

LABELS = {
    "stable_id": "Bridge ID",
    "facility_carried": "Road",
    "features_intersected": "Crosses",
    "county_name": "County",
    "state_name": "State",
    "min_cond": "Overall condition today",
    "deck_cond": "Deck condition",
    "super_cond": "Superstructure condition",
    "sub_cond": "Substructure condition",
    "age_effective": "Age",
    "adt": "Vehicles per day",
    "pct_truck": "Truck share",
    "material": "Main material",
    "design": "Bridge design",
    "owner": "Owner",
    "operating_tons": "Operating weight limit",
    "inventory_tons": "Inventory weight limit",
    "deck_area_sqft": "Deck area",
    "scour": "Scour risk",
    "P_POOR_5YR": "Chance of needing major repair in next 5 years",
    "IS_ALREADY_POOR": "Needs major repair now",
    "RECOMMENDED_ACTION": "Recommended next step",
    "PROGRAM_YEAR": "Act by",
}

ACTION_LABELS = {
    "Replacement - Engineer Review": "Plan replacement",
    "Rehabilitation - Engineer Review": "Plan rehabilitation",
    "Preservation": "Keep maintaining",
}


@st.cache_data
def load_bridges():
    return pd.read_parquet("data/app_data.parquet")


@st.cache_data
def load_metrics():
    with open("model/metrics.json") as f:
        return json.load(f)


def clean(v):
    return "" if pd.isna(v) else str(v)


def num(v, fmt="{:,.0f}"):
    return "-" if pd.isna(v) else fmt.format(v)


def cond_word(v):
    if pd.isna(v):
        return "-"
    v = float(v)
    if v >= 7:
        return "Good"
    if v >= 5:
        return "Fair"
    return "Poor"


def risk_band(row):
    if row["IS_ALREADY_POOR"] == 1:
        return "Needs major repair now"
    p = row["P_POOR_5YR"]
    if p > 0.7:
        return "Very likely"
    if p > 0.3:
        return "Likely"
    return "Less likely"


def bridge_name(row):
    return f"{clean(row['facility_carried'])} over {clean(row['features_intersected'])}"


df = load_bridges().copy()
metrics = load_metrics()  # internal use only, never displayed
df["risk_band"] = df.apply(risk_band, axis=1)
df["action_plain"] = df["RECOMMENDED_ACTION"].map(ACTION_LABELS).fillna(df["RECOMMENDED_ACTION"])

if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_bridge" not in st.session_state:
    st.session_state.selected_bridge = df.sort_values("P_POOR_5YR", ascending=False).iloc[0]["stable_id"]
if "approved_list" not in st.session_state:
    st.session_state.approved_list = []
if "review_queue" not in st.session_state:
    st.session_state.review_queue = []
if "risk_filter" not in st.session_state:
    st.session_state.risk_filter = "all"
if "show_summary" not in st.session_state:
    st.session_state.show_summary = False


def go_to(page):
    st.session_state.page = page


def set_filter(name):
    st.session_state.risk_filter = name


def send_for_check(sid):
    if sid not in st.session_state.review_queue:
        st.session_state.review_queue.append(sid)
    st.session_state.page = "engineer"


def selected_row():
    sid = st.session_state.selected_bridge
    if sid not in set(df["stable_id"]):
        sid = df.sort_values("P_POOR_5YR", ascending=False).iloc[0]["stable_id"]
        st.session_state.selected_bridge = sid
    return sid, df[df["stable_id"] == sid].iloc[0]


# ---------------- Dark minimal theme ----------------
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stApp { font-family:'Inter',-apple-system,'SF Pro Display',sans-serif; }
.stApp { background-color:#0C0F14;
  background-image:radial-gradient(1200px 600px at 50% -10%, #1a2230 0%, #0C0F14 60%);
  color:#F5F5F7; }
[data-testid="stHeader"], footer, [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], #MainMenu { display:none !important; }
.block-container { max-width:1180px; padding-top:0 !important; padding-bottom:64px; }
.wordmark { text-align:center; font-weight:700; font-size:15px; letter-spacing:0.28em;
  text-transform:uppercase; color:#D4AF37; padding-top:40px; }
.hero { text-align:center; padding:8px 0 8px; }
.hero h1 { font-size:52px; font-weight:700; letter-spacing:-0.03em; margin:16px 0 32px;
  color:#F5F5F7; }
.topbar { display:flex; justify-content:space-between; align-items:center; padding:18px 4px; }
.topbar .brand { font-weight:700; font-size:18px; }
.mcard { background:#171C24; border-radius:18px;
  box-shadow:0 8px 28px rgba(0,0,0,0.35); padding:24px; margin:16px 0; }
div[data-testid="stVerticalBlockBorderWrapper"] > div { background:#171C24 !important;
  border:none !important; border-radius:18px !important;
  box-shadow:0 8px 28px rgba(0,0,0,0.35) !important; padding:24px !important; }
div[data-testid="stButton"] button[kind="primary"] { background:#D4AF37 !important;
  color:#0C0F14 !important; border:none !important; border-radius:980px !important;
  padding:12px 32px !important; font-weight:700 !important; }
div[data-testid="stButton"] button[kind="primary"]:hover { background:#E3C25C !important; }
div[data-testid="stButton"] button[kind="secondary"] { background:transparent !important;
  color:#F5F5F7 !important; border:1px solid #4A5568 !important;
  border-radius:980px !important; padding:12px 32px !important; font-weight:600 !important;
  box-shadow:none !important; }
div[data-testid="stButton"] button[kind="tertiary"] { background:transparent !important;
  color:#D4AF37 !important; border:none !important; font-weight:600 !important;
  box-shadow:none !important; padding:8px 12px !important; }
div[data-testid="stButton"] button:disabled { background:#2A3140 !important;
  color:#6B7280 !important; border:none !important; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
  border-radius:12px !important; }
[data-testid="stWidgetLabel"] p { font-size:13px !important; color:#9A9AA0 !important; }
.pill { display:inline-block; font-size:12px; font-weight:600; padding:5px 14px;
  border-radius:980px; color:#fff; }
.pill-now { background:#E5484D; } .pill-very { background:#E8A33D; color:#0C0F14; }
.pill-likely { background:#D4AF37; color:#0C0F14; } .pill-less { background:#4A5568; }
.pill-green { background:#1A9E54; } .pill-gray { background:#2A3140; color:#F5F5F7; }
.legend { display:flex; gap:8px; margin:0 0 12px; flex-wrap:wrap; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }
.lpill { background:#242B36; font-size:12px; font-weight:600; padding:5px 14px;
  border-radius:980px; color:#F5F5F7; }
.bigname { font-size:28px; font-weight:700; letter-spacing:-0.02em; }
.countyline { font-size:15px; color:#9A9AA0; margin-bottom:12px; }
.bigpct { font-size:48px; font-weight:700; letter-spacing:-0.03em; line-height:1.1; }
.specgrid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-top:20px; }
.speclabel { font-size:12px; color:#9A9AA0; text-transform:uppercase; letter-spacing:0.06em; }
.specval { font-size:17px; font-weight:600; margin-top:2px; }
.caption { font-size:13px; color:#9A9AA0; }
.footer { text-align:center; font-size:13px; color:#9A9AA0; margin-top:64px; }
.stepnum { font-size:40px; font-weight:800; color:#D4AF37; line-height:1; }
.steptitle { font-size:17px; font-weight:700; margin:8px 0 4px; }
.stepline { font-size:14px; color:#9A9AA0; }
.arrow { text-align:center; font-size:24px; color:#4A5568; padding-top:40px; }
.flowpill { background:#171C24; border:1px solid #2A3140; border-radius:980px;
  padding:10px 18px; font-size:14px; font-weight:600; text-align:center; }
.toplist button { width:100%; }
@keyframes draw { to { stroke-dashoffset:0; } }
@keyframes glow { 0%,100% { filter:drop-shadow(0 0 0 rgba(212,175,55,0)); }
  50% { filter:drop-shadow(0 0 14px rgba(212,175,55,0.55)); } }
.bridge-draw { stroke-dasharray:1400; stroke-dashoffset:1400;
  animation:draw 2.5s ease-out forwards; }
.bridge-glow { animation:glow 4s ease-in-out 2.5s infinite; }
</style>
""",
    unsafe_allow_html=True,
)

page = st.session_state.page

# ================= HOME =================
if page == "home":
    st.markdown('<div class="wordmark">myBridge</div>', unsafe_allow_html=True)
    st.markdown(
        """
<svg viewBox="0 0 560 200" style="width:min(560px,100%);opacity:0.95;display:block;margin:0 auto;" role="img" aria-label="Suspension bridge">
<g class="bridge-glow" fill="none" stroke="#D4AF37" stroke-width="3" stroke-linecap="round">
<g class="bridge-draw">
<path d="M20,140 L150,45 Q280,190 410,45 L540,140"/>
<path d="M20,140 L540,140"/>
<path d="M150,45 L150,140 M410,45 L410,140"/>
<path d="M132,70 L168,70 M392,70 L428,70"/>
<path d="M180,75 L180,140 M215,99 L215,140 M280,118 L280,140 M345,99 L345,140 M380,75 L380,140"/>
</g>
<path d="M120,168 L440,168" stroke-width="2" opacity="0.45"/>
</g>
<circle r="4.5" fill="#D4AF37">
<animateMotion dur="6s" repeatCount="indefinite" path="M20,140 L540,140"/>
</circle>
</svg>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero"><h1>Every bridge. Know when it needs you.</h1></div>',
        unsafe_allow_html=True,
    )
    b1, b2, b3, b4 = st.columns([1.5, 1.5, 1.5, 1.5], gap="large")
    b2.button("Explore Bridges", type="primary", width="stretch", on_click=go_to, args=("explore",))
    b3.button("Engineer Check", width="stretch", on_click=go_to, args=("engineer",))
    g1, g2, g3 = st.columns([1, 1, 1], gap="large")
    g2.button("How to use myBridge ->", type="tertiary", width="stretch", on_click=go_to, args=("guide",))

# ================= EXPLORE =================
elif page == "explore":
    st.markdown(
        '<div class="topbar"><div></div><div class="brand">myBridge</div><div></div></div>',
        unsafe_allow_html=True,
    )
    st.button("< Home", type="tertiary", on_click=go_to, args=("home",))

    q = st.text_input("Search", placeholder="Search a road, river, town, or county")
    pf = st.session_state.risk_filter
    p1, p2, p3 = st.columns(3, gap="medium")
    p1.button("All", type="primary" if pf == "all" else "secondary", width="stretch",
              on_click=set_filter, args=("all",))
    p2.button("Repair now", type="primary" if pf == "now" else "secondary", width="stretch",
              on_click=set_filter, args=("now",))
    p3.button("Repair soon", type="primary" if pf == "soon" else "secondary", width="stretch",
              on_click=set_filter, args=("soon",))
    rf = st.session_state.risk_filter

    f = df.copy()
    if rf == "now":
        f = f[f["IS_ALREADY_POOR"] == 1]
    elif rf == "soon":
        f = f[(f["IS_ALREADY_POOR"] == 0) & (f["P_POOR_5YR"] > 0.3)]
    if q.strip():
        ql = q.strip().lower()
        f = f[
            f["facility_carried"].fillna("").str.lower().str.contains(ql)
            | f["features_intersected"].fillna("").str.lower().str.contains(ql)
            | f["county_name"].fillna("").str.lower().str.contains(ql)
        ]
    if f.empty:
        st.warning("No bridges match. Try widening the search.")
    else:
        f = f.reset_index(drop=True)
        with st.container(border=True):
            st.markdown(
                '<div class="legend">'
                '<span class="lpill"><span class="dot" style="background:#E5484D"></span>Needs major repair now</span>'
                '<span class="lpill"><span class="dot" style="background:#E8A33D"></span>Very likely</span>'
                '<span class="lpill"><span class="dot" style="background:#D4AF37"></span>Likely</span>'
                '<span class="lpill"><span class="dot" style="background:#6E7681"></span>Less likely</span>'
                "</div>",
                unsafe_allow_html=True,
            )
            mapdf = pd.DataFrame({
                "Road": f["facility_carried"].fillna(""),
                "County": f["county_name"].fillna(""),
                "Chance in 5 years": f["P_POOR_5YR"],
                "Condition today": f["min_cond"].map(cond_word),
                "lat": f["lat"],
                "long": f["long"],
                "band": f["risk_band"],
                "size": f["P_POOR_5YR"] * 18 + 3,
                "sid": f["stable_id"],
            })
            fig = px.scatter_map(
                mapdf,
                lat="lat",
                lon="long",
                color="band",
                size="size",
                hover_name="Road",
                hover_data={
                    "Chance in 5 years": ":.0%",
                    "County": True,
                    "Condition today": True,
                    "lat": False,
                    "long": False,
                    "size": False,
                    "sid": False,
                },
                custom_data=["sid"],
                color_discrete_map={
                    "Needs major repair now": "#E5484D",
                    "Very likely": "#E8A33D",
                    "Likely": "#D4AF37",
                    "Less likely": "#6E7681",
                },
                category_orders={"band": ["Needs major repair now", "Very likely", "Likely", "Less likely"]},
                map_style="carto-darkmatter",
                height=500,
                opacity=0.7,
                size_max=14,
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                uirevision="map",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F5F5F7"),
            )
            sel = st.plotly_chart(fig, key="risk_map", on_select="rerun", width="stretch",
                                  config={"displayModeBar": False})
            if sel and sel.selection and sel.selection.get("points"):
                cd = sel.selection["points"][0].get("customdata")
                if cd:
                    st.session_state.selected_bridge = cd[0] if isinstance(cd, (list, tuple)) else cd

        top10 = f.sort_values("P_POOR_5YR", ascending=False).head(10)
        st.markdown("**Highest chance first**")
        for _, tr in top10.iterrows():
            if st.button(
                f"{bridge_name(tr)} - {clean(tr['county_name'])} - {tr['P_POOR_5YR']:.0%}",
                key=f"top_{tr['stable_id']}",
                width="stretch",
            ):
                st.session_state.selected_bridge = tr["stable_id"]

        sid, row = selected_row()
        band = row["risk_band"]
        pill = {"Needs major repair now": "pill-now", "Very likely": "pill-very",
                "Likely": "pill-likely"}.get(band, "pill-less")
        st.markdown(
            f'<div class="mcard"><div class="bigname">{bridge_name(row)}</div>'
            f'<div class="countyline">{clean(row["county_name"])}</div>'
            f'<div style="margin:6px 0 12px"><span class="pill {pill}">{band}</span></div>'
            f'<div class="bigpct">{row["P_POOR_5YR"]:.0%}</div>'
            f'<div class="caption">chance of needing major repair in the next 5 years</div>'
            f'<div class="specgrid">'
            f'<div><div class="speclabel">Condition today</div><div class="specval">{cond_word(row["min_cond"])}</div></div>'
            f'<div><div class="speclabel">Age</div><div class="specval">{num(row["age_effective"])} yrs</div></div>'
            f'<div><div class="speclabel">Vehicles per day</div><div class="specval">{num(row["adt"])}</div></div>'
            f'<div><div class="speclabel">Recommended next step</div><div class="specval">{clean(row["action_plain"])}</div></div>'
            f'<div><div class="speclabel">Act by</div><div class="specval">{row["PROGRAM_YEAR"]:.0f}</div></div>'
            f'<div><div class="speclabel">County</div><div class="specval">{clean(row["county_name"])}</div></div>'
            f"</div>"
            f'<div class="caption" style="margin-top:12px">Poor means needs major repair, not unsafe to cross.</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2, gap="large")
        if c1.button("Summary", type="primary", width="stretch"):
            st.session_state.show_summary = not st.session_state.show_summary
        c2.button("Send for Engineer Check ->", type="tertiary", width="stretch",
                  on_click=send_for_check, args=(sid,))
        if st.session_state.show_summary:
            with st.container(border=True):
                st.write(
                    f"{bridge_name(row)} has a {row['P_POOR_5YR']:.0%} chance of needing "
                    f"major repair in the next 5 years. Its condition today is "
                    f"{cond_word(row['min_cond'])} and it carries {num(row['adt'])} vehicles a day. "
                    f"Suggested next step: {clean(row['action_plain'])} - act by {row['PROGRAM_YEAR']:.0f}."
                )
        with st.expander("Technical details"):
            pairs = [("stable_id", sid)]
            for col in ["deck_cond", "super_cond", "sub_cond", "operating_tons",
                        "inventory_tons", "deck_area_sqft", "scour", "material",
                        "design", "owner", "min_cond", "year", "segment"]:
                if col in row.index:
                    v = row[col]
                    pairs.append((col, "" if pd.isna(v) else str(v)))
            st.dataframe(pd.DataFrame(pairs, columns=["Field", "Value"]),
                         hide_index=True, width="stretch")

# ================= ENGINEER =================
elif page == "engineer":
    n1, n2 = st.columns(2, gap="medium")
    n1.button("< Home", type="tertiary", on_click=go_to, args=("home",))
    n2.button("< Explore", type="tertiary", on_click=go_to, args=("explore",))
    st.markdown("## Engineer Check")
    st.markdown('<div class="caption">myBridge suggests. You confirm.</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Waiting for engineer check**")
        if not st.session_state.review_queue:
            st.markdown(
                '<div class="caption">Queue is empty. Send bridges here from Explore.</div>',
                unsafe_allow_html=True,
            )
        for qsid in st.session_state.review_queue:
            qr = df[df["stable_id"] == qsid]
            if qr.empty:
                continue
            qr = qr.iloc[0]
            if st.button(
                f"{bridge_name(qr)} - {clean(qr['county_name'])} - {qr['P_POOR_5YR']:.0%}"
                + ("  [Approved]" if qsid in st.session_state.approved_list else ""),
                key=f"q_{qsid}",
                width="stretch",
            ):
                st.session_state.selected_bridge = qsid

    sid, rrow = selected_row()
    with st.container(border=True):
        st.markdown(f"**Under review: {bridge_name(rrow)}**")
        st.markdown(
            f'<div class="caption">{clean(rrow["county_name"])} &mdash; '
            f"{rrow['risk_band']}, {rrow['P_POOR_5YR']:.0%} chance in 5 years.</div>",
            unsafe_allow_html=True,
        )
        if sid in st.session_state.approved_list:
            st.markdown('<span class="pill pill-green">Approved &#10003;</span>', unsafe_allow_html=True)
        er_name = st.text_input("Your name", key="er_name")
        er_license = st.text_input("License number", key="er_license")
        er_up = st.file_uploader("Upload load check (CSV)", type=["csv"], key="er_upload")
        if er_up is not None:
            er_df = pd.read_csv(er_up)
            st.write("Check the numbers before approving - edit below if needed:")
            st.data_editor(er_df, key=f"er_editor_{er_up.file_id}",
                           num_rows="dynamic", width="stretch")
        er_inv = st.number_input("Inventory check (RF)", min_value=0.0, value=0.0,
                                 step=0.01, key="er_inv_rf")
        er_op = st.number_input("Operating check (RF)", min_value=0.0, value=0.0,
                                step=0.01, key="er_op_rf")
        ready = bool(er_name.strip()) and bool(er_license.strip()) and (
            er_up is not None or (er_inv > 0 and er_op > 0)
        )
        e1, e2 = st.columns(2, gap="large")
        if e1.button("Approve", type="primary", disabled=not ready, key="er_approve"):
            if sid not in st.session_state.approved_list:
                st.session_state.approved_list.append(sid)
            st.success(f"{bridge_name(rrow)} approved by {er_name.strip()} (license {er_license.strip()}).")
        if e2.button("Reject", key="er_reject"):
            st.info(f"{bridge_name(rrow)} sent back - flagged for follow-up.")
    if st.session_state.approved_list:
        with st.container(border=True):
            st.markdown(f"**Approved for funding plan ({len(st.session_state.approved_list)})**")
            adf = df[df["stable_id"].isin(st.session_state.approved_list)]
            for _, ar in adf.iterrows():
                st.markdown(
                    f'<div class="caption">{bridge_name(ar)} &mdash; {clean(ar["county_name"])}</div>',
                    unsafe_allow_html=True,
                )
            dl = pd.DataFrame({
                "Bridge": adf["facility_carried"] + " over " + adf["features_intersected"].fillna(""),
                LABELS["county_name"]: adf["county_name"],
                "Overall condition today": adf["min_cond"].map(cond_word),
                "Chance in 5 years": (adf["P_POOR_5YR"] * 100).round(1).astype(str) + "%",
                LABELS["RECOMMENDED_ACTION"]: adf["action_plain"],
            })
            st.download_button("Download approved list (CSV)", dl.to_csv(index=False).encode(),
                               "mybridge_approved.csv", "text/csv")

# ================= GUIDE =================
elif page == "guide":
    st.button("< Home", type="tertiary", on_click=go_to, args=("home",))
    st.markdown("## How to use myBridge")
    steps = [
        ("1", "Find", "Search any road, town, river, or county."),
        ("2", "Read Summary", "Read the chance % and the suggested next step."),
        ("3", "Engineer Check", "An engineer verifies load capacity and approves."),
        ("4", "Approved Plan", "Funded bridges join the approved plan."),
    ]
    cols = st.columns([3, 1, 3, 1, 3, 1, 3], gap="small")
    for i, (n, title, line) in enumerate(steps):
        with cols[i * 2].container(border=True):
            st.markdown(
                f'<div class="stepnum">{n}</div><div class="steptitle">{title}</div>'
                f'<div class="stepline">{line}</div>',
                unsafe_allow_html=True,
            )
        if i < 3:
            cols[i * 2 + 1].markdown('<div class="arrow">&rarr;</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4, gap="medium")
    f1.markdown('<div class="flowpill">Past inspections</div>', unsafe_allow_html=True)
    f2.markdown('<div class="flowpill">myBridge flags at-risk</div>', unsafe_allow_html=True)
    f3.markdown('<div class="flowpill">Engineer confirms</div>', unsafe_allow_html=True)
    f4.markdown('<div class="flowpill">Plan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mcard"><div class="caption">'
        "1. Poor means needs major repair, not unsafe to cross.<br>"
        "2. The chance % comes from 42 years of inspections.<br>"
        "3. Only a licensed engineer approves a bridge for the plan."
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.button("Start Exploring", type="primary", on_click=go_to, args=("explore",))

st.markdown(
    '<div class="footer">myBridge &middot; Predictions from 42 years of inspections &middot; '
    "Llama explains, myBridge predicts, a licensed engineer decides.</div>",
    unsafe_allow_html=True,
)
