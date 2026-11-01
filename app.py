# app.py - myBridge product v3 (frozen v3 model outputs)
# Reads ONLY data/app_data.parquet and model/metrics.json. Never data_NBI.csv.
# Plain language everywhere except Engineer Technical Details and How myBridge Works.

import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="myBridge", layout="wide")

COST_PER_SQFT = 300  # planning assumption for program math, stated in the UI

LABELS = {
    "stable_id": "Bridge ID",
    "segment": "Bridge record",
    "facility_carried": "Road",
    "features_intersected": "Crosses",
    "county_name": "County",
    "state_name": "State",
    "lat": "Latitude",
    "long": "Longitude",
    "year": "Latest check year",
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

NAV_OPTIONS = ["Home", "Find a Bridge", "What to Fix First", "Engineer Check", "How myBridge Works"]


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
metrics = load_metrics()
df["risk_band"] = df.apply(risk_band, axis=1)
df["action_plain"] = df["RECOMMENDED_ACTION"].map(ACTION_LABELS).fillna(df["RECOMMENDED_ACTION"])

if "selected_bridge" not in st.session_state:
    st.session_state.selected_bridge = df.sort_values("P_POOR_5YR", ascending=False).iloc[0]["stable_id"]
if "approved_list" not in st.session_state:
    st.session_state.approved_list = []
if "review_queue" not in st.session_state:
    st.session_state.review_queue = []
if "shortcut" not in st.session_state:
    st.session_state.shortcut = "all"
if "funded_only" not in st.session_state:
    st.session_state.funded_only = False
if "funded_ids" not in st.session_state:
    st.session_state.funded_ids = []
if "show_how" not in st.session_state:
    st.session_state.show_how = False
if "explain" not in st.session_state:
    st.session_state.explain = False


def go_to(section):
    st.session_state.nav = section


def set_shortcut(name, section="Find a Bridge"):
    st.session_state.shortcut = name
    st.session_state.funded_only = False
    st.session_state.nav = section


def clear_filters():
    st.session_state.shortcut = "all"
    st.session_state.funded_only = False


def apply_shortcut(frame):
    s = st.session_state.shortcut
    if s == "now":
        return frame[frame["IS_ALREADY_POOR"] == 1]
    if s == "soon":
        return frame[(frame["IS_ALREADY_POOR"] == 0) & (frame["P_POOR_5YR"] > 0.3)]
    if s == "attention":
        return frame[(frame["IS_ALREADY_POOR"] == 1) | (frame["P_POOR_5YR"] > 0.3)]
    return frame


SHORTCUT_LABELS = {
    "all": "Showing: all bridges",
    "now": "Showing: needs major repair now",
    "soon": "Showing: could need major repair within 5 years",
    "attention": "Showing: bridges needing attention first",
}


def shortcut_note():
    st.markdown(
        f'<div class="caption">{SHORTCUT_LABELS[st.session_state.shortcut]}'
        + (" &bull; funded set only" if st.session_state.funded_only else "")
        + "</div>",
        unsafe_allow_html=True,
    )
    if st.session_state.shortcut != "all" or st.session_state.funded_only:
        st.button("Clear filters", on_click=clear_filters)


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
div[data-testid="stRadio"] > div { justify-content:center; gap:4px;
  background:#E9E9EE; border-radius:980px; padding:4px 8px; width:fit-content; margin:8px auto 0; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
  border-radius:12px !important; }
[data-testid="stWidgetLabel"] p { font-size:13px !important; color:#86868B !important; }
.pill { display:inline-block; font-size:12px; font-weight:600; padding:5px 14px;
  border-radius:980px; color:#fff; }
.pill-now { background:#D70015; } .pill-very { background:#E85D04; }
.pill-likely { background:#E9A13B; } .pill-less { background:#86868B; }
.pill-green { background:#1A9E54; } .pill-gray { background:#E9E9EE; color:#1D1D1F; }
.legend { display:flex; gap:8px; margin:0 0 12px; flex-wrap:wrap; }
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
.guidecard h4 { margin:0 0 6px; font-size:19px; }
.guidecard p { margin:0; font-size:15px; color:#6E6E73; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="topnav"><div class="brand">myBridge</div>'
    '<div class="badgepill">Built on 42 years of inspections</div></div>',
    unsafe_allow_html=True,
)

st.segmented_control("Who is this for?", ["Everyone", "Owner", "Engineer"],
                     default="Everyone", key="audience")
audience = st.session_state.audience

st.radio("Go to", NAV_OPTIONS, horizontal=True, key="nav", label_visibility="collapsed")
nav = st.session_state.nav

total_n = len(df)
poor_n = int((df["IS_ALREADY_POOR"] == 1).sum())
soon_n = int(((df["IS_ALREADY_POOR"] == 0) & (df["P_POOR_5YR"] > 0.3)).sum())


def budget_section(open_card):
    with st.container(border=True):
        st.markdown("**Program budget**")
        budget = st.slider(
            "Annual program budget", 10, 200, 50, 5, format="$%dM/yr", key=f"budget_{open_card}",
            help="Planning estimate only. Assumes cost by deck area, highest chance funded first.",
        )
        cand = df[(df["P_POOR_5YR"] > 0.3) & (df["IS_ALREADY_POOR"] == 0)].sort_values(
            "P_POOR_5YR", ascending=False
        ).copy()
        med_area = cand["deck_area_sqft"].median()
        cand["est_cost"] = cand["deck_area_sqft"].fillna(med_area) * COST_PER_SQFT
        funded_mask = cand["est_cost"].cumsum() <= budget * 1_000_000
        funded_ids = cand.loc[funded_mask, "stable_id"].tolist()
        funded = len(funded_ids)
        watch = len(cand) - funded
        st.markdown(
            f'<div class="outcome">With ${budget}M a year, fund {funded} bridges '
            f"and keep {watch} on the watch list.</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, funded / max(len(cand), 1)))
        st.markdown(
            f'<div class="caption">{funded} of {len(cand)} bridges that could need '
            f"major repair are covered.</div>",
            unsafe_allow_html=True,
        )

        def see_funded(ids=funded_ids):
            st.session_state.funded_ids = ids
            st.session_state.funded_only = True
            st.session_state.shortcut = "all"
            st.session_state.nav = "What to Fix First"

        st.button("See funded bridges", type="primary", on_click=see_funded)
    return funded_ids


# ================= HOME =================
if nav == "Home":
    st.markdown(
        f'<div class="hero"><div class="eyebrow">FHWA NATIONAL BRIDGE INVENTORY</div>'
        f"<h1>Every bridge. Know when it needs you.</h1>"
        f'<div class="herosub">Monitoring {total_n:,} bridges &mdash; {poor_n:,} need '
        f"major repair now and {soon_n:,} could need it within 5 years.</div></div>",
        unsafe_allow_html=True,
    )
    st.button("See bridges needing attention first", type="primary", width="stretch",
              on_click=set_shortcut, args=("attention", "What to Fix First"))
    h2a, h2b, h2c = st.columns([1, 2, 1], gap="large")
    if h2b.button("How myBridge works (1 minute)", width="stretch"):
        st.session_state.show_how = not st.session_state.show_how
    if st.session_state.show_how:
        st.info(
            "myBridge looks at 42 years of inspection history, scores every bridge on its "
            "chance of needing major repair within 5 years, ranks a fundable program under "
            "your budget, and routes each candidate to a licensed engineer for review."
        )

    st.markdown(
        f'<div class="mcard"><div class="statgrid">'
        f'<div class="stat"><div class="statnum">{total_n:,}</div>'
        f'<div class="statlabel">Total bridges checked</div></div>'
        f'<div class="stat"><div class="statnum">{poor_n:,}</div>'
        f'<div class="statlabel">Need major repair now ({100 * poor_n / total_n:.1f}%)</div></div>'
        f'<div class="stat"><div class="statnum">{soon_n:,}</div>'
        f'<div class="statlabel">Could need major repair within 5 years</div></div>'
        f'<div class="stat"><div class="statnum">{len(st.session_state.review_queue):,}</div>'
        f'<div class="statlabel">Waiting for engineer check</div></div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )
    s1, s2, s3 = st.columns(3, gap="large")
    s1.button("Show: all bridges", on_click=set_shortcut, args=("all", "Find a Bridge"))
    s2.button("Show: repair now", on_click=set_shortcut, args=("now", "Find a Bridge"))
    s3.button("Show: repair within 5 years", on_click=set_shortcut, args=("soon", "Find a Bridge"))

    st.markdown("### Start here")
    g1, g2, g3 = st.columns(3, gap="large")
    with g1.container(border=True):
        st.markdown('<div class="guidecard"><h4>Find</h4><p>Search any road, town, river, or county and see every bridge on the map.</p></div>',
                    unsafe_allow_html=True)
        st.button("Find a bridge", on_click=go_to, args=("Find a Bridge",))
    with g2.container(border=True):
        st.markdown('<div class="guidecard"><h4>Understand</h4><p>Each bridge gets a plain-English chance of needing major repair.</p></div>',
                    unsafe_allow_html=True)
        st.button("How it works", on_click=go_to, args=("How myBridge Works",))
    with g3.container(border=True):
        st.markdown('<div class="guidecard"><h4>Decide</h4><p>Fund the top of the list first, then send each one to an engineer.</p></div>',
                    unsafe_allow_html=True)
        st.button("What to fix first", on_click=go_to, args=("What to Fix First",))

    st.markdown("### What would you like to do?")
    t1, t2, t3 = st.columns(3, gap="large")
    t1.button("Plan repairs & budget", type="primary", width="stretch", on_click=go_to, args=("Home",))
    t2.button("Check a bridge", width="stretch", on_click=go_to, args=("Find a Bridge",))
    t3.button("Review bridges for approval", width="stretch", on_click=go_to, args=("Engineer Check",))

    if audience == "Owner":
        budget_section(open_card=True)
    else:
        with st.expander("Program budget", expanded=False):
            st.markdown(
                '<div class="caption">Owner view. Switch the audience toggle above for the full picture.</div>',
                unsafe_allow_html=True,
            )

# ================= FIND A BRIDGE =================
elif nav == "Find a Bridge":
    st.markdown("### Find a Bridge")
    with st.container(border=True):
        q = st.text_input("Search", placeholder="Search a road, town, river, or county")
        fc1, fc2 = st.columns(2, gap="large")
        counties = sorted(df["county_name"].dropna().unique())
        sel_counties = fc1.multiselect("County", counties, default=counties)
        p_min = fc2.slider("Minimum chance", 0, 100, 0, 1, format="%d%%") / 100
    f = apply_shortcut(df)
    f = f[f["county_name"].isin(sel_counties) & (f["P_POOR_5YR"] >= p_min)].copy()
    if q.strip():
        ql = q.strip().lower()
        f = f[
            f["facility_carried"].fillna("").str.lower().str.contains(ql)
            | f["features_intersected"].fillna("").str.lower().str.contains(ql)
            | f["county_name"].fillna("").str.lower().str.contains(ql)
        ]
    shortcut_note()
    if f.empty:
        st.warning("No bridges match. Try widening the search or clearing filters.")
    else:
        f = f.reset_index(drop=True)
        with st.container(border=True):
            st.markdown(
                '<div class="legend">'
                '<span class="lpill"><span class="dot" style="background:#D70015"></span>Needs major repair now</span>'
                '<span class="lpill"><span class="dot" style="background:#E85D04"></span>Very likely</span>'
                '<span class="lpill"><span class="dot" style="background:#E9A13B"></span>Likely</span>'
                '<span class="lpill"><span class="dot" style="background:#86868B"></span>Less likely</span>'
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
                    "Needs major repair now": "#D70015",
                    "Very likely": "#E85D04",
                    "Likely": "#E9A13B",
                    "Less likely": "#86868B",
                },
                category_orders={"band": ["Needs major repair now", "Very likely", "Likely", "Less likely"]},
                map_style="carto-positron",
                height=600,
                opacity=0.75,
                size_max=14,
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False, uirevision="map")
            sel = st.plotly_chart(fig, key="risk_map", on_select="rerun", width="stretch",
                                  config={"displayModeBar": False})
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
        pill = {"Needs major repair now": "pill-now", "Very likely": "pill-very",
                "Likely": "pill-likely"}.get(band, "pill-less")
        st.markdown(
            f'<div class="mcard"><div class="bigname">{bridge_name(row)}</div>'
            f'<div class="countyline">{clean(row["county_name"])} County</div>'
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
        if c1.button("Explain in plain English", type="primary", width="stretch"):
            st.session_state.explain = not st.session_state.explain

        if c2.button("Send for engineer check", width="stretch"):
            if sid not in st.session_state.review_queue:
                st.session_state.review_queue.append(sid)
            st.success("Sent for engineer check - see Engineer Check.")
        if st.session_state.explain:
            with st.container(border=True):
                st.markdown("**Why this bridge is flagged**")
                st.write(
                    f"This bridge has a {row['P_POOR_5YR']:.0%} chance of needing major repair "
                    f"in the next 5 years. Its overall condition today is "
                    f"**{cond_word(row['min_cond'])}**, it is {num(row['age_effective'])} years old, "
                    f"and it carries {num(row['adt'])} vehicles a day."
                )
                st.write(f"Suggested next step: **{clean(row['action_plain'])}** - act by {row['PROGRAM_YEAR']:.0f}.")
        with st.expander("Engineer technical details"):
            pairs = [("stable_id", sid)]
            for col in ["deck_cond", "super_cond", "sub_cond", "operating_tons",
                        "inventory_tons", "deck_area_sqft", "scour", "material",
                        "design", "owner", "min_cond", "year", "segment"]:
                if col in row.index:
                    v = row[col]
                    pairs.append((col, "" if pd.isna(v) else str(v)))
            st.dataframe(pd.DataFrame(pairs, columns=["Field", "Value"]),
                         hide_index=True, width="stretch")

# ================= WHAT TO FIX FIRST =================
elif nav == "What to Fix First":
    st.markdown("### What to Fix First")
    st.markdown(
        '<div class="caption">Highest chance first. Poor means needs major repair, not unsafe to cross.</div>',
        unsafe_allow_html=True,
    )
    shortcut_note()
    prog = apply_shortcut(df)
    if st.session_state.funded_only:
        prog = prog[prog["stable_id"].isin(st.session_state.funded_ids)]
        st.markdown(
            f'<div class="caption">Funded set only ({len(prog)} bridges).</div>',
            unsafe_allow_html=True,
        )
    prog = prog.sort_values("P_POOR_5YR", ascending=False).copy()
    prog["Overall condition today"] = prog["min_cond"].map(cond_word)
    prog["Chance in 5 years"] = (prog["P_POOR_5YR"] * 100).round(1)
    show = pd.DataFrame({
        LABELS["facility_carried"]: prog["facility_carried"],
        LABELS["county_name"]: prog["county_name"],
        "Overall condition today": prog["Overall condition today"],
        "Chance in 5 years": prog["Chance in 5 years"],
        LABELS["RECOMMENDED_ACTION"]: prog["action_plain"],
        LABELS["PROGRAM_YEAR"]: prog["PROGRAM_YEAR"],
    })
    st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        column_config={
            "Chance in 5 years": st.column_config.ProgressColumn(
                "Chance in 5 years", min_value=0.0, max_value=100.0, format="%d%%"
            ),
        },
    )

# ================= ENGINEER CHECK =================
elif nav == "Engineer Check":
    st.markdown("### Engineer Check")
    st.write(
        "myBridge suggests. You verify load capacity in your normal software, "
        "upload the result, check the numbers, then approve."
    )
    with st.container(border=True):
        st.markdown("**Waiting for engineer check**")
        if not st.session_state.review_queue:
            st.markdown(
                '<div class="caption">Queue is empty. Send bridges here from Find a Bridge.</div>',
                unsafe_allow_html=True,
            )
        for qsid in st.session_state.review_queue:
            qr = df[df["stable_id"] == qsid]
            if qr.empty:
                continue
            qr = qr.iloc[0]
            status = (
                '<span class="pill pill-green">Approved for funding plan</span>'
                if qsid in st.session_state.approved_list
                else '<span class="pill pill-gray">Waiting for check</span>'
            )
            st.markdown(
                f'<div class="queuerow"><span>{bridge_name(qr)} &mdash; '
                f"{clean(qr['county_name'])} ({qr['P_POOR_5YR']:.0%})</span>"
                f"<span>{status}</span></div>",
                unsafe_allow_html=True,
            )
    rsid = st.session_state.selected_bridge
    rrow = df[df["stable_id"] == rsid].iloc[0]
    with st.container(border=True):
        st.markdown(f"**Under review: {bridge_name(rrow)}**")
        st.markdown(
            f'<div class="caption">{clean(rrow["county_name"])} County &mdash; '
            f"{rrow['risk_band']}, {rrow['P_POOR_5YR']:.0%} chance in 5 years.</div>",
            unsafe_allow_html=True,
        )
        if rsid in st.session_state.approved_list:
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
            if rsid not in st.session_state.approved_list:
                st.session_state.approved_list.append(rsid)
            st.success(f"{bridge_name(rrow)} approved by {er_name.strip()} (license {er_license.strip()}).")
        if e2.button("Reject", key="er_reject"):
            st.info(f"{bridge_name(rrow)} sent back - flagged for follow-up.")
    if st.session_state.approved_list:
        with st.container(border=True):
            st.markdown(f"**Approved for funding plan ({len(st.session_state.approved_list)})**")
            adf = df[df["stable_id"].isin(st.session_state.approved_list)].copy()
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

# ================= HOW MYBRIDGE WORKS =================
elif nav == "How myBridge Works":
    st.markdown("### How myBridge Works")
    with st.container(border=True):
        st.markdown(
            "myBridge learned from 42 years of bridge inspections. It scores every bridge "
            "on its chance of needing major repair within 5 years, so owners can fund the "
            "highest chances first and engineers can verify each one before anything is decided."
        )
    mv = metrics.get("markov_valid_auc", 0)
    xv = metrics.get("xgb_valid_auc", 0)
    mt = metrics.get("markov_auc", 0)
    xt = metrics.get("xgb_auc", 0)
    with st.container(border=True):
        st.markdown("**Technical results**")
        tech_table = pd.DataFrame(
            {
                "Metric": ["Valid AUC", "Test AUC", "Valid PR-AUC", "Test PR-AUC",
                           "Valid Recall@0.3", "Test Recall@0.3"],
                "myBridge (XGBoost)": [f"{xv:.3f}", f"{xt:.3f}",
                                       f"{metrics.get('xgb_valid_prauc', 0):.3f}",
                                       f"{metrics.get('xgb_prauc', 0):.3f}",
                                       f"{metrics.get('xgb_valid_recall_0.3', 0):.3f}",
                                       f"{metrics.get('poor_recall_0.3', 0):.3f}"],
                "Markov baseline": [f"{mv:.3f}", f"{mt:.3f}", "-", "-", "-", "-"],
            }
        )
        st.table(tech_table)
        st.markdown(
            '<div class="caption">Temporal split only: trained 2000-2015, validated 2016-2018, '
            "tested 2019-2020. Recent years drift, so the held-out test "
            f"(XGBoost {xt:.3f} vs Markov {mt:.3f}) trails validation "
            f"(XGBoost {xv:.3f} vs Markov {mv:.3f}). Target: Poor within 5 years.</div>",
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer">myBridge &middot; Predictions from 42 years of inspections &middot; '
    "Llama explains, myBridge predicts, a licensed engineer decides.</div>",
    unsafe_allow_html=True,
)
