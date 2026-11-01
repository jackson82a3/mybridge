# app.py - myBridge v8 (frozen v3 model outputs, human language)
# Reads ONLY data/app_data.parquet.
# Plain language everywhere except the engineer technical expander.

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


def bridge_name(row):
    return f"{clean(row['facility_carried'])} over {clean(row['features_intersected'])}"


def clean_county(name):
    if pd.isna(name) or str(name).strip() == "":
        return ""
    s = " ".join(str(name).split())
    while "County County" in s:
        s = s.replace("County County", "County")
    return s


def condition_words(v):
    if pd.isna(v):
        return "Not rated"
    v = float(v)
    if v >= 7:
        return "In good shape"
    if v >= 5:
        return "Showing its age, but still doing its job"
    return "Worn and due for major work"


def status_info(row):
    p = row["P_POOR_5YR"]
    if row["IS_ALREADY_POOR"] == 1:
        return ("Needs major work now",
                "Already in poor condition and due for major repair. "
                "Poor does not mean unsafe to cross.")
    if p >= 0.7:
        return ("Plan soon",
                f"Very likely to need major work by {int(row['year']) + 5}. "
                f"About {round(p * 10)} in 10 bridges like this needed major work within 5 years.")
    if p >= 0.3:
        return ("Plan ahead",
                "May need major work within 5 years. "
                "Worth planning now before it becomes urgent.")
    return ("Doing well for now",
            "Most bridges like this stay in good or fair shape for the next 5 years.")


def _adt_text(row):
    if pd.isna(row["adt"]):
        return None
    return f"{int(round(row['adt'] / 100.0) * 100):,}"


def human_summary(row):
    county = clean_county(row["county_name"]) or "This"
    age = int(round(row["age_effective"])) if pd.notna(row["age_effective"]) else None
    adt = _adt_text(row)
    cond = condition_words(row["min_cond"])
    cond_l = cond[0].lower() + cond[1:]
    action = row["action_plain"][0].lower() + row["action_plain"][1:]
    year = int(row["PROGRAM_YEAR"])
    s1 = (f"This {county} bridge is {cond_l} after roughly {age} years on the road."
          if age is not None else f"This {county} bridge is {cond_l}.")
    title, _ = status_info(row)
    if title == "Needs major work now":
        s2 = "It already needs major work, though it remains safe to cross"
    elif title == "Plan soon":
        s2 = "It is very likely to need major work in the next few years"
    elif title == "Plan ahead":
        s2 = "It may need major work within five years, so it is worth planning early"
    else:
        s2 = "It should hold up well over the next five years"
    s2 += (f", and about {adt} vehicles use it every day."
           if adt else ", and its daily traffic is not on record.")
    s3 = f"The suggested next step is to {action} by {year}."
    return f"{s1} {s2} {s3}"


def context_line(row):
    county = clean_county(row["county_name"])
    if pd.isna(row["age_effective"]):
        return county + " · Age not on record"
    age = int(round(row["age_effective"]))
    if pd.isna(row["adt"]):
        return f"{county} · {age} years old"
    return f"{county} · {age} years old · About {int(row['adt']):,} vehicles a day"


df = load_bridges().copy()
df["action_plain"] = df["RECOMMENDED_ACTION"].map(ACTION_LABELS).fillna(df["RECOMMENDED_ACTION"])
df["status_title"] = df.apply(lambda r: status_info(r)[0], axis=1)
df["hover_text"] = df.apply(
    lambda r: f"<b>{bridge_name(r)}</b><br>{clean_county(r['county_name'])}<br><br>"
              f"<b>{status_info(r)[0]}</b><br>{status_info(r)[1]}",
    axis=1,
)

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


def open_bridge(sid):
    st.session_state.selected_bridge = sid
    st.session_state.page = "explore"


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
:root {
  --glass: linear-gradient(135deg, rgba(255,255,255,0.075), rgba(255,255,255,0.03));
  --glass-border: 1px solid rgba(255,255,255,0.09);
  --glass-shadow: 0 20px 50px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.10);
}
html, body, [class*="css"], .stApp { font-family:'Inter',-apple-system,'SF Pro Display',sans-serif; }
.stApp { background-color:#080B0F; background-image:
  radial-gradient(900px 480px at 50% 6%, rgba(212,175,55,0.13), transparent 70%),
  radial-gradient(800px 600px at 95% 100%, rgba(90,120,160,0.08), transparent 70%);
  background-attachment:fixed; color:#F5F5F7; }
[data-testid="stHeader"], footer, [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], #MainMenu { display:none !important; }
.block-container { max-width:1180px; padding-top:0 !important; padding-bottom:64px; }
.wordmark { text-align:center; font-weight:700; font-size:15px; letter-spacing:0.28em;
  text-transform:uppercase; color:#D4AF37; padding-top:40px; }
.hero { text-align:center; padding:8px 0 8px; }
.hero h1 { font-size:52px; font-weight:700; letter-spacing:-0.03em; margin:16px 0 32px;
  color:#F5F5F7; }
svg[role="img"] { filter:drop-shadow(0 0 18px rgba(212,175,55,0.35)); }
.topnav { position:sticky; top:0; z-index:999; display:flex; justify-content:space-between;
  align-items:center; padding:14px 4px; background:rgba(8,11,15,0.55);
  backdrop-filter:blur(20px) saturate(140%); -webkit-backdrop-filter:blur(20px) saturate(140%);
  border-bottom:1px solid rgba(255,255,255,0.09); }
.topnav .brand { font-weight:700; font-size:18px; }
.topbar { position:sticky; top:0; z-index:900; display:flex; justify-content:space-between;
  align-items:center; padding:18px 4px; background:rgba(8,11,15,0.55);
  backdrop-filter:blur(20px) saturate(140%); -webkit-backdrop-filter:blur(20px) saturate(140%);
  border-bottom:1px solid rgba(255,255,255,0.09); }
.topbar .brand { font-weight:700; font-size:18px; }
.mcard { background:var(--glass); border:var(--glass-border); border-radius:22px;
  box-shadow:var(--glass-shadow); backdrop-filter:blur(20px) saturate(140%);
  -webkit-backdrop-filter:blur(20px) saturate(140%); padding:28px 32px; margin:16px 0; }
.bridge-strong { background:linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.045)) !important;
  box-shadow:0 20px 50px rgba(0,0,0,0.35), inset 0 1px 0 rgba(212,175,55,0.25) !important; }
div[data-testid="stVerticalBlockBorderWrapper"] > div { background:var(--glass) !important;
  border:var(--glass-border) !important; border-radius:22px !important;
  box-shadow:var(--glass-shadow) !important; backdrop-filter:blur(20px) saturate(140%) !important;
  -webkit-backdrop-filter:blur(20px) saturate(140%) !important; padding:24px !important; }
[data-testid="stExpander"] { background:var(--glass); border:var(--glass-border);
  border-radius:22px; box-shadow:var(--glass-shadow); backdrop-filter:blur(20px) saturate(140%);
  -webkit-backdrop-filter:blur(20px) saturate(140%); }
div[data-testid="stButton"] button { transition:transform .15s ease, box-shadow .15s ease, filter .15s ease, border-color .15s ease; }
div[data-testid="stButton"] button[kind="primary"] {
  background:linear-gradient(180deg,#E9CB6B,#C9A227) !important; color:#0C0F14 !important;
  border:none !important; border-radius:980px !important; padding:12px 32px !important;
  font-weight:700 !important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,0.45), 0 10px 30px rgba(212,175,55,0.22) !important; }
div[data-testid="stButton"] button[kind="primary"]:hover { transform:translateY(-1px);
  box-shadow:inset 0 1px 0 rgba(255,255,255,0.55), 0 14px 34px rgba(212,175,55,0.34) !important; }
div[data-testid="stButton"] button[kind="secondary"] { background:rgba(255,255,255,0.07) !important;
  backdrop-filter:blur(20px) saturate(140%); -webkit-backdrop-filter:blur(20px) saturate(140%);
  color:#fff !important; border:1px solid rgba(255,255,255,0.14) !important;
  border-radius:980px !important; padding:12px 32px !important; font-weight:600 !important; }
div[data-testid="stButton"] button[kind="secondary"]:hover { transform:translateY(-1px);
  border-color:rgba(255,255,255,0.25) !important;
  box-shadow:0 8px 24px rgba(0,0,0,0.35), 0 0 18px rgba(212,175,55,0.12) !important; }
div[data-testid="stButton"] button[kind="tertiary"] { background:transparent !important;
  color:#D4AF37 !important; border:none !important; font-weight:600 !important;
  box-shadow:none !important; padding:8px 12px !important; }
div[data-testid="stButton"] button[kind="tertiary"]:hover { transform:translateY(-1px); }
div[data-testid="stButton"] button:disabled { background:#2A3140 !important;
  color:#6B7280 !important; border:none !important; transform:none !important; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
  background:rgba(255,255,255,0.05) !important;
  border:1px solid rgba(255,255,255,0.10) !important;
  border-radius:14px !important; color:#F5F5F7 !important; }
[data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus {
  border-color:rgba(212,175,55,0.55) !important;
  box-shadow:0 0 0 1px rgba(212,175,55,0.55) !important; outline:none !important; }
[data-testid="stWidgetLabel"] p { font-size:13px !important; color:#9A9AA0 !important; }
section[data-testid="stFileUploaderDropzone"] {
  border:1px dashed rgba(255,255,255,0.25) !important;
  background:rgba(255,255,255,0.04) !important; border-radius:18px !important; }
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
  background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.09);
  border-radius:18px; overflow:hidden; }
.pill { display:inline-block; font-size:12px; font-weight:600; padding:5px 14px;
  border-radius:980px; }
.pill-now { background:rgba(215,0,21,0.14); color:#FF8585; border:1px solid rgba(215,0,21,0.25); }
.pill-very { background:rgba(232,93,4,0.14); color:#F5A524; border:1px solid rgba(232,93,4,0.25); }
.pill-likely { background:rgba(212,175,55,0.14); color:#E3C25C; border:1px solid rgba(212,175,55,0.25); }
.pill-less { background:rgba(255,255,255,0.08); color:#9A9AA0; border:1px solid rgba(255,255,255,0.25); }
.pill-green { background:rgba(26,158,84,0.16); color:#4ADE80; border:1px solid rgba(26,158,84,0.30); }
.pill-gray { background:rgba(255,255,255,0.08); color:#F5F5F7; border:1px solid rgba(255,255,255,0.12); }
.legend { display:flex; gap:8px; margin:0 0 12px; flex-wrap:wrap; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }
.lpill { background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.09);
  backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
  font-size:12px; font-weight:600; padding:5px 14px; border-radius:980px; color:#F5F5F7; }
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
.flowpill { background:var(--glass); border:var(--glass-border); border-radius:980px;
  backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
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
    b1, b2, b3 = st.columns(3, gap="large")
    b1.button("Explore Bridges", type="primary", width="stretch", on_click=go_to, args=("explore",))
    b2.button("Planning & Budget", width="stretch", on_click=go_to, args=("plan",))
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
                '<span class="lpill"><span class="dot" style="background:#E5484D"></span>Needs major work now</span>'
                '<span class="lpill"><span class="dot" style="background:#E8A33D"></span>Plan soon</span>'
                '<span class="lpill"><span class="dot" style="background:#D4AF37"></span>Plan ahead</span>'
                '<span class="lpill"><span class="dot" style="background:#6E7681"></span>Doing well for now</span>'
                "</div>",
                unsafe_allow_html=True,
            )
            mapdf = pd.DataFrame({
                "lat": f["lat"],
                "long": f["long"],
                "band": f["status_title"],
                "size": f["P_POOR_5YR"] * 18 + 3,
                "sid": f["stable_id"],
            })
            fig = px.scatter_map(
                mapdf,
                lat="lat",
                lon="long",
                color="band",
                size="size",
                custom_data=["sid"],
                color_discrete_map={
                    "Needs major work now": "#E5484D",
                    "Plan soon": "#E8A33D",
                    "Plan ahead": "#D4AF37",
                    "Doing well for now": "#6E7681",
                },
                category_orders={"band": ["Needs major work now", "Plan soon", "Plan ahead", "Doing well for now"]},
                map_style="carto-darkmatter",
                height=500,
                opacity=0.7,
                size_max=14,
            )
            hover_by_sid = dict(zip(f["stable_id"], f["hover_text"]))
            for tr in fig.data:
                if tr.customdata is not None:
                    tr.hovertext = [hover_by_sid.get(s, "") for s in tr.customdata[:, 0]]
            fig.update_traces(hovertemplate="%{hovertext}<extra></extra>")
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                uirevision="map",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F5F5F7"),
                hoverlabel=dict(
                    bgcolor="rgba(16,20,27,0.88)",
                    bordercolor="rgba(255,255,255,0.12)",
                    font=dict(color="white"),
                ),
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
                f"{bridge_name(tr)} - {tr['status_title']}",
                key=f"top_{tr['stable_id']}",
                width="stretch",
            ):
                st.session_state.selected_bridge = tr["stable_id"]
            st.caption(context_line(tr))

        sid, row = selected_row()
        title, outlook = status_info(row)
        pill = {"Needs major work now": "pill-now", "Plan soon": "pill-very",
                "Plan ahead": "pill-likely"}.get(title, "pill-less")
        st.markdown(
            f'<div class="mcard bridge-strong"><div class="bigname">{bridge_name(row)}</div>'
            f'<div class="countyline">{context_line(row)}</div>'
            f'<div style="margin:6px 0 12px"><span class="pill {pill}">{title}</span></div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2, gap="large")
        if c1.button("Overview", type="primary", width="stretch"):
            st.session_state.show_summary = not st.session_state.show_summary
        c2.button("Send for Engineer Check ->", type="tertiary", width="stretch",
                  on_click=send_for_check, args=(sid,))
        if st.session_state.show_summary:
            with st.container(border=True):
                st.write(human_summary(row))
                st.write(outlook)
                st.write(f"**Next step:** {clean(row['action_plain'])} · Act by {int(row['PROGRAM_YEAR'])}")
                st.markdown(
                    '<div class="caption">Poor means needs major repair, not unsafe to cross.</div>',
                    unsafe_allow_html=True,
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

# ================= PLANNING & BUDGET =================
elif page == "plan":
    st.button("< Home", type="tertiary", on_click=go_to, args=("home",))
    st.button("< Explore Bridges", type="tertiary", on_click=go_to, args=("explore",))
    st.markdown("## Planning & Budget")
    with st.container(border=True):
        budget = st.slider("Annual budget", 10, 100, 50, 5, format="$%dM", key="plan_budget")
        cand = df[(df["IS_ALREADY_POOR"] == 1) | (df["P_POOR_5YR"] > 0.3)].copy()
        med_area = cand["deck_area_sqft"].median()
        rates = {"Replacement - Engineer Review": 700, "Rehabilitation - Engineer Review": 350}
        cand["est_cost"] = cand["deck_area_sqft"].fillna(med_area) * cand["RECOMMENDED_ACTION"].map(rates).fillna(75)
        cand = cand.sort_values("P_POOR_5YR", ascending=False)
        funded = cand[cand["est_cost"].cumsum() <= budget * 5 * 1_000_000]
        st.write(
            f"With ${budget}M a year for the next 5 years, we could start with the "
            f"{len(funded)} bridges that need it most."
        )
        st.write("The rest stay on watch and move up as work is finished.")
    for _, fr in funded.head(10).iterrows():
        st.button(
            f"{bridge_name(fr)} - {status_info(fr)[0]} · {clean_county(fr['county_name'])}",
            key=f"fund_{fr['stable_id']}",
            width="stretch",
            on_click=open_bridge,
            args=(fr["stable_id"],),
        )
    st.button("See these bridges ->", type="primary", on_click=go_to, args=("explore",))
    st.caption("Early planning estimates only. An engineer confirms each bridge before money is committed.")

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
                f"{bridge_name(qr)} - {clean_county(qr['county_name'])} - {qr['P_POOR_5YR']:.0%}"
                + ("  [Approved]" if qsid in st.session_state.approved_list else ""),
                key=f"q_{qsid}",
                width="stretch",
            ):
                st.session_state.selected_bridge = qsid

    sid, rrow = selected_row()
    with st.container(border=True):
        st.markdown(f"**Under review: {bridge_name(rrow)}**")
        st.markdown(
            f'<div class="caption">{clean_county(rrow["county_name"])} &mdash; '
            f"{rrow['status_title']}, {rrow['P_POOR_5YR']:.0%} chance in 5 years.</div>",
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
                    f'<div class="caption">{bridge_name(ar)} &mdash; {clean_county(ar["county_name"])}</div>',
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
