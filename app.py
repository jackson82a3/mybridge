# app.py - myBridge v8 (frozen v3 model outputs, human language)
# Reads ONLY data/app_data.parquet.
# Plain language everywhere except the engineer technical expander.

import json
import os
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from agent_tools import answer_question
import finance_model as fm

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
def load_data():
    df = pd.read_parquet("data/app_data.parquet")
    df["action_plain"] = df["RECOMMENDED_ACTION"].map(ACTION_LABELS).fillna(df["RECOMMENDED_ACTION"])
    poor = df["IS_ALREADY_POOR"] == 1
    p = df["P_POOR_5YR"]
    df["status_title"] = np.select(
        [poor, p >= 0.7, p >= 0.3],
        ["Needs major work now", "Plan soon", "Plan ahead"],
        default="Doing well for now",
    )
    horizon_year = int(df["year"].max()) + 5 if "year" in df.columns else 2030
    soon = ("Very likely to need major work by " + str(horizon_year)
            + ". About " + (p * 10).round().astype(int).astype(str)
            + " in 10 bridges like this needed major work within 5 years.")
    df["status_sentence"] = np.select(
        [poor, p >= 0.7, p >= 0.3],
        ["Already in poor condition and due for major repair. Poor does not mean unsafe to cross.",
         soon,
         "May need major work within 5 years. Worth planning now before it becomes urgent."],
        default="Most bridges like this stay in good or fair shape for the next 5 years.",
    )
    county = df["county_name"].fillna("").astype(str).str.split().str.join(" ")
    for _ in range(3):
        county = county.str.replace("County County", "County", regex=False)
    df["county_clean"] = county
    disp_name = (df["facility_carried"].fillna("").astype(str).str.strip() + " over "
                 + df["features_intersected"].fillna("").astype(str).str.strip().str.title())
    age_ok = df["age_effective"].notna()
    age_s = df["age_effective"].round().astype("Int64").astype(str)
    adt_ok = df["adt"].notna()
    adt_s = df["adt"].map(lambda v: f"{int(v):,}" if pd.notna(v) else "")
    ctx = county + np.where(age_ok, " · " + age_s + " years old", " · Age not on record")
    df["context"] = ctx + np.where(adt_ok, " · About " + adt_s + " vehicles a day", "")
    wrapped = df["status_sentence"].astype(str).map(
        lambda s: "<br>".join(textwrap.wrap(s, width=40)))
    df["hover_text"] = ("<b>" + disp_name + "</b><br>" + county + "<br><br><b>"
                        + df["status_title"] + "</b><br>" + wrapped)
    return df


def clean(v):
    return "" if pd.isna(v) else str(v)


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
    year = max(int(row["PROGRAM_YEAR"]), 2027)
    s1 = (f"This {county} bridge is {cond_l} after roughly {age} years on the road."
          if age is not None else f"This {county} bridge is {cond_l}.")
    title = row["status_title"]
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


# ---------------- Llama overview (optional) ----------------
def _secret(name, default=None):
    try:
        v = st.secrets[name]
    except Exception:
        v = None
    if v in (None, ""):
        v = os.environ.get(name)
    return v if v not in (None, "") else default


LLAMA_API_KEY = _secret("LLAMA_API_KEY")
LLAMA_BASE_URL = _secret("LLAMA_BASE_URL", "https://api.llama.com/v1")
LLAMA_MODEL = _secret("LLAMA_MODEL", "Llama-4-Scout-17B-16E-Instruct-FP8")


@st.cache_resource
def _llama_client():
    from openai import OpenAI

    return OpenAI(api_key=LLAMA_API_KEY, base_url=LLAMA_BASE_URL)


def _log_status():
    print(f"[myBridge] Llama status={st.session_state.get('llama_status')}", flush=True)
    print(f"[myBridge] Llama model={LLAMA_MODEL}", flush=True)


def _set_status(sid, code):
    st.session_state["llama_status"] = code
    st.session_state.setdefault("overview_status", {})[sid] = code
    _log_status()


def _skip_reason():
    if not LLAMA_API_KEY:
        return "SKIPPED_MISSING_KEY"
    if LLAMA_API_KEY.strip() == "PASTE_NEW_KEY_HERE":
        return "SKIPPED_PLACEHOLDER_KEY"
    if "groq" in (LLAMA_BASE_URL or "").lower() and not LLAMA_API_KEY.startswith("gsk_"):
        return "SKIPPED_WRONG_KEY_PREFIX"
    return None


def generate_overview(row, status_title, status_sentence):
    cache = st.session_state.setdefault("overview_cache", {})
    sid = row["stable_id"]
    if sid in cache:
        return cache[sid]
    skip = _skip_reason()
    if skip:
        _set_status(sid, skip)
        return human_summary(row)  # fallback is never cached; next click retries
    _set_status(sid, "CALL_STARTED")
    try:
        payload = {
            "road": clean(row["facility_carried"]),
            "crosses": clean(row["features_intersected"]),
            "county": clean_county(row["county_name"]),
            "age_years": int(round(row["age_effective"])) if pd.notna(row["age_effective"]) else None,
            "vehicles_per_day": int(round(row["adt"] / 100.0) * 100) if pd.notna(row["adt"]) else None,
            "condition": condition_words(row["min_cond"]),
            "status": status_title,
            "chance_percent": int(round(row["P_POOR_5YR"] * 100)),
            "next_step": clean(row["action_plain"]),
            "act_by_year": max(int(row["PROGRAM_YEAR"]), 2027),
        }
        resp = _llama_client().chat.completions.create(
            model=LLAMA_MODEL,
            temperature=0.25,
            max_tokens=220,
            messages=[
                {"role": "system",
                 "content": "You are myBridge. Write a short overview for a non-engineer. "
                            "Use ONLY the data provided. Do not invent numbers, costs, causes, "
                            "or dates. Poor means needs major repair, not unsafe to cross."},
                {"role": "user",
                 "content": "Here is one bridge as JSON:\n" + json.dumps(payload)
                            + "\n\nWrite 2-3 natural sentences about this bridge for a "
                              "non-engineer, then 1 final sentence with the recommended "
                              "next step and the act-by year. Plain language only."},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("empty response")
        _set_status(sid, "CALL_SUCCESS")
        cache[sid] = text
        return text
    except Exception as e:
        _set_status(sid, f"CALL_FAILED_{type(e).__name__}")
        detail = getattr(e, "name", None)
        if detail:
            print(f"[myBridge] Llama missing module={detail}", flush=True)
        return human_summary(row)  # fallback is never cached; next click retries


df = load_data()

BASE_YEAR = int(df["year"].max()) if "year" in df.columns else 2025
HORIZON_YEAR = BASE_YEAR + 5
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
if "list_limit" not in st.session_state:
    st.session_state.list_limit = 10
if "list_sig" not in st.session_state:
    st.session_state.list_sig = None
if "ask_answer" not in st.session_state:
    st.session_state.ask_answer = None
if "ask_pending" not in st.session_state:
    st.session_state.ask_pending = False


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


def show_more():
    st.session_state.list_limit += 10


def _ask_submit():
    st.session_state.ask_pending = True


def _ask_chip(text):
    st.session_state.ask_mybridge_input = text
    st.session_state.ask_pending = True


def _ask_clear():
    st.session_state.ask_answer = None
    st.session_state.ask_pending = False
    st.session_state.ask_mybridge_input = ""


def _ask_view(sid):
    st.session_state.bridge_search = ""
    st.session_state.risk_filter = "all"
    st.session_state.selected_bridge = sid
    st.session_state.show_summary = False
    st.session_state.page = "explore"


def _fmt_money(v):
    v = float(v)
    return f"${v / 1_000_000:,.1f}M" if abs(v) >= 1_000_000 else f"${v:,.0f}"


def _focus_plan(sid):
    st.session_state.selected_bridge = sid
    st.session_state.focus_bridge = sid
    st.session_state.page = "plan"


def _send_shortlist(ids):
    for sid in ids:
        if sid not in st.session_state.review_queue:
            st.session_state.review_queue.append(sid)
    st.session_state.page = "engineer"


def _ask_open_plan(budget_m, horizon, strategy):
    st.session_state.plan_budget = int(min(100, max(10, round(budget_m / 5) * 5)))
    st.session_state.plan_horizon = int(horizon)
    st.session_state.plan_strategy = strategy
    st.session_state.page = "plan"


def _run_ask():
    q = st.session_state.get("ask_mybridge_input", "").strip()
    if not q:
        return
    factory = _llama_client if _skip_reason() is None else None
    with st.spinner("Thinking..."):
        st.session_state.ask_answer = answer_question(
            q, df, client_factory=factory, model=LLAMA_MODEL)


def render_nav(items):
    # items = list of (label, target_page, key_name)
    cols = st.columns([1.3, 2.0, 4.7, 2.0], vertical_alignment="center")
    for col, (label, target, key_name) in zip(cols, items):
        with col:
            if st.button(label, key=key_name, use_container_width=False):
                st.session_state.page = target
                st.rerun()
    with cols[-1]:
        st.markdown('<div class="brand-mark">MYBRIDGE</div>', unsafe_allow_html=True)


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
h1 > a, h2 > a, h3 > a { display:none !important; }
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
.caption { font-size:13px; color:#9A9AA0; }
.footer { text-align:center; font-size:13px; color:#9A9AA0; margin-top:64px; }
.stepnum { font-size:40px; font-weight:800; color:#D4AF37; line-height:1; }
.steptitle { font-size:17px; font-weight:700; margin:8px 0 4px; }
.stepline { font-size:14px; color:#9A9AA0; }
.arrow { text-align:center; font-size:24px; color:#4A5568; padding-top:40px; }
.flowpill { background:var(--glass); border:var(--glass-border); border-radius:980px;
  backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
  padding:10px 18px; font-size:14px; font-weight:600; text-align:center; }
.st-key-nav_card_explore, .st-key-nav_card_plan, .st-key-nav_card_engineer {
  background:linear-gradient(135deg, rgba(255,255,255,0.075), rgba(255,255,255,0.03));
  border:1px solid rgba(255,255,255,0.09); border-radius:20px; padding:18px;
  min-height:112px; height:100%;
  transition:border-color 0.25s ease, transform 0.25s ease; }
.st-key-nav_card_explore:hover, .st-key-nav_card_plan:hover, .st-key-nav_card_engineer:hover {
  border-color:rgba(212,175,55,0.40); transform:translateY(-2px); }
@media (max-width: 700px) {
  .st-key-home_nav [data-testid="stHorizontalBlock"] { flex-direction:column !important; }
  .st-key-home_nav [data-testid="stColumn"] { width:100% !important; }
}
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

st.markdown(
    """
<style>
[data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent !important; box-shadow: none !important; }
.block-container, [data-testid="stMain"], [data-testid="stAppViewContainer"] { background: transparent !important; }

.brand-mark { text-align: right; color: #D4AF37; letter-spacing: 0.32em; font-weight: 700; font-size: 14px; padding-top: 6px; background: transparent; }

.st-key-nav_home button, .st-key-nav_explore button {
  width: auto !important; min-width: 0 !important; height: auto !important; min-height: 0 !important;
  padding: 7px 16px !important; border-radius: 980px !important;
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  color: #D8D8DC !important; font-size: 14px !important; font-weight: 400 !important;
  box-shadow: none !important; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
}
.st-key-nav_home button:hover, .st-key-nav_explore button:hover {
  background: rgba(212,175,55,0.10) !important; border-color: rgba(212,175,55,0.35) !important; color: #FFFFFF !important;
}

.st-key-bridge_search input { background: rgba(255,255,255,0.05) !important; color: #F5F5F7 !important; -webkit-text-fill-color: #F5F5F7 !important; caret-color: #D4AF37; border: 1px solid rgba(255,255,255,0.10) !important; border-radius: 14px !important; }
.st-key-bridge_search input::placeholder { color: #8E8E93 !important; opacity: 1 !important; }
.st-key-bridge_search input:focus { border-color: rgba(212,175,55,0.55) !important; box-shadow: 0 0 0 1px rgba(212,175,55,0.22) !important; outline: none !important; }
.st-key-bridge_search label p { color: #9A9AA0 !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
/* GLOBAL INPUT FIX - must stay last */
[data-testid="stTextInput"], [data-testid="stNumberInput"], [data-testid="stTextArea"],
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] {
  background: rgba(255,255,255,0.05) !important; background-color: rgba(255,255,255,0.05) !important;
  color: #F5F5F7 !important; -webkit-text-fill-color: #F5F5F7 !important; caret-color: #D4AF37 !important;
  border: 1px solid rgba(255,255,255,0.10) !important; border-radius: 14px !important; box-shadow: none !important; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stTextArea"] textarea,
[data-baseweb="input"] input, [data-baseweb="base-input"] input, [data-baseweb="select"] input {
  background: transparent !important; background-color: transparent !important;
  color: #F5F5F7 !important; -webkit-text-fill-color: #F5F5F7 !important; caret-color: #D4AF37 !important;
  border: none !important; box-shadow: none !important; outline: none !important; }
[data-testid="stTextInput"] input::placeholder, [data-testid="stNumberInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
  color: #8E8E93 !important; -webkit-text-fill-color: #8E8E93 !important; opacity: 1 !important; }
[data-testid="stTextInput"]:focus-within, [data-testid="stNumberInput"]:focus-within,
[data-testid="stTextArea"]:focus-within, [data-baseweb="input"]:focus-within,
[data-baseweb="base-input"]:focus-within, [data-baseweb="select"]:focus-within {
  border-color: rgba(212,175,55,0.55) !important; box-shadow: 0 0 0 1px rgba(212,175,55,0.25) !important; outline: none !important; }
[data-testid="stWidgetLabel"] p { color: #9A9AA0 !important; }
[data-testid="InputInstructions"] { display: none !important; }
[data-testid="stNumberInput"] button, [data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {
  background: rgba(255,255,255,0.08) !important; color: #F5F5F7 !important; border: 1px solid rgba(255,255,255,0.10) !important; }
[data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,0.03) !important; border: 1px dashed rgba(255,255,255,0.14) !important; color: #C7C7CC !important; }
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span { color: #C7C7CC !important; }
[data-testid="stFileUploaderDropzone"] button {
  background: rgba(255,255,255,0.08) !important; color: #FFFFFF !important; border: 1px solid rgba(255,255,255,0.14) !important; }

.st-key-ask_mybridge { max-width: 760px; margin: 30px auto 0 !important; }
.st-key-ask_mybridge div[data-testid="stButton"] button[kind="primary"] { height: 44px !important; }
h1 a, [data-testid="StyledLinkIconContainer"] { display: none !important; }
.st-key-nav_card_explore [data-testid="stButton"] button,
.st-key-nav_card_plan [data-testid="stButton"] button,
.st-key-nav_card_engineer [data-testid="stButton"] button {
  background: transparent !important; border: none !important; color: #D4AF37 !important;
  padding: 4px 0 !important; font-size: 14px !important; font-weight: 600 !important;
  box-shadow: none !important; height: auto !important; min-height: 0 !important; }
.st-key-nav_card_explore [data-testid="stButton"] button:hover,
.st-key-nav_card_plan [data-testid="stButton"] button:hover,
.st-key-nav_card_engineer [data-testid="stButton"] button:hover { color: #FFFFFF !important; }
.st-key-prompt_chips [data-testid="stButton"] button {
  height: auto !important; min-height: 30px !important; padding: 5px 12px !important;
  font-size: 12.5px !important; font-weight: 400 !important; color: #A1A1AA !important;
  background: transparent !important; border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 980px !important; box-shadow: none !important;
  white-space: normal !important; line-height: 1.3 !important; }
.st-key-prompt_chips [data-testid="stButton"] button:hover {
  color: #FFFFFF !important; border-color: rgba(212,175,55,0.35) !important;
  background: rgba(212,175,55,0.08) !important; }
.st-key-next_step [data-testid="stButton"] button {
  width: 100% !important; height: 52px !important; background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(212,175,55,0.30) !important; border-radius: 16px !important;
  color: #D4AF37 !important; font-size: 15px !important; font-weight: 600 !important;
  box-shadow: none !important; }
.st-key-next_step [data-testid="stButton"] button:hover {
  background: rgba(212,175,55,0.10) !important; border-color: rgba(212,175,55,0.45) !important;
  color: #FFFFFF !important; }
</style>
""",
    unsafe_allow_html=True,
)

components.html(
    "<script>"
    "const _fixInputs=()=>{window.parent.document.querySelectorAll('input, textarea')"
    ".forEach(el=>{el.setAttribute('spellcheck','false');el.setAttribute('autocomplete','off');"
    "el.setAttribute('autocorrect','off');el.setAttribute('autocapitalize','off');});};"
    "_fixInputs();setTimeout(_fixInputs,300);setTimeout(_fixInputs,1000);"
    "</script>",
    height=0,
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
    with st.container(key="home_nav"):
        n1, n2, n3 = st.columns(3, gap="large")
        with n1:
            with st.container(key="nav_card_explore"):
                st.markdown(
                    '<div style="color:#D4AF37;font-size:18px;">⌖</div>'
                    '<div style="font-size:17px;font-weight:600;margin:6px 0 4px;">Explore Bridges</div>'
                    '<div style="font-size:13px;color:#9A9AA0;line-height:1.45;">Search and understand any bridge.</div>',
                    unsafe_allow_html=True,
                )
                st.button("Open ->", key="nav_open_explore", on_click=go_to, args=("explore",))
        with n2:
            with st.container(key="nav_card_plan"):
                st.markdown(
                    '<div style="color:#D4AF37;font-size:18px;">◈</div>'
                    '<div style="font-size:17px;font-weight:600;margin:6px 0 4px;">Planning &amp; Budget</div>'
                    '<div style="font-size:13px;color:#9A9AA0;line-height:1.45;">Build a funded plan with cost ranges.</div>',
                    unsafe_allow_html=True,
                )
                st.button("Open ->", key="nav_open_plan", on_click=go_to, args=("plan",))
        with n3:
            with st.container(key="nav_card_engineer"):
                st.markdown(
                    '<div style="color:#D4AF37;font-size:18px;">✓</div>'
                    '<div style="font-size:17px;font-weight:600;margin:6px 0 4px;">Engineer Check</div>'
                    '<div style="font-size:13px;color:#9A9AA0;line-height:1.45;">Verify load ratings and approve.</div>',
                    unsafe_allow_html=True,
                )
                st.button("Open ->", key="nav_open_engineer", on_click=go_to, args=("engineer",))
    with st.container(key="ask_mybridge"):
        st.markdown(
            '<div style="color:#D4AF37;font-size:13px;font-weight:700;'
            'letter-spacing:0.18em;margin:6px 0 10px;">Ask myBridge</div>',
            unsafe_allow_html=True,
        )
        a_in, a_btn = st.columns([4, 1], gap="medium")
        a_in.text_input(
            "Ask myBridge",
            placeholder="Ask about a bridge, place, budget, or repair cost…",
            key="ask_mybridge_input",
            label_visibility="collapsed",
        )
        a_btn.button("Ask", type="primary", width="stretch", on_click=_ask_submit)
        with st.container(key="prompt_chips"):
            st.markdown(
                '<div style="font-size:12px;color:#86868B;margin:2px 0 8px;">Try asking:</div>',
                unsafe_allow_html=True,
            )
            chips = [
                "Bridges needing work in Chisago County",
                "Busy bridges over a river",
                "Plan with $40M a year",
            ]
            chip_cols = st.columns([1.5, 1.05, 0.95], gap="small")
            for i, (col, chip) in enumerate(zip(chip_cols, chips)):
                col.button("✦ " + chip, key=f"ask_chip_{i}", width="stretch",
                           on_click=_ask_chip, args=(chip,))
    if st.session_state.ask_pending:
        st.session_state.ask_pending = False
        _run_ask()
    ans = st.session_state.ask_answer
    if ans:
        with st.container(border=True):
            st.markdown(f"**I understood:** {ans['understood']}")
            st.write(ans["answer"])
            open_label = "Open bridge" if ans["kind"] == "estimate_bridge_cost" else "View on map"
            for b in ans["bridges"][:5]:
                r1, r2, r3 = st.columns([3, 1, 1], gap="small")
                veh = (f" · About {b['vehicles_per_day']:,} vehicles a day"
                       if b.get("vehicles_per_day") else "")
                yr = f" · Target {b['target_year']}" if b.get("target_year") else ""
                r1.markdown(f"**{b['name']}**  \n{b['county']} · {b['status']}{veh}{yr}")
                r2.button(open_label, key=f"ask_view_{b['stable_id']}",
                          on_click=_ask_view, args=(b["stable_id"],))
                r3.button("Send for Engineer Check", key=f"ask_send_{b['stable_id']}",
                          on_click=send_for_check, args=(b["stable_id"],))
            if ans.get("plan"):
                pl = ans["plan"]
                st.button("Open this plan", type="primary", on_click=_ask_open_plan,
                          args=(pl["annual_budget"] / 1_000_000, pl["horizon_years"], pl["strategy"]))
            st.button("Clear", on_click=_ask_clear)
    g1, g2, g3 = st.columns([1, 1, 1], gap="large")
    g2.button("How to use myBridge ->", type="tertiary", width="stretch", on_click=go_to, args=("guide",))

# ================= EXPLORE =================
elif page == "explore":
    render_nav([("← Home", "home", "nav_home")])

    q = st.text_input("Search", placeholder="Search a road, river, town, or county",
                      key="bridge_search")
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
            band_style = {
                "Needs major work now": ("#E5484D", 9),
                "Plan soon": ("#E8A33D", 7),
                "Plan ahead": ("#D4AF37", 6),
                "Doing well for now": ("#6E7681", 5),
            }
            fig = go.Figure()
            for band_name in ["Needs major work now", "Plan soon", "Plan ahead", "Doing well for now"]:
                sub = f[f["status_title"] == band_name]
                color, msize = band_style[band_name]
                fig.add_trace(go.Scattermap(
                    lat=sub["lat"],
                    lon=sub["long"],
                    mode="markers",
                    marker=dict(size=msize, color=color, opacity=0.85),
                    customdata=sub["stable_id"],
                    hovertext=sub["hover_text"],
                    hovertemplate="%{hovertext}<extra></extra>",
                    name=band_name,
                    showlegend=False,
                ))
            fig.update_layout(
                map=dict(
                    style="carto-darkmatter",
                    center=dict(lat=float(f["lat"].mean()), lon=float(f["long"].mean())),
                    zoom=6,
                ),
                height=500,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F5F5F7"),
                hoverlabel=dict(
                    bgcolor="rgba(14,17,23,0.94)",
                    bordercolor="rgba(255,255,255,0.12)",
                    font=dict(family="Inter", size=13, color="#F5F5F7"),
                    align="left",
                ),
                uirevision="map",
            )
            sel = st.plotly_chart(fig, key="risk_map", on_select="rerun", width="stretch",
                                  config={"displayModeBar": False})
            if sel and sel.selection and sel.selection.get("points"):
                cd = sel.selection["points"][0].get("customdata")
                if cd:
                    st.session_state.selected_bridge = cd[0] if isinstance(cd, (list, tuple)) else cd

        sig = (q.strip().lower(), rf)
        if st.session_state.list_sig != sig:
            st.session_state.list_sig = sig
            st.session_state.list_limit = 10
        ranked = f.sort_values("P_POOR_5YR", ascending=False)
        st.markdown("**Highest chance first**")
        for _, tr in ranked.head(st.session_state.list_limit).iterrows():
            if st.button(
                f"{bridge_name(tr)} - {tr['status_title']}",
                key=f"top_{tr['stable_id']}",
                width="stretch",
            ):
                st.session_state.selected_bridge = tr["stable_id"]
            st.caption(tr["context"])
        if len(ranked) > st.session_state.list_limit:
            st.button("Show more", on_click=show_more)

        sid, row = selected_row()
        title, outlook = row["status_title"], row["status_sentence"]
        pill = {"Needs major work now": "pill-now", "Plan soon": "pill-very",
                "Plan ahead": "pill-likely"}.get(title, "pill-less")
        st.markdown(
            f'<div class="mcard bridge-strong"><div class="bigname">{bridge_name(row)}</div>'
            f'<div class="countyline">{row["context"]}</div>'
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
                _cache = st.session_state.setdefault("overview_cache", {})
                if sid in _cache:
                    st.write(_cache[sid])
                else:
                    with st.spinner("Preparing overview..."):
                        st.write(generate_overview(row, title, outlook))
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
        with st.container(key="next_step"):
            st.button("See what this could cost ->", width="stretch",
                      on_click=_focus_plan, args=(sid,))

# ================= PLANNING & BUDGET =================
elif page == "plan":
    render_nav([("← Home", "home", "nav_home"), ("← Explore Bridges", "explore", "nav_explore")])
    st.markdown("## Planning & Budget")
    with st.container(border=True):
        budget = st.slider("Annual budget", 10, 100, 50, 5, format="$%dM", key="plan_budget")
        horizon = st.slider("Plan years", 1, 10, 5, 1, key="plan_horizon")
        strategy = st.selectbox(
            "Strategy",
            ["worst_first", "best_value", "fix_fair_first"],
            format_func=lambda s: {"worst_first": "Worst first", "best_value": "Best value",
                                   "fix_fair_first": "Fair bridges first"}[s],
            key="plan_strategy",
        )
        with st.expander("Assumptions"):
            esc = st.number_input("Escalation (% per year)", 0.0, 10.0, 4.0, 0.5, key="plan_esc")
            cont = st.number_input("Contingency (%)", 0.0, 40.0, 20.0, 1.0, key="plan_cont")
            regional = st.number_input("Regional multiplier", 0.5, 2.0, 1.0, 0.05, key="plan_regional")
        assumptions = {"escalation_rate": esc / 100.0, "contingency_pct": cont / 100.0,
                       "regional_multiplier": regional}
        sig = (budget, horizon, strategy, esc, cont, regional)
        if st.session_state.get("plan_sig") != sig:
            st.session_state.plan_sig = sig
            st.session_state.plan_result = fm.build_plan(
                df, budget * 1_000_000, horizon, strategy, assumptions)
        plan = st.session_state.plan_result
        funded = plan["funded"]
        st.write(
            f"With ${budget}M a year for the next {horizon} years, we could start with the "
            f"{len(funded)} bridges that need it most."
        )
        st.write(
            f"Early planning range in total: {_fmt_money(plan['total_low'])} to "
            f"{_fmt_money(plan['total_high'])}. The rest stay on watch and move up as work is finished."
        )
        fig = go.Figure(go.Bar(
            x=[a["year"] for a in plan["annual_costs"]],
            y=[a["spent"] / 1_000_000 for a in plan["annual_costs"]],
            marker=dict(color="rgba(212,175,55,0.75)"),
            hovertemplate="%{x}: $%{y:.1f}M<extra></extra>",
        ))
        fig.update_layout(
            height=260, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F5F5F7"),
            xaxis=dict(title="Year", showgrid=False),
            yaxis=dict(title="$M spent", gridcolor="rgba(255,255,255,0.08)"),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with st.container(border=True):
        st.markdown("**How these estimates are built**")
        st.markdown(
            '<div class="caption">From deck size, what the bridge crosses, traffic, complexity, '
            "and build year. Early planning range: about -20% to +35%. "
            "An engineer confirms final cost.</div>",
            unsafe_allow_html=True,
        )
    by_id = {r["stable_id"]: r for _, r in df.iterrows()}
    shortlist = []
    for f in funded[:10]:
        fr = by_id[f["stable_id"]]
        shortlist.append(f["stable_id"])
        st.button(
            f"{bridge_name(fr)} - {fr['status_title']} · {clean_county(fr['county_name'])}"
            f" · {_fmt_money(f['likely'])} in {f['target_year']}",
            key=f"fund_{f['stable_id']}",
            width="stretch",
            on_click=open_bridge,
            args=(f["stable_id"],),
        )
        est = fm.estimate_cost(fr, action=f["action"], target_year=f["target_year"],
                               assumptions=assumptions)
        comp = est["components"]
        adj = " · ".join(
            f"{v['label']} {(v['value'] - 1) * 100:+.0f}%" for v in est["factors"].values())
        with st.expander("How this estimate is built"):
            st.markdown(
                f"Deck size: {est['deck_area_sqft']:,.0f} sq ft · {est['action']} at "
                f"${est['base_rate_per_sqft']:,.0f} per sq ft  \n"
                f"Base cost: {_fmt_money(est['deck_area_sqft'] * est['base_rate_per_sqft'])}  \n"
                f"Adjustments: {adj}  \n"
                f"Construction: {_fmt_money(comp['construction'])}  \n"
                f"Design: {_fmt_money(comp['design'])} · Construction inspection: "
                f"{_fmt_money(comp['construction_inspection'])} · Contingency: "
                f"{_fmt_money(comp['contingency'])} · Escalation to {est['target_year']}: "
                f"{_fmt_money(comp['escalation'])}  \n"
                f"**Likely {_fmt_money(est['likely'])}** · Low {_fmt_money(est['low'])} · "
                f"High {_fmt_money(est['high'])}"
            )
    with st.container(key="next_step"):
        st.button("Send shortlist for Engineer Check ->", width="stretch",
                  on_click=_send_shortlist, args=(shortlist,))
    st.caption("Early planning estimates only. An engineer confirms each bridge before money is committed.")

# ================= ENGINEER =================
elif page == "engineer":
    render_nav([("← Home", "home", "nav_home"), ("← Explore Bridges", "explore", "nav_explore")])
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
        st.markdown(
            f'<div style="font-weight:700;font-size:16px;color:#F5F5F7;">Under review: {bridge_name(rrow)}</div>',
            unsafe_allow_html=True,
        )
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
    st.button("Planning & Budget ->", type="tertiary", on_click=go_to, args=("plan",))

# ================= GUIDE =================
elif page == "guide":
    render_nav([("← Home", "home", "nav_home")])
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
        '<div style="text-align:center;font-size:13px;color:#86868B;margin-top:48px;">'
        "myBridge · Predictions from 42 years of inspections · Llama explains, myBridge predicts, "
        "a licensed engineer decides.</div>",
        unsafe_allow_html=True,
    )
