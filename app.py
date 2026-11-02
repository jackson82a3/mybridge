# app.py - myBridge v8 (frozen v3 model outputs, human language)
# Reads ONLY data/app_data.parquet.
# Plain language everywhere except the engineer technical expander.

import hashlib
import io
import json
import os
import re
import textwrap
from datetime import datetime

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
    df["disp_name"] = (df["facility_carried"].fillna("").astype(str).str.strip() + " over "
                       + df["features_intersected"].fillna("").astype(str).str.strip().map(_title_features))
    age_ok = df["age_effective"].notna()
    age_s = df["age_effective"].round().astype("Int64").astype(str)
    adt_ok = df["adt"].notna()
    adt_s = df["adt"].map(lambda v: f"{int(v):,}" if pd.notna(v) else "")
    ctx = county + np.where(age_ok, " · " + age_s + " years old", " · Age not on record")
    df["context"] = ctx + np.where(adt_ok, " · About " + adt_s + " vehicles a day", "")
    wrapped = df["status_sentence"].astype(str).map(
        lambda s: "<br>".join(textwrap.wrap(s, width=40)))
    df["hover_text"] = ("<b>" + df["disp_name"] + "</b><br>" + county + "<br><br><b>"
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


_ROUTE_TOKEN = re.compile(r"(?i)^(us|mn|i|sr|cr|hwy)-?\d+$|^(us|mn|sr|i|cr)$")


def _title_features(text):
    words = []
    for w in str(text).split():
        core = w.strip(".,()")
        words.append(w.upper() if _ROUTE_TOKEN.match(core) else w.capitalize())
    return " ".join(words)


def bridge_name(row):
    fac = " ".join(clean(row["facility_carried"]).split())
    feat = _title_features(" ".join(clean(row["features_intersected"]).split()))
    return f"{fac} over {feat}"


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
if "review_queue" not in st.session_state:
    st.session_state.review_queue = {}
if isinstance(st.session_state.review_queue, list):  # legacy list -> dict
    st.session_state.review_queue = {
        sid: {"status": "Waiting for check", "submitted_from": "Explore",
              "queued_at": datetime.now().isoformat(timespec="seconds")}
        for sid in st.session_state.review_queue}
if "decisions" not in st.session_state:
    st.session_state.decisions = []
if "approved_ids" not in st.session_state:
    st.session_state.approved_ids = set()
st.session_state.approved_ids = {
    d["stable_id"] for d in st.session_state.decisions if d.get("decision") == "Approved"}
if "approved_list" not in st.session_state:
    st.session_state.approved_list = []
st.session_state.approved_list = sorted(st.session_state.approved_ids)
if "review_selected" not in st.session_state:
    st.session_state.review_selected = None
if "rating_cache" not in st.session_state:
    st.session_state.rating_cache = {}
if "last_decision" not in st.session_state:
    st.session_state.last_decision = None
if "review_notice" not in st.session_state:
    st.session_state.review_notice = None
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
if "expanded_bridge" not in st.session_state:
    st.session_state.expanded_bridge = None
if "hr_filter" not in st.session_state:
    st.session_state.hr_filter = "all"
if "hr_limit" not in st.session_state:
    st.session_state.hr_limit = 10
if "hr_sig" not in st.session_state:
    st.session_state.hr_sig = None
if "hr_scope" not in st.session_state:
    st.session_state.hr_scope = "backlog"
if "plan_funded_ids" not in st.session_state:
    st.session_state.plan_funded_ids = []


def go_to(page):
    st.session_state.page = page


def set_filter(name):
    st.session_state.risk_filter = name


def set_hr_filter(name):
    st.session_state.hr_filter = name


def set_hr_scope(name):
    st.session_state.hr_scope = name


def _view_program(ids):
    st.session_state.plan_funded_ids = list(ids)
    st.session_state.hr_scope = "funded"
    st.session_state.page = "highrisk"


def _hr_more():
    st.session_state.hr_limit += 10


def _toggle_hr(sid):
    st.session_state.expanded_bridge = (
        None if st.session_state.expanded_bridge == sid else sid)



ACTIVE_QUEUE_STATUSES = ("Waiting for check", "Under review", "Load rating verified")


def submit_for_review(sid, source="Explore"):
    """Queue a bridge for Engineer-in-the-Loop. Prevents duplicates."""
    q = st.session_state.review_queue
    entry = q.get(sid)
    if entry and entry.get("status") in ACTIVE_QUEUE_STATUSES:
        st.session_state.review_notice = "That bridge is already waiting for an engineer check."
        return False
    q[sid] = {"status": "Waiting for check", "submitted_from": source,
              "queued_at": datetime.now().isoformat(timespec="seconds")}
    match = df[df["stable_id"] == sid]
    name = bridge_name(match.iloc[0]) if not match.empty else "Bridge"
    st.session_state.review_notice = f"Added for engineer check: {name}."
    return True


def _select_review(sid):
    st.session_state.selected_bridge = sid
    st.session_state.review_selected = sid
    entry = st.session_state.review_queue.get(sid)
    if entry and entry.get("status") == "Waiting for check":
        entry["status"] = "Under review"


def _chance_phrase(row):
    if int(row["IS_ALREADY_POOR"]) == 1:
        return "Already in poor condition"
    return f"{float(row['P_POOR_5YR']) * 100:.0f}% chance in 5 years"


def _capacity_guidance(inv_rf, op_rf):
    if op_rf < 1.0:
        return ("Capacity concern",
                "The operating check is below 1.0, so this bridge may need posting "
                "or strengthening considered before any plan is approved.")
    if inv_rf < 1.0:
        return ("Worth monitoring",
                "The operating check is adequate, but the inventory check is below "
                "1.0, so heavier everyday loads deserve a closer look.")
    return ("Capacity looks adequate",
            "Both load checks are at or above 1.0, so capacity does not appear to "
            "limit the options for this bridge.")


def _rf_value(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


_RATING_KEYS = ("bridge_id", "rating_date", "software", "method",
                "controlling_member", "inventory_rf", "operating_rf",
                "posting_recommended")


def _parse_rating_bytes(raw, file_name):
    """Parse a load rating file locally. CSV is read directly; PDF/TXT text is
    extracted locally and, when Llama is configured, structured by Llama at
    temperature 0. Returns only the whitelisted rating fields."""
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    out = {k: None for k in _RATING_KEYS}
    try:
        if ext == "csv":
            frame = pd.read_csv(io.BytesIO(raw), dtype=str)
            if frame.empty:
                return out
            row = {str(k).strip().lower(): v for k, v in frame.iloc[0].to_dict().items()}
            for k in _RATING_KEYS:
                if k in row and not (isinstance(row[k], float) and pd.isna(row[k])):
                    out[k] = row[k]
            out["inventory_rf"] = _rf_value(out["inventory_rf"])
            out["operating_rf"] = _rf_value(out["operating_rf"])
            return out
        if ext == "pdf":
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                text = "\n".join((pg.extract_text() or "") for pg in pdf.pages)
        else:
            text = raw.decode("utf-8", errors="ignore")
        text = text[:8000]
        if _skip_reason() is not None:
            return out
        resp = _llama_client().chat.completions.create(
            model=LLAMA_MODEL, temperature=0, max_tokens=300,
            messages=[
                {"role": "system",
                 "content": "Extract load rating fields. Return ONLY JSON with keys "
                            "rating_date, software, method, controlling_member, "
                            "inventory_rf, operating_rf, posting_recommended. Use null "
                            "for anything not present. Never invent values."},
                {"role": "user", "content": text},
            ],
        )
        raw_text = (resp.choices[0].message.content or "").strip()
        raw_text = re.sub(r"^```(?:json)?|```$", "", raw_text).strip()
        parsed = json.loads(raw_text)
        for k in _RATING_KEYS:
            if k in parsed:
                out[k] = parsed[k]
        out["inventory_rf"] = _rf_value(out["inventory_rf"])
        out["operating_rf"] = _rf_value(out["operating_rf"])
        if isinstance(out.get("posting_recommended"), bool):
            out["posting_recommended"] = "Yes" if out["posting_recommended"] else "No"
    except Exception:
        pass  # silent fallback: engineer enters values manually
    return out


def send_for_check(sid, source="Explore"):
    submit_for_review(sid, source)
    st.session_state.selected_bridge = sid
    st.session_state.review_selected = sid
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


def format_money(v):
    v = float(v)
    if abs(v) >= 1_000_000:
        text = f"{v / 1_000_000:,.1f}".removesuffix(".0")
        return f"${text}M"
    if abs(v) >= 10_000:
        return f"${v / 1_000:,.0f}k"
    return f"${v:,.0f}"


def _row_subline(row):
    parts = [row["status_title"], clean_county(row["county_name"])]
    if int(row["IS_ALREADY_POOR"]) != 1:
        parts.append(f"{float(row['P_POOR_5YR']) * 100:.0f}% chance in 5 years")
    return " · ".join(parts)


def _breakdown_text(est):
    comp = est["components"]
    adj = " · ".join(
        f"{v['label']} {(v['value'] - 1) * 100:+.0f}%" for v in est["factors"].values())
    return (
        f"Deck size: {est['deck_area_sqft']:,.0f} sq ft · {est['action']} at "
        f"${est['base_rate_per_sqft']:,.0f} per sq ft  \n"
        f"Base cost: ${est['deck_area_sqft'] * est['base_rate_per_sqft']:,.0f}  \n"
        f"Adjustments: {adj}  \n"
        f"Construction: ${comp['construction']:,.0f}  \n"
        f"Design: ${comp['design']:,.0f} · Construction inspection: "
        f"${comp['construction_inspection']:,.0f} · Contingency: ${comp['contingency']:,.0f} · "
        f"Escalation to {est['target_year']}: ${comp['escalation']:,.0f}  \n"
        f"**Likely ${est['likely']:,.0f}** · Low ${est['low']:,.0f} · High ${est['high']:,.0f}"
    )


def _focus_plan(sid):
    st.session_state.selected_bridge = sid
    st.session_state.focus_bridge = sid
    st.session_state.page = "plan"


def _send_shortlist(ids):
    for sid in ids:
        submit_for_review(sid, "Plan & Estimate")
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


def render_bridge_list(items, mode):
    with st.container(key="bridge_list"):
        for idx, it in enumerate(items):
            if idx:
                st.markdown(
                    '<div style="border-top:1px solid rgba(255,255,255,0.06);"></div>',
                    unsafe_allow_html=True,
                )
            if mode == "open":
                st.button(f"{it['line1']}\n{it['line2']}", key=f"open_{it['sid']}",
                          width="stretch", on_click=open_bridge, args=(it["sid"],))
            else:
                st.button(f"{it['line1']} \u203a\n{it['line2']}", key=f"expand_{it['sid']}",
                          width="stretch", on_click=_toggle_hr, args=(it["sid"],))
                if st.session_state.expanded_bridge == it["sid"]:
                    tr, est = it["row"], it["est"]
                    st.markdown(f'<div class="caption">{tr["context"]}</div>',
                                unsafe_allow_html=True)
                    st.write(human_summary(tr))
                    st.markdown(
                        f"**Planning estimate:** {format_money(est['likely'])} "
                        f"(range {format_money(est['low'])} to {format_money(est['high'])}) "
                        f"in {est['target_year']}"
                    )
                    v1, v2 = st.columns(2, gap="medium")
                    v1.button("View bridge", key=f"list_view_{it['sid']}", width="stretch",
                              on_click=_ask_view, args=(it["sid"],))
                    v2.button("Send to Engineer-in-the-Loop", key=f"list_send_{it['sid']}",
                              width="stretch", on_click=send_for_check,
                              args=(it["sid"], "High Risk"))
                    with st.expander("How this estimate is built"):
                        st.markdown(_breakdown_text(est))


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
  --metal-hi: #FFFEFB;
  --metal-1: #F4F0E4;
  --metal-2: #E6DFCE;
  --metal-3: #CFC5AB;
  --metal-4: #A99E83;
  --metal-glow: rgba(230,223,206,0.28);
  --metal-border: rgba(230,223,206,0.55);
}
html, body, [class*="css"], .stApp { font-family:'Inter',-apple-system,'SF Pro Display',sans-serif; }
.stApp { background-color:#080B0F; background-image:
  radial-gradient(900px 480px at 50% 6%, rgba(230,223,206,0.13), transparent 70%),
  radial-gradient(800px 600px at 95% 100%, rgba(90,120,160,0.08), transparent 70%);
  background-attachment:fixed; color:#F5F5F7; }
[data-testid="stHeader"], footer, [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], #MainMenu { display:none !important; }
.block-container { max-width:1180px; padding-top:0 !important; padding-bottom:64px; }
.wordmark { text-align:center; font-weight:700; font-size:15px; letter-spacing:0.28em;
  text-transform:uppercase; padding-top:40px;
  background:linear-gradient(180deg, #FFFFFF, #E6DFCE 55%, #B9AE94);
  -webkit-background-clip:text; background-clip:text;
  color:transparent; -webkit-text-fill-color:transparent; }
.hero { text-align:center; padding:8px 0 8px; }
.hero h1 { font-size:52px; font-weight:700; letter-spacing:-0.03em; margin:16px 0 32px;
  color:#F5F5F7; }
svg[role="img"] { filter:drop-shadow(0 0 14px rgba(240,236,222,0.22)) drop-shadow(0 0 34px rgba(190,178,148,0.14)); }
h1 > a, h2 > a, h3 > a { display:none !important; }
.mcard { background:var(--glass); border:var(--glass-border); border-radius:22px;
  box-shadow:var(--glass-shadow); backdrop-filter:blur(20px) saturate(140%);
  -webkit-backdrop-filter:blur(20px) saturate(140%); padding:28px 32px; margin:16px 0; }
.bridge-strong { background:linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.045)) !important;
  box-shadow:0 20px 50px rgba(0,0,0,0.35), inset 0 1px 0 rgba(230,223,206,0.25) !important; }
div[data-testid="stVerticalBlockBorderWrapper"] > div { background:var(--glass) !important;
  border:var(--glass-border) !important; border-radius:22px !important;
  box-shadow:var(--glass-shadow) !important; backdrop-filter:blur(20px) saturate(140%) !important;
  -webkit-backdrop-filter:blur(20px) saturate(140%) !important; padding:24px !important; }
[data-testid="stExpander"] { background:var(--glass); border:var(--glass-border);
  border-radius:22px; box-shadow:var(--glass-shadow); backdrop-filter:blur(20px) saturate(140%);
  -webkit-backdrop-filter:blur(20px) saturate(140%); }
div[data-testid="stButton"] button { transition:transform .15s ease, box-shadow .15s ease, filter .15s ease, border-color .15s ease; }
div[data-testid="stButton"] button[kind="primary"] {
  background:linear-gradient(180deg, #FFFEFB 0%, #EDE7D7 48%, #CFC5AB 100%) !important; color:#171A1F !important;
  border:1px solid rgba(255,255,255,0.55) !important; border-radius:980px !important; padding:12px 32px !important;
  font-weight:700 !important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,0.85), 0 10px 28px rgba(230,223,206,0.16) !important; }
div[data-testid="stButton"] button[kind="primary"]:hover { transform:translateY(-1px);
  box-shadow:inset 0 1px 0 rgba(255,255,255,0.9), 0 14px 34px rgba(230,223,206,0.22) !important; }
div[data-testid="stButton"] button[kind="secondary"] { background:rgba(255,255,255,0.07) !important;
  backdrop-filter:blur(20px) saturate(140%); -webkit-backdrop-filter:blur(20px) saturate(140%);
  color:#fff !important; border:1px solid rgba(255,255,255,0.14) !important;
  border-radius:980px !important; padding:12px 32px !important; font-weight:600 !important; }
div[data-testid="stButton"] button[kind="secondary"]:hover { transform:translateY(-1px);
  border-color:rgba(255,255,255,0.25) !important;
  box-shadow:0 8px 24px rgba(0,0,0,0.35), 0 0 18px rgba(230,223,206,0.12) !important; }
div[data-testid="stButton"] button[kind="tertiary"] { background:transparent !important;
  color:#E6DFCE !important; border:none !important; font-weight:600 !important;
  box-shadow:none !important; padding:8px 12px !important; }
div[data-testid="stButton"] button[kind="tertiary"]:hover { transform:translateY(-1px); }
div[data-testid="stButton"] button:disabled { background:#2A3140 !important;
  color:#6B7280 !important; border:none !important; transform:none !important; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
  background:rgba(255,255,255,0.05) !important;
  border:1px solid rgba(255,255,255,0.10) !important;
  border-radius:14px !important; color:#F5F5F7 !important; }
[data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus {
  border-color:rgba(230,223,206,0.55) !important;
  box-shadow:0 0 0 1px rgba(230,223,206,0.55) !important; outline:none !important; }
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
.pill-likely { background:rgba(230,223,206,0.14); color:#E6DFCE; border:1px solid rgba(230,223,206,0.25); }
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
.stepnum { font-size:40px; font-weight:800; color:#E6DFCE; line-height:1; }
.steptitle { font-size:17px; font-weight:700; margin:8px 0 4px; }
.stepline { font-size:14px; color:#9A9AA0; }
.arrow { text-align:center; font-size:24px; color:#4A5568; padding-top:40px; }
.flowpill { background:var(--glass); border:var(--glass-border); border-radius:980px;
  backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
  padding:10px 18px; font-size:14px; font-weight:600; text-align:center; }
@keyframes draw { to { stroke-dashoffset:0; } }
@keyframes glow { 0%,100% { filter:drop-shadow(0 0 0 rgba(230,223,206,0)); }
  50% { filter:drop-shadow(0 0 14px rgba(240,236,222,0.22)); } }
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

.brand-mark { text-align: right; letter-spacing: 0.32em; font-weight: 700; font-size: 14px; padding-top: 6px;
  background: linear-gradient(180deg, #FFFFFF, #E6DFCE 55%, #B9AE94);
  -webkit-background-clip: text; background-clip: text;
  color: transparent; -webkit-text-fill-color: transparent; }

.st-key-nav_home button, .st-key-nav_explore button {
  width: auto !important; min-width: 0 !important; height: auto !important; min-height: 0 !important;
  padding: 7px 16px !important; border-radius: 980px !important;
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  color: #D8D8DC !important; font-size: 14px !important; font-weight: 400 !important;
  box-shadow: none !important; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
}
.st-key-nav_home button:hover, .st-key-nav_explore button:hover {
  background: rgba(230,223,206,0.10) !important; border-color: rgba(230,223,206,0.55) !important; color: #FFFFFF !important;
}

.st-key-bridge_search input { background: rgba(255,255,255,0.05) !important; color: #F5F5F7 !important; -webkit-text-fill-color: #F5F5F7 !important; caret-color: #E6DFCE; border: 1px solid rgba(255,255,255,0.10) !important; border-radius: 14px !important; }
.st-key-bridge_search input::placeholder { color: #8E8E93 !important; opacity: 1 !important; }
.st-key-bridge_search input:focus { border-color: rgba(230,223,206,0.55) !important; box-shadow: 0 0 0 1px rgba(230,223,206,0.20) !important; outline: none !important; }
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
  color: #F5F5F7 !important; -webkit-text-fill-color: #F5F5F7 !important; caret-color: #E6DFCE !important;
  border: 1px solid rgba(255,255,255,0.10) !important; border-radius: 14px !important; box-shadow: none !important; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stTextArea"] textarea,
[data-baseweb="input"] input, [data-baseweb="base-input"] input, [data-baseweb="select"] input {
  background: transparent !important; background-color: transparent !important;
  color: #F5F5F7 !important; -webkit-text-fill-color: #F5F5F7 !important; caret-color: #E6DFCE !important;
  border: none !important; box-shadow: none !important; outline: none !important; }
[data-testid="stTextInput"] input::placeholder, [data-testid="stNumberInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
  color: #8E8E93 !important; -webkit-text-fill-color: #8E8E93 !important; opacity: 1 !important; }
[data-testid="stTextInput"]:focus-within, [data-testid="stNumberInput"]:focus-within,
[data-testid="stTextArea"]:focus-within, [data-baseweb="input"]:focus-within,
[data-baseweb="base-input"]:focus-within, [data-baseweb="select"]:focus-within {
  border-color: rgba(230,223,206,0.55) !important; box-shadow: 0 0 0 1px rgba(230,223,206,0.20) !important; outline: none !important; }
[data-testid="stWidgetLabel"] p { color: #9A9AA0 !important; }
[data-testid="InputInstructions"] { display: none !important; }
[data-testid="stNumberInput"] button, [data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {
  background: rgba(255,255,255,0.08) !important; color: #F5F5F7 !important; border: 1px solid rgba(255,255,255,0.10) !important; }
[data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,0.03) !important; border: 1px dashed rgba(255,255,255,0.14) !important; color: #C7C7CC !important; }
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span { color: #C7C7CC !important; }
[data-testid="stFileUploaderDropzone"] button {
  background: rgba(255,255,255,0.08) !important; color: #FFFFFF !important; border: 1px solid rgba(255,255,255,0.14) !important; }
[data-testid="stProgress"] > div { background: rgba(255,255,255,0.08) !important; }
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, #F4F0E4, #E6DFCE) !important; }

.st-key-ask_mybridge { max-width: 760px; margin: 30px auto 0 !important; }
.st-key-ask_mybridge div[data-testid="stButton"] button[kind="primary"] { height: 44px !important; }
h1 a, [data-testid="StyledLinkIconContainer"] { display: none !important; }
.st-key-suggested_inquiries [data-testid="stButton"] button {
  background: transparent !important; border: none !important; box-shadow: none !important;
  color: #9A9AA0 !important; font-size: 13px !important; font-weight: 400 !important;
  padding: 3px 0 !important; height: auto !important; min-height: 0 !important;
  text-align: left !important; justify-content: flex-start !important;
  white-space: normal !important; line-height: 1.4 !important; }
.st-key-suggested_inquiries [data-testid="stButton"] button:hover { color: #E6DFCE !important; }
.st-key-suggested_inquiries [data-testid="stButton"] button p {
  text-align: left !important; margin: 0 !important; white-space: normal !important; }
.st-key-bridge_list [data-testid="stButton"] button {
  width: 100% !important; height: auto !important; min-height: 0 !important;
  background: transparent !important; border: none !important; box-shadow: none !important;
  border-radius: 10px !important; padding: 10px 6px !important;
  font-size: 14px !important; font-weight: 500 !important; color: #F5F5F7 !important;
  text-align: left !important; justify-content: flex-start !important;
  white-space: pre-line !important; }
.st-key-bridge_list [data-testid="stButton"] button:hover {
  background: rgba(230,223,206,0.06) !important; color: #FFFFFF !important; }
.st-key-bridge_list [data-testid="stButton"] button p {
  text-align: left !important; white-space: pre-line !important; line-height: 1.35 !important; margin: 0 !important; }
.st-key-next_step [data-testid="stButton"] button {
  width: 100% !important; height: 52px !important; background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(230,223,206,0.55) !important; border-radius: 16px !important;
  color: #E6DFCE !important; font-size: 15px !important; font-weight: 600 !important;
  box-shadow: none !important; }
.st-key-next_step [data-testid="stButton"] button:hover {
  background: rgba(230,223,206,0.10) !important; border-color: rgba(230,223,206,0.45) !important;
  color: #FFFFFF !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
/* HOME TILES - small rectangles */
.st-key-home_tiles {
  max-width: 680px !important;
  margin: 10px auto 0 auto !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.st-key-home_tiles [data-testid="stHorizontalBlock"] {
  gap: 16px !important;
  justify-content: center !important;
}
.st-key-home_tiles [data-testid="stColumn"] {
  display: flex !important;
  justify-content: center !important;
}
.st-key-home_tiles .stButton {
  width: 200px !important;
  max-width: 200px !important;
  flex: 0 0 200px !important;
  display: flex !important;
  justify-content: center !important;
}
.stApp .st-key-home_tiles .stButton > button {
  width: 200px !important;
  min-width: 200px !important;
  max-width: 200px !important;
  height: 104px !important;
  min-height: 104px !important;
  max-height: 104px !important;
  aspect-ratio: auto !important;
  border-radius: 18px !important;
  padding: 14px 16px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  text-align: center !important;
  white-space: normal !important;
  line-height: 1.25 !important;
  font-size: 16px !important;
  font-weight: 600 !important;
  color: #F5F5F7 !important;
  background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.028)) !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 10px 26px rgba(0,0,0,0.26) !important;
  backdrop-filter: blur(18px) saturate(140%) !important;
  -webkit-backdrop-filter: blur(18px) saturate(140%) !important;
  transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease, background 0.25s ease !important;
}
.stApp .st-key-home_tiles .stButton > button p {
  text-align: center !important;
  margin: 0 !important;
  white-space: normal !important;
  font-size: 16px !important;
  line-height: 1.25 !important;
}
.stApp .st-key-home_tiles [data-testid="stMarkdownContainer"] {
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
  max-width: 118px !important;
  margin: 0 auto !important;
}
.stApp .st-key-home_tiles .stButton > button:hover,
.stApp .st-key-home_tiles .stButton > button:focus-visible {
  transform: translateY(-2px) !important;
  background: linear-gradient(135deg, rgba(230,223,206,0.13), rgba(255,255,255,0.035)) !important;
  border-color: rgba(230,223,206,0.55) !important;
  color: #FFFFFF !important;
  outline: none !important;
  box-shadow: inset 0 1px 0 rgba(230,223,206,0.25), 0 0 0 1px rgba(230,223,206,0.18), 0 0 30px rgba(230,223,206,0.28), 0 16px 34px rgba(0,0,0,0.34) !important;
}
@media (max-width: 700px) {
  .stApp .st-key-home_tiles .stButton,
  .stApp .st-key-home_tiles .stButton > button {
    width: 168px !important;
    min-width: 168px !important;
    max-width: 168px !important;
    height: 92px !important;
    min-height: 92px !important;
    max-height: 92px !important;
    font-size: 15px !important;
  }
}
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
<defs>
<linearGradient id="metalGradient" x1="0%" y1="0%" x2="100%" y2="0%">
<stop offset="0%" stop-color="#8F8875"/>
<stop offset="22%" stop-color="#F6F2E6"/>
<stop offset="48%" stop-color="#CFC5AB"/>
<stop offset="68%" stop-color="#FFFFFF"/>
<stop offset="100%" stop-color="#B3A789"/>
</linearGradient>
</defs>
<g class="bridge-glow" fill="none" stroke="url(#metalGradient)" stroke-width="3" stroke-linecap="round">
<g class="bridge-draw">
<path d="M20,140 L150,45 Q280,190 410,45 L540,140"/>
<path d="M20,140 L540,140"/>
<path d="M150,45 L150,140 M410,45 L410,140"/>
<path d="M132,70 L168,70 M392,70 L428,70"/>
<path d="M180,75 L180,140 M215,99 L215,140 M280,118 L280,140 M345,99 L345,140 M380,75 L380,140"/>
</g>
<path d="M120,168 L440,168" stroke="#A99E83" stroke-width="2" opacity="0.45"/>
</g>
<circle r="4.5" fill="#F7F3E8">
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
    with st.container(key="home_tiles"):
        t1, t2, t3 = st.columns(3, gap="large")
        t1.button("Explore",
                  key="tile_explore", width="stretch", on_click=go_to, args=("explore",))
        t2.button("$ Plan & Estimate",
                  key="tile_plan", width="stretch", on_click=go_to, args=("plan",))
        t3.button("Engineer-in-the-Loop",
                  key="tile_engineer", width="stretch", on_click=go_to, args=("engineer",))
    with st.container(key="ask_mybridge"):
        st.markdown(
            '<div style="color:#F4F0E4;font-size:13px;font-weight:700;'
            'letter-spacing:0.18em;margin:6px 0 10px;">Ask myBridge</div>',
            unsafe_allow_html=True,
        )
        a_in, a_btn = st.columns([4, 1], gap="medium")
        a_in.text_input(
            "Ask myBridge",
            placeholder="Ask about bridge condition, priorities, program funding, or cost estimates…",
            key="ask_mybridge_input",
            label_visibility="collapsed",
        )
        a_btn.button("Ask", type="primary", width="stretch", on_click=_ask_submit)
        with st.container(key="suggested_inquiries"):
            st.markdown(
                '<div style="font-size:12px;color:#86868B;margin:10px 0 4px;">Suggested inquiries</div>',
                unsafe_allow_html=True,
            )
            inquiries = [
                "Identify bridges requiring major repair in Marion County.",
                "Prepare a prioritized five-year program within $40 million annually.",
                "Provide a planning-level estimate for I-65 SB over Keystone Avenue.",
            ]
            for i, text in enumerate(inquiries):
                st.button("\u203a " + text, key=f"ask_chip_{i}", width="stretch",
                          on_click=_ask_chip, args=(text,))
    if st.session_state.ask_pending:
        st.session_state.ask_pending = False
        _run_ask()
    ans = st.session_state.ask_answer
    if ans:
        with st.container(border=True):
            st.markdown(f"**I understood:** {ans['understood']}")
            st.write(ans["answer"])
            open_label = "View bridge" if ans["kind"] == "estimate_bridge_cost" else "View on map"
            for b in ans["bridges"][:5]:
                r1, r2, r3 = st.columns([3, 1, 1], gap="small")
                veh = (f" · About {b['vehicles_per_day']:,} vehicles a day"
                       if b.get("vehicles_per_day") else "")
                yr = f" · Target {b['target_year']}" if b.get("target_year") else ""
                r1.markdown(f"**{b['name']}**  \n{b['county']} · {b['status']}{veh}{yr}")
                r2.button(open_label, key=f"ask_view_{b['stable_id']}",
                          on_click=_ask_view, args=(b["stable_id"],))
                r3.button("Send to Engineer-in-the-Loop", key=f"ask_send_{b['stable_id']}",
                          on_click=send_for_check, args=(b["stable_id"], "Ask myBridge"))
            if ans.get("plan"):
                pl = ans["plan"]
                st.button("See this plan", type="primary", on_click=_ask_open_plan,
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
            btxt, bbtn = st.columns([4, 1])
            btxt.markdown(
                f"**{int((df['IS_ALREADY_POOR'] == 1).sum()):,} need major work now · "
                f"{int(((df['IS_ALREADY_POOR'] == 0) & (df['P_POOR_5YR'] > 0.3)).sum()):,} "
                "could within 5 years**"
            )
            bbtn.button("View ->", key="hr_banner", width="stretch",
                        on_click=go_to, args=("highrisk",))
        with st.container(border=True):
            st.markdown(
                '<div class="legend">'
                '<span class="lpill"><span class="dot" style="background:#E5484D"></span>Needs major work now</span>'
                '<span class="lpill"><span class="dot" style="background:#E8A33D"></span>Plan soon</span>'
                '<span class="lpill"><span class="dot" style="background:#E6DFCE"></span>Plan ahead</span>'
                '<span class="lpill"><span class="dot" style="background:#6E7681"></span>Doing well for now</span>'
                "</div>",
                unsafe_allow_html=True,
            )
            band_style = {
                "Needs major work now": ("#E5484D", 9),
                "Plan soon": ("#E8A33D", 7),
                "Plan ahead": ("#E6DFCE", 6),
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
        if sid in st.session_state.get("approved_ids", set()):
            st.markdown(
                '<span class="pill pill-likely">Approved for funding plan</span>',
                unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")
        if c1.button("Overview", type="primary", width="stretch"):
            st.session_state.show_summary = not st.session_state.show_summary
        c2.button("Send to Engineer-in-the-Loop", type="tertiary", width="stretch",
                  on_click=send_for_check, args=(sid, "Explore"))
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
            est = fm.estimate_cost(row, target_year=max(int(row["PROGRAM_YEAR"]), 2027))
            st.markdown("**Estimate breakdown**")
            st.markdown(_breakdown_text(est))
        with st.container(key="next_step"):
            st.button("See what this could cost ->", width="stretch",
                      on_click=_focus_plan, args=(sid,))

# ================= HIGH RISK =================
elif page == "highrisk":
    render_nav([("\u2190 Home", "home", "nav_home"), ("\u2190 Explore", "explore", "nav_explore")])
    st.markdown("## High Risk")
    st.markdown('<div class="caption">Bridges needing attention first.</div>', unsafe_allow_html=True)

    funded_ids = st.session_state.get("plan_funded_ids") or []
    scope = st.session_state.hr_scope
    if scope == "funded" and not funded_ids:
        scope = "backlog"
    s1c, s2c, s3c = st.columns(3, gap="medium")
    s1c.button("High-risk backlog", type="primary" if scope == "backlog" else "secondary",
               width="stretch", on_click=set_hr_scope, args=("backlog",))
    s2c.button("Funded in current plan", type="primary" if scope == "funded" else "secondary",
               width="stretch", on_click=set_hr_scope, args=("funded",))
    s3c.button("Still waiting", type="primary" if scope == "waiting" else "secondary",
               width="stretch", on_click=set_hr_scope, args=("waiting",))
    scope = st.session_state.hr_scope
    if scope == "funded" and not funded_ids:
        scope = "backlog"

    hq = st.text_input("Search", placeholder="Search a road, river, town, or county",
                       key="highrisk_search")
    hf = st.session_state.hr_filter
    h1c, h2c, h3c = st.columns(3, gap="medium")
    h1c.button("All high risk", type="primary" if hf == "all" else "secondary", width="stretch",
               on_click=set_hr_filter, args=("all",))
    h2c.button("Repair now", type="primary" if hf == "now" else "secondary", width="stretch",
               on_click=set_hr_filter, args=("now",))
    h3c.button("Plan soon", type="primary" if hf == "soon" else "secondary", width="stretch",
               on_click=set_hr_filter, args=("soon",))
    hf = st.session_state.hr_filter

    hr_all = df[(df["IS_ALREADY_POOR"] == 1) | (df["P_POOR_5YR"] > 0.3)]
    funded_map = {}
    if st.session_state.get("plan_result"):
        funded_map = {f["stable_id"]: f for f in st.session_state.plan_result["funded"]}
    if scope == "funded":
        hr = df[df["stable_id"].isin(funded_ids)].copy()
        order = {sid: i for i, sid in enumerate(funded_ids)}
        hr["_ord"] = hr["stable_id"].map(order)
        hr = hr.sort_values("_ord")
    elif scope == "waiting":
        hr = hr_all[~hr_all["stable_id"].isin(funded_ids)]
    else:
        hr = hr_all
    if hf == "now":
        hr = hr[hr["IS_ALREADY_POOR"] == 1]
    elif hf == "soon":
        hr = hr[(hr["IS_ALREADY_POOR"] == 0) & (hr["P_POOR_5YR"] > 0.3)]
    if hq.strip():
        ql = hq.strip().lower()
        hr = hr[
            hr["facility_carried"].fillna("").str.lower().str.contains(ql)
            | hr["features_intersected"].fillna("").str.lower().str.contains(ql)
            | hr["county_name"].fillna("").str.lower().str.contains(ql)
        ]
    if scope != "funded":
        hr = hr.sort_values("P_POOR_5YR", ascending=False)
    sig = (hq.strip().lower(), hf, scope)
    if st.session_state.hr_sig != sig:
        st.session_state.hr_sig = sig
        st.session_state.hr_limit = 10
    hr = hr.drop_duplicates("stable_id")
    with st.container(border=True):
        if hr.empty:
            st.markdown(
                '<div class="caption">No bridges match. Try widening the search.</div>',
                unsafe_allow_html=True)
        items = []
        for _, tr in hr.head(st.session_state.hr_limit).iterrows():
            fe = funded_map.get(tr["stable_id"])
            target = fe["target_year"] if fe else max(int(tr["PROGRAM_YEAR"]), 2027)
            est = fm.estimate_cost(tr, target_year=target)
            year_label = f"Start {est['target_year']}" if fe else f"Act by {est['target_year']}"
            items.append({
                "sid": tr["stable_id"], "row": tr, "est": est,
                "line1": tr["disp_name"],
                "line2": (f"{tr['status_title']} \u00b7 {clean_county(tr['county_name'])} \u00b7 "
                          f"{format_money(est['likely'])} \u00b7 {year_label}"
                          + (" \u00b7 Approved for funding plan"
                             if tr["stable_id"] in st.session_state.get("approved_ids", set())
                             else "")),
            })
        render_bridge_list(items, "expand")
        if len(hr) > st.session_state.hr_limit:
            st.button("Show more", key="hr_more", on_click=_hr_more)

# ================= PLAN & ESTIMATE =================
elif page == "plan":
    render_nav([("← Home", "home", "nav_home"), ("← Explore", "explore", "nav_explore")])
    st.markdown("## $ Plan & Estimate")
    with st.container(border=True):
        budget = st.slider("Annual budget", 10, 100, 50, 5, format="$%dM/yr", key="plan_budget")
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
        sig = ("v2", budget, horizon, strategy, esc, cont, regional)
        if st.session_state.get("plan_sig") != sig:
            st.session_state.plan_sig = sig
            st.session_state.plan_result = fm.build_plan(
                df, budget * 1_000_000, horizon, strategy, assumptions)
        plan = st.session_state.plan_result
        funded = plan["funded"]
        st.write(
            f"With {format_money(budget * 1_000_000)} a year for the next {horizon} years, "
            f"we could start with the {len(funded)} bridges that need it most."
        )
    if not funded:
        st.markdown(
            '<div class="caption">No bridges fit this budget yet. Try a higher yearly budget.</div>',
            unsafe_allow_html=True,
        )
    else:
        total_hr = int(((df["IS_ALREADY_POOR"] == 1) | (df["P_POOR_5YR"] > 0.3)).sum())
        pct = (len(funded) / total_hr * 100) if total_hr else 0.0
        annual = plan["annual_costs"]
        with st.container(border=True):
            st.markdown("**Plan at a glance**")
            s1, s2, s3 = st.columns(3)
            s1.markdown(
                f'<div style="font-size:30px;font-weight:700;">{len(funded):,}</div>'
                '<div class="caption">Bridges planned</div>', unsafe_allow_html=True)
            s2.markdown(
                f'<div style="font-size:30px;font-weight:700;">'
                f'{format_money(plan["total_low"])} \u2013 {format_money(plan["total_high"])}</div>'
                '<div class="caption">Planning range \u00b7 Early planning estimate</div>',
                unsafe_allow_html=True)
            s3.markdown(
                f'<div style="font-size:30px;font-weight:700;">{len(funded):,} of {total_hr:,}</div>'
                f'<div class="caption">High-risk bridges covered \u00b7 {pct:.0f}%</div>',
                unsafe_allow_html=True)
            st.markdown("**Bridges starting each year**")
            st.markdown(
                '<div class="caption">Each bridge is scheduled as one whole project, so years vary.</div>',
                unsafe_allow_html=True)
            years = [a["year"] for a in annual]
            totals = [a["count"] for a in annual]
            hover_cd = [[format_money(a["low"]), format_money(a["high"]),
                         a["count"], a["cumulative_count"]] for a in annual]
            action_colors = {"Plan replacement": "#C96A6A",
                             "Plan rehabilitation": "#E6DFCE",
                             "Preventive work": "#7A8699"}
            fig = go.Figure()
            for act, color in action_colors.items():
                fig.add_trace(go.Bar(
                    x=years,
                    y=[a["by_action"].get(act, 0) for a in annual],
                    name=act,
                    marker=dict(color=color),
                    customdata=hover_cd,
                    hovertemplate=("%{x}<br>%{customdata[2]} bridges starting<br>"
                                   "Planning estimate %{customdata[0]} \u2013 %{customdata[1]}<br>"
                                   "%{customdata[3]} planned by end of %{x}<extra></extra>"),
                ))
            fig.add_trace(go.Scatter(
                x=years, y=totals, mode="text",
                text=[str(t) for t in totals],
                textposition="top center",
                textfont=dict(color="#F5F5F7", size=13),
                showlegend=False, hoverinfo="skip",
            ))
            fig.update_layout(
                barmode="stack", height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F5F5F7"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                xaxis=dict(title="Year", showgrid=False),
                yaxis=dict(title="Bridges", dtick=1, gridcolor="rgba(255,255,255,0.06)",
                           range=[0, max(totals) * 1.18 if totals else 1]),
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.progress(len(funded) / total_hr if total_hr else 0.0)
            st.markdown(
                f'<div class="caption" style="text-align:right;">{len(funded):,} funded \u00b7 '
                f'{total_hr - len(funded):,} still waiting for a later plan</div>',
                unsafe_allow_html=True,
            )
        with st.expander("Year by year"):
            ydf = pd.DataFrame({
                "Year": [a["year"] for a in annual],
                "Bridges planned": [a["count"] for a in annual],
                "Planning estimate": [
                    f"{format_money(a['low'])} \u2013 {format_money(a['high'])}" for a in annual],
                "Planned so far": [a["cumulative_count"] for a in annual],
            })
            st.dataframe(ydf, hide_index=True, width="stretch")
    if funded:
        by_id = {r["stable_id"]: r for _, r in df.iterrows()}
        items = []
        for f in funded[:5]:
            fr = by_id[f["stable_id"]]
            items.append({
                "sid": f["stable_id"], "row": fr, "est": None,
                "line1": fr["disp_name"],
                "line2": (f"{fr['status_title']} \u00b7 {clean_county(fr['county_name'])} \u00b7 "
                          f"{format_money(f['likely'])} in {f['target_year']}"),
            })
        with st.container(border=True):
            st.markdown("**Funded program**")
            st.markdown(
                f'<div class="caption">{len(funded):,} bridges \u00b7 '
                f'{format_money(plan["total_low"])} \u2013 {format_money(plan["total_high"])} \u00b7 '
                f'{annual[0]["year"]}\u2013{annual[-1]["year"]}</div>',
                unsafe_allow_html=True,
            )
            render_bridge_list(items, "open")
            with st.container(key="next_step"):
                st.button("Send shortlist to Engineer-in-the-Loop", width="stretch",
                          on_click=_send_shortlist, args=([it["sid"] for it in items],))
            st.button("View complete program \u2192", type="tertiary",
                      on_click=_view_program, args=([f["stable_id"] for f in funded],))
            st.caption("Early planning estimates only. An engineer confirms each bridge before money is committed.")

# ================= ENGINEER =================
elif page == "engineer":
    render_nav([("← Home", "home", "nav_home"), ("← Explore", "explore", "nav_explore")])
    st.markdown("## Engineer-in-the-Loop")
    st.markdown('<div class="caption">myBridge suggests. You confirm.</div>', unsafe_allow_html=True)

    _notice = st.session_state.pop("review_notice", None)
    if _notice:
        st.success(_notice)
    _last = st.session_state.get("last_decision")
    if _last:
        st.success(
            f"Decision {_last['decision_id']} recorded: {_last['bridge_name']} — "
            f"{_last['decision']} ({_last['final_action']}, act by {_last['final_year']})."
        )

    queue = st.session_state.review_queue
    approved_ids = st.session_state.get("approved_ids", set())
    n_wait = sum(1 for e in queue.values() if e.get("status") == "Waiting for check")
    n_under = sum(1 for e in queue.values() if e.get("status") == "Under review")
    c_wait, c_under, c_appr = st.columns(3, gap="medium")
    c_wait.metric("Waiting for check", f"{n_wait:,}")
    c_under.metric("Under review", f"{n_under:,}")
    c_appr.metric("Approved this session", f"{len(approved_ids):,}")

    active_ids = [sid for sid, e in queue.items()
                  if e.get("status") in ACTIVE_QUEUE_STATUSES]
    with st.container(border=True):
        st.markdown("**Review queue**")
        if not active_ids:
            st.markdown(
                '<div class="caption">Queue is empty. Send bridges here from Explore, '
                "High Risk, or Plan &amp; Estimate.</div>",
                unsafe_allow_html=True,
            )
        status_pill = {
            "Waiting for check": "pill-gray",
            "Under review": "pill-likely",
            "Load rating verified": "pill-green",
        }
        for qsid in active_ids:
            qr = df[df["stable_id"] == qsid]
            if qr.empty:
                continue
            qr = qr.iloc[0]
            entry = queue[qsid]
            qcols = st.columns([5, 2], gap="medium")
            qcols[0].button(
                f"{bridge_name(qr)}\n{qr['status_title']} · "
                f"{clean_county(qr['county_name'])} · {_chance_phrase(qr)} · "
                f"From {entry.get('submitted_from', 'Explore')}",
                key=f"qrow_{qsid}", width="stretch",
                on_click=_select_review, args=(qsid,),
            )
            qcols[1].markdown(
                f'<span class="pill {status_pill.get(entry.get("status"), "pill-gray")}">'
                f"{entry.get('status')}</span>",
                unsafe_allow_html=True,
            )

    # ---- Review workspace for the selected bridge ----
    ws_sid = st.session_state.get("review_selected") or st.session_state.selected_bridge
    if ws_sid not in set(df["stable_id"]):
        ws_sid = st.session_state.selected_bridge
    _, rrow = selected_row() if ws_sid == st.session_state.selected_bridge else (
        ws_sid, df[df["stable_id"] == ws_sid].iloc[0])
    ws_entry = queue.get(ws_sid, {})
    act_by = max(int(rrow["PROGRAM_YEAR"]), 2027)

    with st.container(border=True):
        st.markdown(f"### {bridge_name(rrow)}")
        if ws_sid in approved_ids:
            st.markdown(
                '<span class="pill pill-likely">Approved for funding plan</span>',
                unsafe_allow_html=True,
            )
        st.markdown("**myBridge suggestion — cannot be edited**")
        st.markdown(
            f'<div class="caption">{rrow["status_title"]} · Suggested next step: '
            f'{rrow["action_plain"]} · Act by {act_by} · {_chance_phrase(rrow)} · '
            f'{clean_county(rrow["county_name"])}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Load rating**")
        sample_path = "data/sample_load_rating.csv"
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as _sf:
                st.download_button(
                    "Download sample load rating (CSV)", _sf.read(),
                    "sample_load_rating.csv", "text/csv", key="er_sample_dl",
                )
        er_up = st.file_uploader(
            "Upload load rating", type=["csv", "pdf", "txt"], key="er_upload")
        rating = {}
        file_sha, file_name, mismatch = "", "", False
        if er_up is not None:
            raw = er_up.getvalue()
            file_sha = hashlib.sha256(raw).hexdigest()
            file_name = er_up.name
            cache_key = f"{ws_sid}|{file_name}|{file_sha}"
            if cache_key not in st.session_state.rating_cache:
                st.session_state.rating_cache[cache_key] = _parse_rating_bytes(
                    raw, file_name)
            rating = dict(st.session_state.rating_cache[cache_key])
            up_bid = str(rating.get("bridge_id") or "").strip()
            mismatch = bool(up_bid) and up_bid != str(ws_sid)
            if mismatch:
                st.warning(
                    "This file names a different bridge than the one under review. "
                    "Check the file, or acknowledge below to attach it anyway."
                )
                st.checkbox(
                    "I understand this file names a different bridge and I want to "
                    "attach it anyway.",
                    key=f"er_ack_{ws_sid}_{file_sha[:8]}",
                )
            editor_df = pd.DataFrame([{
                "Rating date": rating.get("rating_date") or "",
                "Software": rating.get("software") or "",
                "Method": rating.get("method") or "",
                "Controlling member": rating.get("controlling_member") or "",
                "Inventory RF": rating.get("inventory_rf"),
                "Operating RF": rating.get("operating_rf"),
                "Posting recommended": rating.get("posting_recommended") or "",
            }])
            edited = st.data_editor(
                editor_df, key=f"er_editor_{ws_sid}_{file_sha[:8]}",
                hide_index=True, width="stretch",
            )
            erow = edited.iloc[0]
            rating.update({
                "rating_date": erow["Rating date"],
                "software": erow["Software"],
                "method": erow["Method"],
                "controlling_member": erow["Controlling member"],
                "inventory_rf": _rf_value(erow["Inventory RF"]),
                "operating_rf": _rf_value(erow["Operating RF"]),
                "posting_recommended": erow["Posting recommended"],
            })
            st.markdown(
                f'<div class="caption">File: {file_name} · '
                f"Fingerprint (SHA-256): {file_sha[:16]}…</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="caption">Upload a CSV, PDF, or text load rating, '
                "or enter the checks manually below.</div>",
                unsafe_allow_html=True,
            )
        er_inv_manual = st.number_input(
            "Inventory check (RF)", min_value=0.0, value=0.0, step=0.01, key="er_inv_rf")
        er_op_manual = st.number_input(
            "Operating check (RF)", min_value=0.0, value=0.0, step=0.01, key="er_op_rf")
        inv_rf = rating.get("inventory_rf")
        if inv_rf is None and er_inv_manual > 0:
            inv_rf = float(er_inv_manual)
        op_rf = rating.get("operating_rf")
        if op_rf is None and er_op_manual > 0:
            op_rf = float(er_op_manual)

        verified = st.checkbox(
            "I verified these values against the load rating report.",
            key=f"er_verified_{ws_sid}",
        )
        if ws_sid in queue and queue[ws_sid].get("status") in ACTIVE_QUEUE_STATUSES:
            if verified:
                queue[ws_sid]["status"] = "Load rating verified"
            elif queue[ws_sid].get("status") == "Load rating verified":
                queue[ws_sid]["status"] = "Under review"

        if inv_rf is not None and op_rf is not None:
            level, guidance = _capacity_guidance(inv_rf, op_rf)
            st.markdown(f"**Capacity guidance — {level}**")
            st.markdown(f'<div class="caption">{guidance}</div>', unsafe_allow_html=True)

        st.markdown("**Your decision**")
        action_options = [
            "Plan replacement", "Plan rehabilitation", "Preventive work",
            "Monitor — no major work", "Request re-inspection",
        ]
        default_action = {
            "Plan replacement": 0, "Plan rehabilitation": 1,
        }.get(rrow["action_plain"], 2)
        final_action = st.selectbox(
            "Final action", action_options, index=default_action, key="er_final_action")
        final_year = st.number_input(
            "Final Act-by year", min_value=2027, max_value=2060, value=act_by,
            step=1, key="er_final_year")
        decision_choice = st.radio(
            "Decision", ["Approve", "Reject", "Defer"], horizontal=True, key="er_decision")
        notes = st.text_area("Notes", key="er_notes", height=90)
        defer_year = None
        if decision_choice == "Defer":
            defer_year = st.number_input(
                "Review again in year", min_value=2027, max_value=2060, value=2028,
                step=1, key="er_defer_year")
        er_name = st.text_input("Your name", key="er_name")
        er_license = st.text_input("License number", key="er_license")
        attested = st.checkbox(
            "I reviewed the load rating and confirm capacity was considered in "
            "this decision.",
            key="er_attest",
        )
        ack_ok = (not mismatch) or bool(
            st.session_state.get(f"er_ack_{ws_sid}_{file_sha[:8]}"))
        missing = []
        if not er_name.strip():
            missing.append("your name")
        if not er_license.strip():
            missing.append("license number")
        if not attested:
            missing.append("attestation")
        if not (verified or final_action in
                ("Preventive work", "Monitor — no major work")):
            missing.append("a verified load rating (or choose Preventive work / Monitor)")
        if not ack_ok:
            missing.append("acknowledgement that the file names a different bridge")
        if decision_choice == "Reject" and not notes.strip():
            missing.append("a reason in Notes for rejecting")
        if decision_choice == "Defer" and not defer_year:
            missing.append("a review year for deferring")
        if missing:
            st.markdown(
                '<div class="caption">Still needed: ' + "; ".join(missing) + ".</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="caption">Demo attestation only — not a verified PE seal.</div>',
            unsafe_allow_html=True,
        )
        if st.button("Submit decision", type="primary", disabled=bool(missing),
                     key="er_submit"):
            if final_action == "Request re-inspection":
                decision_label = "Re-inspection requested"
            else:
                decision_label = {"Approve": "Approved", "Reject": "Rejected",
                                  "Defer": "Deferred"}[decision_choice]
            record = {
                "decision_id": f"DEC-2026-{len(st.session_state.decisions) + 1:04d}",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "stable_id": ws_sid,
                "bridge_name": bridge_name(rrow),
                "county": clean_county(rrow["county_name"]),
                "model_status": rrow["status_title"],
                "model_suggestion": rrow["action_plain"],
                "chance_at_review": float(rrow["P_POOR_5YR"]),
                "engineer_name": er_name.strip(),
                "license": er_license.strip(),
                "decision": decision_label,
                "final_action": final_action,
                "final_year": int(final_year),
                "review_year": int(defer_year) if defer_year else None,
                "inventory_rf": inv_rf,
                "operating_rf": op_rf,
                "controlling_member": rating.get("controlling_member") or "",
                "posting_recommended": rating.get("posting_recommended") or "",
                "source_file": file_name,
                "file_sha256": file_sha,
                "notes": notes.strip(),
            }
            st.session_state.decisions.append(record)
            if ws_sid in queue:
                queue[ws_sid]["status"] = decision_label
            st.session_state.approved_ids = {
                d["stable_id"] for d in st.session_state.decisions
                if d["decision"] == "Approved"}
            st.session_state.approved_list = sorted(st.session_state.approved_ids)
            st.session_state.last_decision = record
            st.session_state.review_selected = next(
                (s for s, e in queue.items()
                 if e.get("status") in ACTIVE_QUEUE_STATUSES), ws_sid)
            st.rerun()

    decisions = st.session_state.decisions
    approved_records = [d for d in decisions if d["decision"] == "Approved"]
    with st.container(border=True):
        st.markdown("**Approved Funding Plan**")
        if not approved_records:
            st.markdown(
                '<div class="caption">No approved bridges yet this session.</div>',
                unsafe_allow_html=True,
            )
        for d in approved_records:
            st.markdown(
                f'<div class="caption">{d["bridge_name"]} · {d["final_action"]} · '
                f'Act by {d["final_year"]} · {d["engineer_name"]}</div>',
                unsafe_allow_html=True,
            )
        if approved_records:
            prog = pd.DataFrame([{
                "Bridge": d["bridge_name"], "County": d["county"],
                "Final action": d["final_action"], "Act-by year": d["final_year"],
                "Engineer": d["engineer_name"], "Decision ID": d["decision_id"],
                "Date": d["timestamp"][:10],
            } for d in approved_records])
            st.download_button(
                "Download approved program (CSV)", prog.to_csv(index=False).encode(),
                "approved_program.csv", "text/csv", key="dl_approved_program",
            )
    with st.container(border=True):
        st.markdown("**Decision History**")
        if not decisions:
            st.markdown(
                '<div class="caption">No decisions recorded yet this session.</div>',
                unsafe_allow_html=True,
            )
        for d in decisions:
            st.markdown(
                f'<div class="caption">{d["decision_id"]} · {d["bridge_name"]} · '
                f'{d["decision"]} · {d["final_action"]} · Act by {d["final_year"]} · '
                f'{d["engineer_name"]} · {d["timestamp"][:10]}</div>',
                unsafe_allow_html=True,
            )
        if decisions:
            audit = pd.DataFrame([{
                "Decision ID": d["decision_id"], "Date and time": d["timestamp"],
                "Bridge": d["bridge_name"], "County": d["county"],
                "myBridge suggestion": d["model_suggestion"],
                "Chance at review": f"{d['chance_at_review']:.0%}",
                "Engineer": d["engineer_name"], "License": d["license"],
                "Decision": d["decision"], "Final action": d["final_action"],
                "Act-by year": d["final_year"],
                "Review year": d["review_year"] or "",
                "Inventory RF": d["inventory_rf"] if d["inventory_rf"] is not None else "",
                "Operating RF": d["operating_rf"] if d["operating_rf"] is not None else "",
                "Controlling member": d["controlling_member"],
                "Posting recommended": d["posting_recommended"],
                "Source file": d["source_file"], "File SHA-256": d["file_sha256"],
                "Notes": d["notes"],
            } for d in decisions])
            st.download_button(
                "Download decision audit log (CSV)", audit.to_csv(index=False).encode(),
                "decision_audit_log.csv", "text/csv", key="dl_audit_log",
            )
    st.button("Plan & Estimate ->", type="tertiary", on_click=go_to, args=("plan",))

# ================= GUIDE =================
elif page == "guide":
    render_nav([("← Home", "home", "nav_home")])
    st.markdown("## How to use myBridge")
    steps = [
        ("1", "Find", "Search any road, town, river, or county."),
        ("2", "Read Summary", "Read the chance % and the suggested next step."),
        ("3", "Engineer-in-the-Loop", "An engineer verifies load capacity and approves."),
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
