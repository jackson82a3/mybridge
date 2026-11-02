# agent_tools.py - Ask myBridge tools and routing.
# Python validates and executes every tool against the bridge dataframe and
# calculates every count and cost. Llama only (a) picks one tool as strict
# JSON and (b) optionally rewords the finished result. Any failure falls back
# to deterministic keyword parsing and a template summary. No API key is ever
# printed or logged here; callers pass a client factory or None.

import json
import re

import pandas as pd

import finance_model as fm

ALLOWED_STATUSES = ("repair_now", "repair_soon", "doing_well")
ALLOWED_STRATEGIES = ("worst_first", "best_value", "fix_fair_first")
ALLOWED_ACTIONS = ("replacement", "rehabilitation", "preservation")
TOOLS = ("search_bridges", "get_network_summary", "estimate_bridge_cost", "build_budget_plan")

STATUS_TITLES = {
    "repair_now": "Needs major work now",
    "repair_soon": "Plan soon",
    "doing_well": "Doing well for now",
}
STRATEGY_WORDS = {
    "worst_first": "worst first",
    "best_value": "best value",
    "fix_fair_first": "fair bridges first",
}
ACTION_WORDS = {
    "replacement": "Replace",
    "rehabilitation": "Rehabilitate",
    "preservation": "Preserve",
}

_WATER = ("river", "creek", "stream", "lake", "channel", "bayou", "slough")


# ---------------- human-facing helpers ----------------

def _clean_text(v):
    return " ".join(str(v if v is not None and not (isinstance(v, float) and pd.isna(v)) else "").split())


def clean_county(name):
    s = _clean_text(name)
    while "County County" in s:
        s = s.replace("County County", "County")
    return s


def _status_title(row):
    if "status_title" in row and _clean_text(row.get("status_title")):
        return _clean_text(row["status_title"])
    if int(row.get("IS_ALREADY_POOR", 0) or 0) == 1:
        return "Needs major work now"
    p = float(row.get("P_POOR_5YR", 0) or 0)
    if p >= 0.7:
        return "Plan soon"
    if p >= 0.3:
        return "Plan ahead"
    return "Doing well for now"


_ROUTE_TOKEN = re.compile(r"(?i)^(us|mn|i|sr|cr|hwy)-?\d+$|^(us|mn|sr|i|cr)$")


def _title_features(text):
    words = []
    for w in str(text).split():
        core = w.strip(".,()")
        words.append(w.upper() if _ROUTE_TOKEN.match(core) else w.capitalize())
    return " ".join(words)


def _human_bridge(row):
    adt = row.get("adt")
    vehicles = None
    if adt is not None and not (isinstance(adt, float) and pd.isna(adt)):
        vehicles = int(round(float(adt) / 100.0) * 100)
    return {
        "stable_id": row["stable_id"],
        "name": f"{_clean_text(row.get('facility_carried'))} over {_title_features(_clean_text(row.get('features_intersected')))}",
        "county": clean_county(row.get("county_name")),
        "status": _status_title(row),
        "vehicles_per_day": vehicles,
    }


def _money(v):
    v = float(v)
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:,.1f}M"
    if abs(v) >= 1_000:
        return f"${v / 1_000:,.0f}K"
    return f"${v:,.0f}"


def _counties(df):
    return sorted({clean_county(c) for c in df["county_name"].dropna().unique() if clean_county(c)})


def _resolve_county(df, county):
    if county in (None, ""):
        return None
    want = _clean_text(county).lower()
    for c in _counties(df):
        if c.lower() == want or c.lower().removesuffix(" county") == want.removesuffix(" county"):
            return c
    raise ValueError(f"unknown county: {county!r}")


def _status_mask(df, status):
    poor = df["IS_ALREADY_POOR"] == 1
    if status == "repair_now":
        return poor
    if status == "repair_soon":
        return (~poor) & (df["P_POOR_5YR"] > 0.3)
    return (~poor) & (df["P_POOR_5YR"] <= 0.3)


# ---------------- the four tools ----------------

def search_bridges(df, county=None, status=None, crossing=None, busy=False, limit=5):
    county = _resolve_county(df, county)
    if status is not None and status not in ALLOWED_STATUSES:
        raise ValueError(f"unknown status: {status!r}")
    f = df
    if county:
        f = f[f["county_name"].map(clean_county) == county]
    if status:
        f = f[_status_mask(f, status)]
    if crossing:
        cw = str(crossing).lower()
        words = _WATER if cw == "river" else (cw,)
        f = f[f["features_intersected"].astype(str).str.lower().str.contains("|".join(words), regex=True)]
    f = f.sort_values("adt" if busy else "P_POOR_5YR", ascending=False)
    bridges = [_human_bridge(r) for _, r in f.head(int(limit)).iterrows()]
    return {"kind": "search_bridges", "count": int(len(f)), "bridges": bridges,
            "county": county, "status": status}


def get_network_summary(df, county=None):
    county = _resolve_county(df, county)
    f = df[f["county_name"].map(clean_county) == county] if county else df
    return {
        "kind": "get_network_summary",
        "scope": county or "All counties",
        "total": int(len(f)),
        "needs_major_work_now": int((f["IS_ALREADY_POOR"] == 1).sum()),
        "plan_soon": int(_status_mask(f, "repair_soon").sum()),
        "doing_well": int(_status_mask(f, "doing_well").sum()),
    }


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _find_bridge(df, query):
    q = str(query or "")
    m = re.search(r"(.+?)\s+over\s+(.+)", q, flags=re.I)
    fac_q, feat_q = (m.group(1), m.group(2)) if m else (q, "")
    feat_q = re.sub(r"^(the|a|an)\s+", "", feat_q.strip(), flags=re.I).rstrip("?. ")
    fac = df["facility_carried"].astype(str).map(_norm)
    feat = df["features_intersected"].astype(str).map(_norm)
    words = fac_q.split()
    tries = [fac_q] + [" ".join(words[-n:]) for n in (3, 2, 1) if len(words) >= n]
    hits = df.iloc[0:0]
    for cand in tries:
        cn = _norm(cand)
        if not cn:
            continue
        h = df[fac.str.contains(cn, regex=False)]
        if not h.empty:
            hits = h
            break
    if hits.empty:
        hits = df
    if feat_q:
        fn = _norm(feat_q)[:14]
        h2 = hits[feat.loc[hits.index].str.contains(fn, regex=False)]
        if h2.empty:
            h2 = df[feat.str.contains(fn, regex=False)]
        if not h2.empty:
            hits = h2
    if hits.empty:
        raise ValueError("no bridge matched that name")
    return hits.sort_values("P_POOR_5YR", ascending=False).iloc[0]


def estimate_bridge_cost(df, query, action=None, target_year=None):
    if action is not None and str(action).lower() not in ALLOWED_ACTIONS:
        raise ValueError(f"unknown action: {action!r}")
    row = _find_bridge(df, query)
    act = None
    if action:
        act = {"replacement": "Replacement", "rehabilitation": "Rehabilitation",
               "preservation": "Preservation"}[str(action).lower()]
    est = fm.estimate_cost(row, action=act, target_year=target_year)
    bridge = _human_bridge(row)
    return {"kind": "estimate_bridge_cost", "bridge": bridge,
            "action": est["action"], "target_year": est["target_year"],
            "low": est["low"], "likely": est["likely"], "high": est["high"]}


def build_budget_plan(df, annual_budget, horizon_years=5, strategy="worst_first"):
    if strategy not in ALLOWED_STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy!r}")
    plan = fm.build_plan(df, annual_budget, horizon_years, strategy)
    by_id = {r["stable_id"]: r for _, r in df.iterrows()}
    top = []
    for f in plan["funded"][:5]:
        b = _human_bridge(by_id[f["stable_id"]])
        b["target_year"] = f["target_year"]
        top.append(b)
    return {
        "kind": "build_budget_plan",
        "annual_budget": plan["annual_budget"],
        "horizon_years": plan["horizon_years"],
        "strategy": plan["strategy"],
        "funded_count": len(plan["funded"]),
        "deferred_count": len(plan["deferred_ids"]),
        "total_low": plan["total_low"],
        "total_likely": plan["total_likely"],
        "total_high": plan["total_high"],
        "annual_costs": plan["annual_costs"],
        "top_funded": top,
    }


def _execute(df, call):
    tool, p = call["tool"], call.get("params", {})
    if tool == "search_bridges":
        return search_bridges(df, county=p.get("county"), status=p.get("status"),
                              crossing=p.get("crossing"), busy=bool(p.get("busy", False)))
    if tool == "get_network_summary":
        return get_network_summary(df, county=p.get("county"))
    if tool == "estimate_bridge_cost":
        return estimate_bridge_cost(df, p.get("query", ""), action=p.get("action"),
                                    target_year=p.get("target_year"))
    return build_budget_plan(df, p.get("annual_budget"), p.get("horizon_years", 5),
                             p.get("strategy", "worst_first"))


# ---------------- routing: Llama JSON, validated, else keywords ----------------

_ROUTER_SYSTEM = (
    "You route questions for myBridge. Reply with ONLY one JSON object, no other text. "
    'Shape: {"tool": one of ["search_bridges","get_network_summary","estimate_bridge_cost","build_budget_plan"], "params": {...}}. '
    "search_bridges params: county (string or null), status (repair_now|repair_soon|doing_well|null), "
    "crossing (river|null), busy (true|false). "
    "get_network_summary params: county (string or null). "
    "estimate_bridge_cost params: query (bridge name like 'MN 243 over St. Croix River'), "
    "action (replacement|rehabilitation|preservation|null), target_year (number or null). "
    "build_budget_plan params: annual_budget (dollars per year, number), horizon_years (1-10), "
    "strategy (worst_first|best_value|fix_fair_first). "
    "Never invent counties, bridges, or numbers."
)


def _validate_call(raw, df):
    if not isinstance(raw, dict) or raw.get("tool") not in TOOLS:
        raise ValueError("bad tool call")
    tool = raw["tool"]
    p = raw.get("params") or {}
    if not isinstance(p, dict):
        raise ValueError("bad params")
    out = {}
    if tool in ("search_bridges", "get_network_summary"):
        out["county"] = _resolve_county(df, p.get("county"))
    if tool == "search_bridges":
        st = p.get("status")
        if st is not None and st not in ALLOWED_STATUSES:
            raise ValueError("bad status")
        out["status"] = st
        cr = p.get("crossing")
        out["crossing"] = "river" if str(cr).lower() in ("river", "water") else None
        out["busy"] = bool(p.get("busy", False))
    elif tool == "estimate_bridge_cost":
        q = _clean_text(p.get("query"))
        if not q or len(q) > 140:
            raise ValueError("bad query")
        out["query"] = q
        act = p.get("action")
        if act is not None and str(act).lower() not in ALLOWED_ACTIONS:
            raise ValueError("bad action")
        out["action"] = str(act).lower() if act else None
        ty = p.get("target_year")
        out["target_year"] = int(ty) if isinstance(ty, (int, float)) and 2027 <= int(ty) <= 2060 else None
    elif tool == "build_budget_plan":
        b = p.get("annual_budget")
        if not isinstance(b, (int, float)) or not (100_000 <= float(b) <= 500_000_000):
            raise ValueError("bad budget")
        out["annual_budget"] = float(b)
        h = p.get("horizon_years", 5)
        out["horizon_years"] = int(h) if isinstance(h, (int, float)) and 1 <= int(h) <= 10 else 5
        s = p.get("strategy", "worst_first")
        if s not in ALLOWED_STRATEGIES:
            raise ValueError("bad strategy")
        out["strategy"] = s
    return {"tool": tool, "params": out}


def _llama_pick(question, df, client_factory, model):
    client = client_factory()
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=300,
        messages=[
            {"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user", "content": "Counties include: " + ", ".join(_counties(df)[:40])
             + "\n\nQuestion: " + question},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON in router reply")
    return _validate_call(json.loads(m.group(0)), df)


def _keyword_parse(question, df):
    q = " " + str(question).lower() + " "
    county = None
    for c in _counties(df):
        if c.lower().removesuffix(" county") in q or c.lower() in q:
            county = c
            break
    money = re.search(r"\$\s*(\d+(?:\.\d+)?)\s*(m|mm|million|k|thousand)?", q)
    if money or ("budget" in q and ("fix" in q or "plan" in q or "year" in q)):
        amount = None
        if money:
            v = float(money.group(1))
            unit = money.group(2)
            amount = v * (1_000_000 if unit in ("m", "mm", "million") else 1_000 if unit in ("k", "thousand") else 1)
            if amount < 100_000:  # "$40 a year" style input means $40M in this app
                amount *= 1_000_000
        strategy = "best_value" if "best value" in q else "fix_fair_first" if "fair" in q else "worst_first"
        return {"tool": "build_budget_plan",
                "params": {"annual_budget": amount or 50_000_000, "horizon_years": 5, "strategy": strategy}}
    if ("how much" in q or "cost" in q or "estimate" in q or "replace" in q or "rehabilitat" in q) and " over " in q:
        action = "replacement" if "replac" in q else "rehabilitation" if "rehabilitat" in q else None
        return {"tool": "estimate_bridge_cost", "params": {"query": str(question), "action": action}}
    if "how many" in q or "summary" in q or "overview of" in q:
        return {"tool": "get_network_summary", "params": {"county": county}}
    status = None
    if "major work now" in q or "repair now" in q or "needing work" in q or "need work" in q or "needs work" in q or "major repair" in q:
        status = "repair_now"
    elif "repair soon" in q or "plan soon" in q:
        status = "repair_soon"
    elif "doing well" in q or "good shape" in q:
        status = "doing_well"
    crossing = "river" if any(w in q for w in (" river", " creek", " stream", " water")) else None
    busy = "busy" in q or "busiest" in q or "most traffic" in q
    return {"tool": "search_bridges",
            "params": {"county": county, "status": status, "crossing": crossing, "busy": busy}}


# ---------------- understood text, template summary, Llama wording ----------------

def _understood(call):
    tool, p = call["tool"], call["params"]
    if tool == "search_bridges":
        bits = []
        if p.get("status"):
            bits.append(STATUS_TITLES[p["status"]] + " bridges")
        else:
            bits.append("Bridges")
        if p.get("county"):
            bits.append("in " + p["county"])
        if p.get("crossing"):
            bits.append("over a river")
        if p.get("busy"):
            bits.append("· busiest first")
        return " ".join(bits)
    if tool == "get_network_summary":
        return "Bridge summary" + ((" for " + p["county"]) if p.get("county") else " for all counties")
    if tool == "estimate_bridge_cost":
        act = ACTION_WORDS.get(p.get("action") or "", "Work on")
        return f"Cost to {act.lower()} the bridge in: “{p['query']}”"
    return (f"A {_money(p['annual_budget'])}-a-year plan over {p['horizon_years']} years "
            f"({STRATEGY_WORDS[p['strategy']]})")


def _template_summary(result):
    kind = result["kind"]
    if kind == "search_bridges":
        n, bridges = result["count"], result["bridges"]
        if n == 0:
            return "No bridges match that. Try another county, place, or status."
        first = bridges[0]
        return (f"{n} bridge{'s' if n != 1 else ''} match. First up is {first['name']} in "
                f"{first['county']} — {first['status'].lower()}. "
                "Open one below to see it on the map.")
    if kind == "get_network_summary":
        return (f"Of {result['total']} bridges in {result['scope']}, "
                f"{result['needs_major_work_now']} need major work now, "
                f"{result['plan_soon']} should be planned soon, and "
                f"{result['doing_well']} are doing well for now.")
    if kind == "estimate_bridge_cost":
        b = result["bridge"]
        return (f"{result['action']} of {b['name']} in {b['county']} could cost about "
                f"{_money(result['likely'])} (roughly {_money(result['low'])} to {_money(result['high'])}) "
                f"in {result['target_year']} dollars. This is an early planning estimate, "
                "not a bid. An engineer confirms the real scope and cost.")
    top = result["top_funded"][0] if result["top_funded"] else None
    lead = (f" With {_money(result['annual_budget'])} a year for {result['horizon_years']} years, "
            f"{result['funded_count']} bridges could get started")
    if top:
        lead += f", beginning with {top['name']} in {top['county']} ({top['target_year']})"
    lead += (f". About {result['deferred_count']} bridges would stay on the watch list. "
             f"Early planning range: {_money(result['total_low'])} to {_money(result['total_high'])} "
             f"in total. Open the plan below to see it in Plan & Estimate.")
    return lead.strip()


def _llama_wording(question, result, client_factory, model):
    client = client_factory()
    payload = {k: v for k, v in result.items() if k != "kind"}
    resp = client.chat.completions.create(
        model=model, temperature=0.25, max_tokens=180,
        messages=[
            {"role": "system",
             "content": "You are myBridge, an asset-management assistant. Using ONLY the result "
                        "JSON, write 2-3 concise, professional sentences for a bridge program "
                        "manager. Do not invent numbers, costs, causes, or dates. Do not use "
                        "casual wording. Never expose raw field names. Poor means needs major "
                        "repair, not unsafe to cross."},
            {"role": "user", "content": "Question: " + question + "\nResult JSON:\n"
             + json.dumps(payload, default=str)},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("empty wording")
    return text


def answer_question(question, df, client_factory=None, model=None):
    """Full Ask pipeline. Never raises; never returns raw field names or errors."""
    q = _clean_text(question)
    call = None
    if client_factory is not None and model:
        try:
            call = _llama_pick(q, df, client_factory, model)
        except Exception:
            call = None
    if call is None:
        call = _keyword_parse(q, df)
    try:
        result = _execute(df, call)
    except Exception:
        call = {"tool": "get_network_summary", "params": {"county": None}}
        result = _execute(df, call)
    answer = None
    if client_factory is not None and model:
        try:
            answer = _llama_wording(q, result, client_factory, model)
        except Exception:
            answer = None
    if not answer:
        answer = _template_summary(result)
    bridges = result.get("bridges") or result.get("top_funded") or (
        [result["bridge"]] if result.get("bridge") else [])
    understood = _understood(call)
    if result["kind"] == "estimate_bridge_cost":
        verb = {"Replacement": "replace", "Rehabilitation": "rehabilitate",
                "Preservation": "preserve"}[result["action"]]
        understood = f"Cost to {verb} {result['bridge']['name']}"
    return {
        "question": q,
        "understood": understood,
        "answer": answer,
        "kind": result["kind"],
        "bridges": bridges,
        "plan": result if result["kind"] == "build_budget_plan" else None,
        "cost": result if result["kind"] == "estimate_bridge_cost" else None,
    }
