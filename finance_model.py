# finance_model.py - deterministic planning cost estimates for myBridge.
# Pure functions only: no Streamlit, no I/O, no randomness. Every number an
# agent answer shows comes from here, never from a language model.

import math

import pandas as pd

MIN_TARGET_YEAR = 2027

DEFAULT_ASSUMPTIONS = {
    # Construction cost per square foot of deck, by action (planning level).
    "base_rates": {"Replacement": 700.0, "Rehabilitation": 350.0, "Preservation": 75.0},
    "design_pct": 0.10,        # engineering design, share of construction
    "inspection_pct": 0.08,    # construction inspection, share of construction
    "contingency_pct": 0.20,   # planning contingency, share of subtotal
    "escalation_rate": 0.04,   # cost growth per year until the target year
    "base_year": 2026,         # year the base rates are priced in
    "low_factor": 0.85,        # low end of the planning range, share of likely
    "high_factor": 1.30,       # high end of the planning range, share of likely
    "default_deck_area_sqft": 5000.0,
    "regional_multiplier": 1.0,  # regional pricing adjustment
    # Expected service life added by each action, in years (best-value scoring).
    "life_added": {"Replacement": 50.0, "Rehabilitation": 25.0, "Preservation": 10.0},
}

# Design types that make a project more complex to build.
_COMPLEX_DESIGNS = ("truss", "arch", "suspension", "movable", "cable", "segmental")
_WATER_WORDS = ("river", "creek", "stream", "bayou", "slough", "lake", "channel", "water")
_RAIL_WORDS = ("railroad", "railway", " rr ", "rail")


def _merged(assumptions):
    a = dict(DEFAULT_ASSUMPTIONS)
    if assumptions:
        a.update(assumptions)
    return a


def _num(v, default=None):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) else f


def action_for_row(row):
    """Map a bridge's recommended action text to Replacement/Rehabilitation/Preservation."""
    text = str(row.get("RECOMMENDED_ACTION", "") or "")
    if text.startswith("Replacement"):
        return "Replacement"
    if text.startswith("Rehabilitation"):
        return "Rehabilitation"
    return "Preservation"


def _resolve_action(action, row, a):
    if action:
        norm = str(action).strip().lower()
        for name in a["base_rates"]:
            if norm == name.lower():
                return name
        raise ValueError(f"unknown action: {action!r}")
    if row is not None:
        return action_for_row(row)
    return "Rehabilitation"


def _crossing_factor(features):
    t = f" {str(features or '').lower()} "
    if any(w in t for w in _WATER_WORDS):
        return 1.15, "water crossing"
    if any(w in t for w in _RAIL_WORDS):
        return 1.10, "railroad crossing"
    return 1.00, "road crossing"


def _traffic_factor(adt):
    if adt is None:
        return 1.00, "traffic not on record"
    if adt >= 20000:
        return 1.12, "very busy road"
    if adt >= 5000:
        return 1.06, "busy road"
    if adt >= 1000:
        return 1.03, "moderate traffic"
    return 1.00, "quiet road"


def _complexity_factor(design, material):
    d = str(design or "").lower()
    m = str(material or "").lower()
    if any(w in d for w in _COMPLEX_DESIGNS):
        f, label = 1.20, "complex structure"
    elif "box beam" in d or "girder" in d:
        f, label = 1.05, "girder structure"
    else:
        f, label = 1.00, "standard structure"
    if "steel" in m:
        f = round(f * 1.05, 4)
        label += ", steel"
    return f, label


def _size_factor(area):
    if area >= 50000:
        return 0.92, "very large deck"
    if area >= 20000:
        return 0.96, "large deck"
    if area >= 5000:
        return 1.00, "typical size"
    if area >= 1000:
        return 1.05, "small deck"
    return 1.10, "very small deck"


def estimate_cost(row, action=None, target_year=None, assumptions=None):
    """Planning estimate for one bridge. Returns low/likely/high plus every
    component and adjustment used. Target years are clamped to 2027 or later."""
    a = _merged(assumptions)
    act = _resolve_action(action, row, a)
    base_year = int(a["base_year"])
    ty = int(target_year) if target_year else base_year + 1
    ty = max(ty, MIN_TARGET_YEAR, base_year + 1)

    area = _num(row.get("deck_area_sqft")) if row is not None else None
    if not area or area <= 0:
        area = float(a["default_deck_area_sqft"])
    rate = float(a["base_rates"][act])

    cf, cf_label = _crossing_factor(row.get("features_intersected") if row is not None else "")
    adt = _num(row.get("adt")) if row is not None else None
    tf, tf_label = _traffic_factor(adt)
    xf, xf_label = _complexity_factor(
        row.get("design") if row is not None else "",
        row.get("material") if row is not None else "",
    )
    sf, sf_label = _size_factor(area)
    rm = float(a.get("regional_multiplier", 1.0))

    construction = area * rate * cf * tf * xf * sf * rm
    design = construction * float(a["design_pct"])
    inspection = construction * float(a["inspection_pct"])
    subtotal = construction + design + inspection
    contingency = subtotal * float(a["contingency_pct"])
    priced = subtotal + contingency
    esc_years = max(ty - base_year, 0)
    escalated = priced * (1.0 + float(a["escalation_rate"])) ** esc_years

    likely = escalated
    return {
        "action": act,
        "target_year": ty,
        "deck_area_sqft": round(area, 1),
        "base_rate_per_sqft": rate,
        "factors": {
            "crossing": {"value": cf, "label": cf_label},
            "traffic": {"value": tf, "label": tf_label},
            "complexity": {"value": xf, "label": xf_label},
            "size": {"value": sf, "label": sf_label},
            "regional": {"value": rm, "label": "regional pricing"},
        },
        "components": {
            "construction": round(construction),
            "design": round(design),
            "construction_inspection": round(inspection),
            "contingency": round(contingency),
            "escalation": round(escalated - priced),
        },
        "low": round(likely * float(a["low_factor"])),
        "likely": round(likely),
        "high": round(likely * float(a["high_factor"])),
    }


def _ordered_candidates(df, strategy, a):
    cand = df[(df["IS_ALREADY_POOR"] == 1) | (df["P_POOR_5YR"] > 0.3)].copy()
    if cand.empty:
        return cand
    cand["_poor"] = (cand["IS_ALREADY_POOR"] == 1).astype(int)
    if strategy == "worst_first":
        cand = cand.sort_values(["_poor", "P_POOR_5YR"], ascending=[False, False])
    elif strategy == "fix_fair_first":
        mc = pd.to_numeric(cand["min_cond"], errors="coerce")
        cand["_fair"] = ((mc >= 5) & (mc <= 6)).astype(int)
        cand = cand.sort_values(["_fair", "_poor", "P_POOR_5YR"], ascending=[False, False, False])
    elif strategy == "best_value":
        scores = []
        for _, r in cand.iterrows():
            est = estimate_cost(r, assumptions=a)
            urgency = 1.0 if r["IS_ALREADY_POOR"] == 1 else float(r["P_POOR_5YR"])
            adt = _num(r.get("adt"), 0.0) or 0.0
            traffic = min(adt, 50000.0) / 50000.0
            life = float(a["life_added"][est["action"]])
            scores.append((0.7 * urgency + 0.3 * traffic) * life / max(est["likely"], 1.0))
        cand["_score"] = scores
        cand = cand.sort_values("_score", ascending=False)
    else:  # pragma: no cover - guarded by build_plan validation
        raise ValueError(f"unknown strategy: {strategy!r}")
    return cand


def build_plan(df, annual_budget, horizon_years=5, strategy="worst_first", assumptions=None):
    """Greedy funding plan. Funded bridges get a target year inside the
    horizon; whatever does not fit stays deferred and still at risk."""
    a = _merged(assumptions)
    if strategy not in ("worst_first", "best_value", "fix_fair_first"):
        raise ValueError(f"unknown strategy: {strategy!r}")
    budget = _num(annual_budget)
    if not budget or budget <= 0:
        raise ValueError("annual_budget must be positive")
    horizon = int(horizon_years)
    if horizon < 1 or horizon > 10:
        raise ValueError("horizon_years must be between 1 and 10")

    cand = _ordered_candidates(df, strategy, a)
    base_year = int(a["base_year"])
    pending = list(cand.index)
    rows = {i: cand.loc[i] for i in pending}
    plain_action = {"Replacement": "Plan replacement",
                    "Rehabilitation": "Plan rehabilitation",
                    "Preservation": "Preventive work"}
    funded, annual = [], []
    carry = 0.0
    cumulative = 0
    for h in range(1, horizon + 1):
        year = base_year + h
        cash = budget + carry  # unused budget carries forward
        y_low = y_likely = y_high = 0
        by_action = {}
        still = []
        for i in pending:
            est = estimate_cost(rows[i], target_year=year, assumptions=a)
            if est["likely"] <= cash:  # one whole project, earliest year it fits
                funded.append({
                    "stable_id": rows[i]["stable_id"],
                    "action": est["action"],
                    "target_year": year,
                    "low": est["low"],
                    "likely": est["likely"],
                    "high": est["high"],
                })
                cash -= est["likely"]
                y_low += est["low"]
                y_likely += est["likely"]
                y_high += est["high"]
                name = plain_action[est["action"]]
                by_action[name] = by_action.get(name, 0) + 1
            else:
                still.append(i)
        pending = still
        carry = cash
        count = sum(by_action.values())
        cumulative += count
        annual.append({
            "year": year, "budget": budget, "spent": round(y_likely),
            "funded_count": count, "count": count, "by_action": by_action,
            "low": y_low, "likely": y_likely, "high": y_high,
            "cumulative_count": cumulative,
        })
    return {
        "strategy": strategy,
        "annual_budget": budget,
        "horizon_years": horizon,
        "funded": funded,
        "deferred_ids": [rows[i]["stable_id"] for i in pending],
        "annual_costs": annual,
        "total_low": sum(f["low"] for f in funded),
        "total_likely": sum(f["likely"] for f in funded),
        "total_high": sum(f["high"] for f in funded),
        "remaining_at_risk": len(pending),
    }
