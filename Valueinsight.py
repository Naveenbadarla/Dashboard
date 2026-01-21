# app.py
# Streamlit "Flex Value Insights" — turns valuation runs into segments + insights + roadmap
# Copy-paste, then run:  streamlit run app.py

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Flex Value Insights", page_icon="⚡", layout="wide")

# =========================================================
# 0) Expected schema (flexible, missing columns handled)
# =========================================================
RECOMMENDED_COLUMNS = [
    # Identity / context
    "asset_type",          # "EV", "PV_BATTERY", "HEAT_PUMP"
    "country",             # e.g., "DE"
    "dso",                 # e.g., "Westnetz"
    "price_regime",        # e.g., "2024_hist", "2025_hist", "stress_high_vol"
    "market_stack",        # e.g., "DA+ID", "DA+ID+FCR", ...

    # Behavior / flexibility features (EV-centric, OK if NA for others)
    "sessions_per_week",
    "kwh_per_session",
    "avg_window_hours",    # availability window length (plug-in to plug-out)
    "override_rate",       # 0..1
    "plugin_hour_mean",    # 0..23 (optional)
    "plugout_hour_mean",   # 0..23 (optional)
    "min_soc",             # 0..1 (optional)
    "target_soc_departure",# 0..1 (optional)

    # Asset technical (optional)
    "ev_max_kw",
    "ev_battery_kwh",
    "pv_kwp",
    "battery_kwh",
    "hp_thermal_storage_kwhth",
    "comfort_band_c",

    # Outputs
    "value_total_eur_y",
    "value_da_eur_y",
    "value_id_eur_y",
    "value_as_eur_y",      # ancillary services aggregated (FCR/aFRR)
    "reliability_score",   # 0..1 (if you have it); else derived proxy
    "customer_impact_score" # 0..1 higher = more impact (if you have it); else derived proxy
]

# =========================================================
# 1) Utilities
# =========================================================
def _clamp01(x):
    try:
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return np.nan

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize column names a bit
    df.columns = [c.strip() for c in df.columns]

    # Minimal required
    if "asset_type" not in df.columns:
        df["asset_type"] = "EV"  # default guess

    # Derivations
    if "kwh_per_week" not in df.columns:
        if "sessions_per_week" in df.columns and "kwh_per_session" in df.columns:
            df["kwh_per_week"] = pd.to_numeric(df["sessions_per_week"], errors="coerce") * pd.to_numeric(df["kwh_per_session"], errors="coerce")
        else:
            df["kwh_per_week"] = np.nan

    # Default breakdown if not present
    if "value_total_eur_y" not in df.columns:
        # Try sum breakdown
        parts = []
        for c in ["value_da_eur_y", "value_id_eur_y", "value_as_eur_y"]:
            if c in df.columns:
                parts.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0))
        if parts:
            df["value_total_eur_y"] = sum(parts)
        else:
            df["value_total_eur_y"] = np.nan

    for c in ["value_da_eur_y", "value_id_eur_y", "value_as_eur_y"]:
        if c not in df.columns:
            df[c] = np.nan

    # Convert numeric columns
    numeric_cols = [
        "sessions_per_week","kwh_per_session","avg_window_hours","override_rate",
        "plugin_hour_mean","plugout_hour_mean","min_soc","target_soc_departure",
        "ev_max_kw","ev_battery_kwh","pv_kwp","battery_kwh",
        "hp_thermal_storage_kwhth","comfort_band_c",
        "value_total_eur_y","value_da_eur_y","value_id_eur_y","value_as_eur_y",
        "reliability_score","customer_impact_score","kwh_per_week"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clean override
    if "override_rate" in df.columns:
        df["override_rate"] = df["override_rate"].apply(lambda x: _clamp01(x) if pd.notnull(x) else np.nan)

    # Reliability proxy if missing
    if "reliability_score" not in df.columns or df["reliability_score"].isna().all():
        # Simple proxy: long window helps, high overrides hurt
        win = df.get("avg_window_hours", pd.Series(np.nan, index=df.index))
        ov  = df.get("override_rate", pd.Series(np.nan, index=df.index)).fillna(0.0)
        # map window 2..14h -> 0..1
        rel = (win - 2.0) / (14.0 - 2.0)
        rel = rel.clip(lower=0.0, upper=1.0).fillna(0.5)
        rel = rel * (1.0 - ov.clip(0,1))
        df["reliability_score"] = rel.clip(0,1)

    # Customer impact proxy if missing (higher = more impact)
    if "customer_impact_score" not in df.columns or df["customer_impact_score"].isna().all():
        # Impact rises with override rate and shorter windows (because scheduling becomes more intrusive)
        win = df.get("avg_window_hours", pd.Series(np.nan, index=df.index)).fillna(8.0)
        ov  = df.get("override_rate", pd.Series(np.nan, index=df.index)).fillna(0.0)
        short_win = (8.0 - win) / 8.0  # window >8 reduces
        impact = (0.6 * ov + 0.4 * short_win).clip(0,1)
        df["customer_impact_score"] = impact

    # Fill context defaults
    for c, default in [("country","DE"),("dso","(unknown)"),("price_regime","(unknown)"),("market_stack","DA+ID")]:
        if c not in df.columns:
            df[c] = default
        df[c] = df[c].fillna(default).astype(str)

    df["asset_type"] = df["asset_type"].astype(str).str.upper().str.replace(" ", "_")

    return df

def generate_sample(n=2500, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    asset = rng.choice(["EV","PV_BATTERY","HEAT_PUMP"], size=n, p=[0.55,0.30,0.15])
    country = np.array(["DE"])*n
    dso = rng.choice(["Westnetz","Avacon","Bayernwerk","Netze BW","(unknown)"], size=n, p=[0.22,0.18,0.18,0.18,0.24])
    price_regime = rng.choice(["2024_hist","2025_hist","stress_high_vol"], size=n, p=[0.45,0.45,0.10])
    market_stack = rng.choice(["DA+ID","DA+ID+FCR","DA+ID+aFRR","DA+ID+FCR+aFRR"], size=n, p=[0.55,0.15,0.15,0.15])

    sessions = rng.integers(1, 8, size=n)
    kwh_sess = np.round(rng.uniform(6, 40, size=n), 1)
    window = np.round(rng.uniform(2.0, 14.0, size=n), 1)
    override = np.round(rng.beta(2, 8, size=n), 2)  # mostly low
    plugin_h = np.round(rng.normal(18, 2.5, size=n), 1).clip(0,23)
    plugout_h = (plugin_h + window).clip(0,23)

    # Tech
    ev_max_kw = rng.choice([3.7, 7.4, 11.0], size=n, p=[0.25,0.45,0.30])
    ev_batt = rng.choice([40, 55, 75], size=n, p=[0.35,0.40,0.25])
    pv_kwp = np.round(rng.uniform(3, 12, size=n), 1)
    batt_kwh = np.round(rng.uniform(5, 15, size=n), 1)
    hp_store = np.round(rng.uniform(0, 30, size=n), 1)
    comfort_band = np.round(rng.uniform(1.0, 2.5, size=n), 1)

    # Base value signal (very simplified, just for demo)
    kwh_week = sessions * kwh_sess
    rel = ((window - 2)/(14-2)).clip(0,1) * (1 - override)
    rel = np.clip(rel, 0, 1)

    # asset multipliers
    asset_mult = np.where(asset=="EV", 1.0, np.where(asset=="PV_BATTERY", 1.15, 0.85))
    stack_mult = np.vectorize({"DA+ID":1.0,"DA+ID+FCR":1.15,"DA+ID+aFRR":1.12,"DA+ID+FCR+aFRR":1.25}.get)(market_stack)
    regime_mult = np.vectorize({"2024_hist":1.0,"2025_hist":0.95,"stress_high_vol":1.18}.get)(price_regime)

    # value in €/year
    value_total = (0.045 * kwh_week * 52) * asset_mult * stack_mult * regime_mult * (0.5 + 0.5*rel)
    # add noise
    value_total = np.maximum(0, value_total + rng.normal(0, 25, size=n))

    # breakdown (rough)
    da_share = np.where(np.isin(market_stack, ["DA+ID","DA+ID+FCR","DA+ID+aFRR","DA+ID+FCR+aFRR"]), 0.55, 0.6)
    id_share = 0.30
    as_share = 1 - da_share - id_share
    value_da = value_total * da_share
    value_id = value_total * id_share
    value_as = value_total * as_share

    df = pd.DataFrame({
        "asset_type": asset,
        "country": country,
        "dso": dso,
        "price_regime": price_regime,
        "market_stack": market_stack,
        "sessions_per_week": sessions,
        "kwh_per_session": kwh_sess,
        "avg_window_hours": window,
        "override_rate": override,
        "plugin_hour_mean": plugin_h,
        "plugout_hour_mean": plugout_h,
        "ev_max_kw": ev_max_kw,
        "ev_battery_kwh": ev_batt,
        "pv_kwp": pv_kwp,
        "battery_kwh": batt_kwh,
        "hp_thermal_storage_kwhth": hp_store,
        "comfort_band_c": comfort_band,
        "value_total_eur_y": value_total,
        "value_da_eur_y": value_da,
        "value_id_eur_y": value_id,
        "value_as_eur_y": value_as,
    })
    return ensure_columns(df)

# =========================================================
# 2) Segmentation rules (behavior-first)
# =========================================================
def segment_ev(row) -> str:
    s = row.get("sessions_per_week", np.nan)
    k = row.get("kwh_per_session", np.nan)
    w = row.get("avg_window_hours", np.nan)
    o = row.get("override_rate", 0.0)

    if pd.isna(s) or pd.isna(k) or pd.isna(w):
        return "EV_UNCLASSIFIED"

    kwh_week = s * k

    if o >= 0.30:
        return "EV_HIGH_OVERRIDE"

    # A: Weekend charger (2–3 sessions/week, high kWh, long window)
    if 2 <= s <= 3 and k >= 18 and w >= 8:
        return "EV_WEEKEND_CHARGER"

    # B: Daily commuter (5–7 sessions/week, low/med kWh, consistent overnight-ish)
    if s >= 5 and k <= 18 and w >= 7:
        return "EV_DAILY_COMMUTER"

    # C: High-energy daily (>=5 sessions/week and high kWh)
    if s >= 5 and k > 18:
        return "EV_HIGH_ENERGY_DAILY"

    # D: Opportunistic / short-window
    if w < 5 or s <= 2:
        return "EV_OPPORTUNISTIC_SHORT_WINDOW"

    return "EV_STANDARD"

def segment_pv_battery(row) -> str:
    pv = row.get("pv_kwp", np.nan)
    b = row.get("battery_kwh", np.nan)
    o = row.get("override_rate", 0.0)  # if NA, treated as 0 elsewhere

    if pd.isna(pv) or pd.isna(b):
        return "PVB_UNCLASSIFIED"

    if o >= 0.30:
        return "PVB_HIGH_OVERRIDE"

    # rough sizing heuristics
    if pv >= 8 and b >= 10:
        return "PVB_LARGE_SYSTEM"
    if pv >= 6 and b < 10:
        return "PVB_EXPORT_LEANING"
    if pv < 6 and b >= 10:
        return "PVB_STORAGE_HEAVY"
    return "PVB_MID_SYSTEM"

def segment_heat_pump(row) -> str:
    store = row.get("hp_thermal_storage_kwhth", np.nan)
    band = row.get("comfort_band_c", np.nan)
    o = row.get("override_rate", 0.0)

    if pd.isna(store) and pd.isna(band):
        return "HP_UNCLASSIFIED"

    if o >= 0.30:
        return "HP_HIGH_OVERRIDE"

    # inertia proxy: higher storage + wider band -> more flexible
    store = 0.0 if pd.isna(store) else store
    band = 1.5 if pd.isna(band) else band
    inertia = (store / 20.0) + (band / 2.0)  # 0..~2

    if inertia >= 1.3:
        return "HP_HIGH_INERTIA"
    if inertia >= 0.8:
        return "HP_MED_INERTIA"
    return "HP_LOW_INERTIA"

def apply_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    seg = []
    for _, r in df.iterrows():
        a = str(r.get("asset_type","")).upper()
        if a == "EV":
            seg.append(segment_ev(r))
        elif a in ["PV_BATTERY","PVB","PV&BATTERY","PV_BAT"]:
            seg.append(segment_pv_battery(r))
        elif a in ["HEAT_PUMP","HP"]:
            seg.append(segment_heat_pump(r))
        else:
            seg.append("UNCLASSIFIED")
    df["segment"] = seg
    return df

# =========================================================
# 3) Insight computations
# =========================================================
def segment_signature(df: pd.DataFrame) -> pd.DataFrame:
    # p10/p50/p90 + composition + reliability/impact
    def q(x, p): return np.nanpercentile(x, p) if np.isfinite(x).any() else np.nan

    g = df.groupby(["asset_type","segment"], dropna=False)
    out = g.agg(
        n=("value_total_eur_y","size"),
        value_p10=("value_total_eur_y", lambda x: q(x.values, 10)),
        value_p50=("value_total_eur_y", lambda x: q(x.values, 50)),
        value_p90=("value_total_eur_y", lambda x: q(x.values, 90)),
        rel_mean=("reliability_score","mean"),
        impact_mean=("customer_impact_score","mean"),
        da_mean=("value_da_eur_y","mean"),
        id_mean=("value_id_eur_y","mean"),
        as_mean=("value_as_eur_y","mean"),
        total_mean=("value_total_eur_y","mean"),
        overrides=("override_rate","mean"),
        window=("avg_window_hours","mean"),
        kwh_week=("kwh_per_week","mean"),
        sessions=("sessions_per_week","mean"),
        kwh_sess=("kwh_per_session","mean"),
    ).reset_index()

    # shares
    denom = out["da_mean"].fillna(0)+out["id_mean"].fillna(0)+out["as_mean"].fillna(0)
    denom = denom.replace(0, np.nan)
    out["da_share"] = (out["da_mean"]/denom).fillna(0)
    out["id_share"] = (out["id_mean"]/denom).fillna(0)
    out["as_share"] = (out["as_mean"]/denom).fillna(0)

    # simple attractiveness & readiness scores (edit weights in UI later)
    # attractiveness: value (p50) * reliability - impact penalty; scaled
    out["attractiveness"] = (
        (out["value_p50"].fillna(out["total_mean"]) * (0.5 + 0.5*out["rel_mean"].fillna(0.5)))
        * (1.0 - 0.5*out["impact_mean"].fillna(0.3))
    )
    # readiness proxy: low overrides + good window + (optional) market stack simplicity
    out["readiness"] = (
        (1.0 - out["overrides"].fillna(0.1)).clip(0,1) * (out["window"].fillna(8)/12.0).clip(0,1)
    ) * 100

    # normalize attractiveness to 0..100 for plotting
    a = out["attractiveness"].replace([np.inf,-np.inf], np.nan).fillna(0)
    if a.max() > 0:
        out["attractiveness"] = (a / a.max()) * 100
    else:
        out["attractiveness"] = 0.0

    return out

def propose_fit(row) -> str:
    # quick proposition recommendation from signature
    v = row.get("value_p50", 0)
    rel = row.get("rel_mean", 0.5)
    imp = row.get("impact_mean", 0.3)
    ov = row.get("overrides", 0.1)
    as_share = row.get("as_share", 0.0)

    if ov >= 0.30:
        return "Behavior-shaping offer (nudges + guardrails) + small fixed bonus"
    if rel >= 0.70 and imp <= 0.35 and v >= 120:
        return "Simple fixed bonus (scale fast) or Hybrid (base + top-up)"
    if v >= 120 and (rel < 0.70 or imp > 0.35):
        return "Hybrid (base + performance top-up) to manage volatility/impact"
    if as_share >= 0.25 and rel >= 0.70:
        return "Hybrid + optional AS participation (reliability-based eligibility)"
    return "Performance-based (ct/kWh shifted) or tiered bonus"

# =========================================================
# 4) Sidebar: data load + global filters
# =========================================================
st.sidebar.title("⚡ Flex Value Insights")

with st.sidebar.expander("1) Load valuation results", expanded=True):
    upload = st.file_uploader("Upload CSV (one row = one run)", type=["csv"])
    use_sample = st.checkbox("Use sample data (if you don't have a file yet)", value=(upload is None))

    if upload is not None and not use_sample:
        df_raw = pd.read_csv(upload)
        st.success(f"Loaded {len(df_raw):,} rows")
    else:
        df_raw = generate_sample(n=2500, seed=7)
        st.info("Using built-in sample data (replace with your model export CSV).")

df = ensure_columns(df_raw)
df = apply_segmentation(df)

with st.sidebar.expander("2) Filters", expanded=False):
    asset_filter = st.multiselect("Asset type", sorted(df["asset_type"].unique().tolist()), default=sorted(df["asset_type"].unique().tolist()))
    country_filter = st.multiselect("Country", sorted(df["country"].unique().tolist()), default=sorted(df["country"].unique().tolist()))
    dso_filter = st.multiselect("DSO", sorted(df["dso"].unique().tolist()), default=sorted(df["dso"].unique().tolist()))
    stack_filter = st.multiselect("Market stack", sorted(df["market_stack"].unique().tolist()), default=sorted(df["market_stack"].unique().tolist()))
    regime_filter = st.multiselect("Price regime", sorted(df["price_regime"].unique().tolist()), default=sorted(df["price_regime"].unique().tolist()))

dff = df[
    df["asset_type"].isin(asset_filter)
    & df["country"].isin(country_filter)
    & df["dso"].isin(dso_filter)
    & df["market_stack"].isin(stack_filter)
    & df["price_regime"].isin(regime_filter)
].copy()

sig = segment_signature(dff)
sig["prop_fit"] = sig.apply(propose_fit, axis=1)

# =========================================================
# 5) Navigation
# =========================================================
page = st.sidebar.radio(
    "Navigate",
    ["Executive Overview", "Segment Explorer", "Segment Value Signatures", "Roadmap Prioritization", "Data & Export"],
    index=0
)

# =========================================================
# 6) Pages
# =========================================================
def kpi_row():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Runs", f"{len(dff):,}")
    with c2:
        st.metric("Segments", f"{sig['segment'].nunique():,}")
    with c3:
        st.metric("Median value (€/asset/y)", f"{np.nanmedian(dff['value_total_eur_y']):.1f}")
    with c4:
        st.metric("Avg reliability", f"{np.nanmean(dff['reliability_score']):.2f}")

if page == "Executive Overview":
    st.title("Executive Overview")
    kpi_row()

    st.subheader("Which segments are big and valuable?")
    # Bubble: size vs value with reliability color
    bubble = sig.copy()
    bubble["size"] = bubble["n"]
    bubble["value"] = bubble["value_p50"].fillna(bubble["total_mean"])
    fig = px.scatter(
        bubble,
        x="size",
        y="value",
        color="rel_mean",
        size="size",
        hover_data=["asset_type","segment","value_p10","value_p50","value_p90","impact_mean","prop_fit"],
        facet_col="asset_type",
        title="Segment size vs median value (color = reliability)"
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Value composition by segment (DA / ID / AS)")
    # stacked bars: mean by segment
    topN = st.slider("Show top N segments by median value", 5, 30, 12)
    top = sig.sort_values("value_p50", ascending=False).head(topN).copy()
    top["label"] = top["asset_type"] + " | " + top["segment"]

    comp = top[["label","da_share","id_share","as_share"]].melt(id_vars=["label"], var_name="channel", value_name="share")
    fig2 = px.bar(comp, x="label", y="share", color="channel", barmode="stack", title="Average value share by channel")
    fig2.update_layout(height=420, xaxis_title="", yaxis_title="Share")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("What’s in it for them (customer impact vs value)")
    fig3 = px.scatter(
        sig,
        x="impact_mean",
        y="value_p50",
        color="asset_type",
        size="n",
        hover_data=["segment","rel_mean","prop_fit"],
        title="Median value vs customer impact (bigger = more runs)"
    )
    fig3.update_layout(height=420, xaxis_title="Customer impact (proxy)", yaxis_title="Median value (€/asset/year)")
    st.plotly_chart(fig3, use_container_width=True)

elif page == "Segment Explorer":
    st.title("Segment Explorer")
    kpi_row()

    left, right = st.columns([1.1, 1.9])
    with left:
        st.subheader("Pick a segment")
        asset_pick = st.selectbox("Asset", sorted(sig["asset_type"].unique()))
        seg_pick = st.selectbox("Segment", sorted(sig[sig["asset_type"]==asset_pick]["segment"].unique()))
        row = sig[(sig["asset_type"]==asset_pick) & (sig["segment"]==seg_pick)].iloc[0]

        st.markdown("### Segment snapshot")
        st.metric("Median value (€/y)", f"{row['value_p50']:.1f}")
        st.metric("Reliability", f"{row['rel_mean']:.2f}")
        st.metric("Customer impact", f"{row['impact_mean']:.2f}")
        st.metric("Runs in filter", f"{int(row['n']):,}")

        st.markdown("### Recommended proposition fit")
        st.write(row["prop_fit"])

        st.markdown("### Behavior averages")
        st.write(pd.DataFrame({
            "metric": ["sessions/week","kWh/session","kWh/week","window (h)","override rate"],
            "value": [
                row.get("sessions", np.nan),
                row.get("kwh_sess", np.nan),
                row.get("kwh_week", np.nan),
                row.get("window", np.nan),
                row.get("overrides", np.nan),
            ]
        }))

    with right:
        st.subheader("Distribution: value (€/asset/year)")
        seg_df = dff[(dff["asset_type"]==asset_pick) & (dff["segment"]==seg_pick)].copy()

        fig = px.histogram(seg_df, x="value_total_eur_y", nbins=40, title="Value distribution")
        fig.update_layout(height=360, xaxis_title="€/asset/year", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Drivers (simple, interpretable view)")
        # show how value varies with a few key knobs
        c1, c2 = st.columns(2)
        with c1:
            figA = px.scatter(seg_df, x="avg_window_hours", y="value_total_eur_y", trendline="ols",
                              title="Value vs availability window")
            figA.update_layout(height=320, xaxis_title="avg_window_hours", yaxis_title="€/asset/year")
            st.plotly_chart(figA, use_container_width=True)
        with c2:
            figB = px.scatter(seg_df, x="override_rate", y="value_total_eur_y", trendline="ols",
                              title="Value vs override rate")
            figB.update_layout(height=320, xaxis_title="override_rate", yaxis_title="€/asset/year")
            st.plotly_chart(figB, use_container_width=True)

        st.subheader("Channel breakdown (mean)")
        mean_break = seg_df[["value_da_eur_y","value_id_eur_y","value_as_eur_y"]].mean(numeric_only=True).fillna(0)
        pie = pd.DataFrame({"channel": mean_break.index, "eur_y": mean_break.values})
        figC = px.pie(pie, names="channel", values="eur_y", title="Mean contribution by channel")
        figC.update_layout(height=320)
        st.plotly_chart(figC, use_container_width=True)

elif page == "Segment Value Signatures":
    st.title("Segment Value Signatures")
    kpi_row()

    st.subheader("Signature table (p10 / p50 / p90, composition, reliability, impact)")
    show_cols = [
        "asset_type","segment","n",
        "value_p10","value_p50","value_p90",
        "rel_mean","impact_mean",
        "da_share","id_share","as_share",
        "sessions","kwh_sess","kwh_week","window","overrides",
        "prop_fit"
    ]
    st.dataframe(sig[show_cols].sort_values(["asset_type","value_p50"], ascending=[True,False]), use_container_width=True, height=520)

    st.subheader("Heatmap: median value by segment")
    pivot = sig.pivot_table(index="segment", columns="asset_type", values="value_p50", aggfunc="mean")
    pivot = pivot.sort_values(by=pivot.columns.tolist(), ascending=False, na_position="last")
    fig = px.imshow(pivot.fillna(0), aspect="auto", title="Median value heatmap (€/asset/year)")
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Roadmap Prioritization":
    st.title("Roadmap Prioritization")
    kpi_row()

    st.subheader("Attractiveness vs Readiness (segment roadmap)")
    # Allow tuning weights quickly
    with st.expander("Tune scoring (optional)", expanded=False):
        st.write("These weights are simple heuristics. You can later replace with your internal scoring model.")
        w_rel = st.slider("Reliability weight", 0.0, 1.0, 0.6, 0.05)
        w_imp = st.slider("Impact penalty", 0.0, 1.0, 0.4, 0.05)
        w_val = st.slider("Value weight", 0.0, 2.0, 1.0, 0.1)
        # recompute attractiveness quickly (still normalized)
        s2 = sig.copy()
        raw = (w_val * s2["value_p50"].fillna(s2["total_mean"])) * (0.5 + w_rel*s2["rel_mean"].fillna(0.5)) * (1.0 - w_imp*s2["impact_mean"].fillna(0.3))
        raw = raw.replace([np.inf,-np.inf], np.nan).fillna(0)
        s2["attractiveness"] = (raw / raw.max() * 100) if raw.max() > 0 else 0.0
        sig_plot = s2
    if "sig_plot" not in locals():
        sig_plot = sig.copy()

    sig_plot["label"] = sig_plot["asset_type"] + " | " + sig_plot["segment"]
    fig = px.scatter(
        sig_plot,
        x="readiness",
        y="attractiveness",
        color="asset_type",
        size="n",
        hover_data=["segment","value_p50","rel_mean","impact_mean","prop_fit","da_share","id_share","as_share"],
        title="Roadmap map (bigger bubble = more runs)"
    )
    fig.update_layout(height=520, xaxis_title="Execution readiness (proxy, 0..100)", yaxis_title="Segment attractiveness (0..100)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top roadmap candidates (by combined score)")
    sig_plot["combined"] = 0.55*sig_plot["attractiveness"] + 0.45*sig_plot["readiness"]
    top = sig_plot.sort_values("combined", ascending=False).head(15)
    st.dataframe(
        top[["asset_type","segment","n","value_p50","rel_mean","impact_mean","readiness","attractiveness","combined","prop_fit"]]
        .reset_index(drop=True),
        use_container_width=True
    )

    st.info(
        "Tip: Use this page in steering meetings: filter by country/DSO/market_stack/price_regime on the left to see how priorities shift."
    )

elif page == "Data & Export":
    st.title("Data & Export")
    st.write("Use this to validate your export, check missing columns, and download the enriched dataset (with segments).")

    st.subheader("Column coverage check")
    missing = [c for c in RECOMMENDED_COLUMNS if c not in df_raw.columns]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Found columns**")
        st.write(sorted(df_raw.columns.tolist()))
    with c2:
        st.markdown("**Missing recommended columns** (not mandatory)")
        st.write(missing if missing else "None 🎉")

    st.subheader("Preview (filtered)")
    st.dataframe(dff.head(200), use_container_width=True, height=420)

    st.subheader("Download enriched data")
    csv_bytes = dff.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered + segmented CSV", data=csv_bytes, file_name="flex_value_enriched_segmented.csv", mime="text/csv")

    st.subheader("Download segment signatures")
    sig_bytes = sig.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download segment_signature.csv", data=sig_bytes, file_name="segment_signature.csv", mime="text/csv")

# =========================================================
# Footer
# =========================================================
st.markdown(
    """
    <div style="margin-top:20px; font-size:12px; color:#888;">
    Model outputs → Flex Value Cube → Segments → Value Signatures → Proposition Fit → Roadmap.
    </div>
    """,
    unsafe_allow_html=True
)
