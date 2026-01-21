import os
import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EV Flex Value Segmentation Lab", page_icon="⚡", layout="wide")

# =========================
# Storage
# =========================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "ev_value_db.csv")

BASE_COLUMNS = [
    "entry_id",
    "created_ts",
    "label",

    # Customer behavior inputs
    "kwh_per_session",
    "frequency_type",          # daily / x_per_week / weekdays / weekends / custom
    "sessions_per_week",       # numeric
    "weekdays_only",           # 0/1
    "weekends_only",           # 0/1
    "custom_days",             # string like "Mon,Tue,Wed"
    "plug_in_hour",            # 0..23
    "plug_out_hour",           # 0..23

    # Optimization / scenario inputs
    "market_stack",            # DA / DA+ID / DA+ID+aFRR / etc.
    "grid_fee_scenario",       # Standard / Time-variable / custom
    "scenario_tag",            # free text (future)
    "value_eur_per_year",      # target metric

    # Optional notes
    "notes",
]

# =========================
# Helpers
# =========================
def now_ts():
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def safe_float(x):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def safe_int(x):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return np.nan
        return int(x)
    except Exception:
        return np.nan

def load_db() -> pd.DataFrame:
    if os.path.exists(DB_PATH):
        df = pd.read_csv(DB_PATH)
        # Ensure all expected columns exist
        for c in BASE_COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        df = df[BASE_COLUMNS]
        return df
    return pd.DataFrame(columns=BASE_COLUMNS)

def save_db(df: pd.DataFrame):
    df.to_csv(DB_PATH, index=False)

def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["kwh_per_session"] = pd.to_numeric(d["kwh_per_session"], errors="coerce")
    d["sessions_per_week"] = pd.to_numeric(d["sessions_per_week"], errors="coerce")
    d["plug_in_hour"] = pd.to_numeric(d["plug_in_hour"], errors="coerce")
    d["plug_out_hour"] = pd.to_numeric(d["plug_out_hour"], errors="coerce")
    d["value_eur_per_year"] = pd.to_numeric(d["value_eur_per_year"], errors="coerce")

    # Window length in hours across midnight
    # Example: 18 -> 6 means (24-18)+6 = 12
    pin = d["plug_in_hour"]
    pout = d["plug_out_hour"]
    d["window_hours"] = np.where(
        pin.isna() | pout.isna(),
        np.nan,
        np.where(pout >= pin, pout - pin, (24 - pin) + pout)
    )

    d["kwh_per_week"] = d["kwh_per_session"] * d["sessions_per_week"]

    # Weekend/weekday flags -> "weekend_share"
    # If weekends_only => 1.0; weekdays_only => 0.0; else estimate from sessions/week:
    # crude heuristic: if sessions_per_week <=2 assume weekend-ish 0.7; else 0.3
    wo = d["weekends_only"].fillna(0).astype(int)
    wd = d["weekdays_only"].fillna(0).astype(int)
    s = d["sessions_per_week"]

    weekend_share = np.where(
        wo == 1, 1.0,
        np.where(wd == 1, 0.0,
                 np.where(s.isna(), np.nan,
                          np.where(s <= 2, 0.7, 0.3)))
    )
    d["weekend_share"] = weekend_share

    # Convenience features for analysis
    d["is_time_variable_grid_fee"] = d["grid_fee_scenario"].astype(str).str.lower().str.contains("time").astype(int)
    d["market_stack_simple"] = d["market_stack"].astype(str).str.upper().str.replace(" ", "")

    # crude "availability_quality" proxy (longer window better)
    d["availability_quality"] = np.clip((d["window_hours"] - 2) / 12, 0, 1)

    # Value density metrics
    d["eur_per_kwh_week"] = d["value_eur_per_year"] / d["kwh_per_week"].replace(0, np.nan)
    d["eur_per_window_hour"] = d["value_eur_per_year"] / d["window_hours"].replace(0, np.nan)

    return d

def segment_ev_row(r) -> str:
    """
    Simple behavior-first segments for EV (editable rules).
    """
    k = safe_float(r.get("kwh_per_session"))
    s = safe_float(r.get("sessions_per_week"))
    w = safe_float(r.get("window_hours"))
    weekend_share = safe_float(r.get("weekend_share"))

    if any(pd.isna(x) for x in [k, s, w]):
        return "UNCLASSIFIED"

    # Weekend charger: low frequency, higher kWh, long window, weekend-heavy
    if (2 <= s <= 3) and (k >= 18) and (w >= 8) and (weekend_share >= 0.5):
        return "WEEKEND_CHARGER"

    # Daily commuter: high frequency, low/med kWh, decent window, weekday-heavy
    if (s >= 5) and (k <= 18) and (w >= 7) and (weekend_share <= 0.4):
        return "DAILY_COMMUTER"

    # High-energy daily
    if (s >= 5) and (k > 18):
        return "HIGH_ENERGY_DAILY"

    # Opportunistic / short window
    if (w < 5) or (s <= 2):
        return "OPPORTUNISTIC_SHORT_WINDOW"

    return "STANDARD"

def apply_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["segment"] = d.apply(segment_ev_row, axis=1)
    return d

def robust_corr(df: pd.DataFrame, x: str, y: str) -> float:
    a = pd.to_numeric(df[x], errors="coerce")
    b = pd.to_numeric(df[y], errors="coerce")
    m = a.notna() & b.notna()
    if m.sum() < 5:
        return np.nan
    return float(a[m].corr(b[m]))

def binned_effect(df: pd.DataFrame, feature: str, target: str, bins=6):
    x = pd.to_numeric(df[feature], errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 10:
        return pd.DataFrame(columns=["bin", "count", "x_mid", "y_mean"])

    xs = x[m]
    ys = y[m]

    # quantile bins (more stable with skewed data)
    try:
        q = pd.qcut(xs, q=bins, duplicates="drop")
    except Exception:
        q = pd.cut(xs, bins=bins)

    out = pd.DataFrame({"bin": q, "x": xs, "y": ys}).groupby("bin", dropna=True).agg(
        count=("y", "size"),
        x_mid=("x", "median"),
        y_mean=("y", "mean")
    ).reset_index()

    out["bin"] = out["bin"].astype(str)
    return out.sort_values("x_mid")

# =========================
# UI
# =========================
st.title("⚡ EV Flex Value Segmentation Lab")
st.caption("Manually log valuation outcomes by customer type & scenario → auto segmentation → insights & value-driver analysis.")

# Load
if "db" not in st.session_state:
    st.session_state.db = load_db()

db = st.session_state.db

# Sidebar filters
st.sidebar.header("Filters")
if len(db) == 0:
    st.sidebar.info("No entries yet. Add scenarios using the form.")

asset_note = st.sidebar.markdown("**EV-only for now** ✅")

# Derived + segmentation
df = compute_derived(db)
df = apply_segmentation(df)

# Filter widgets
market_options = sorted(df["market_stack"].dropna().astype(str).unique().tolist())
grid_options = sorted(df["grid_fee_scenario"].dropna().astype(str).unique().tolist())
seg_options = sorted(df["segment"].dropna().astype(str).unique().tolist())

f_market = st.sidebar.multiselect("Market stack", market_options, default=market_options)
f_grid = st.sidebar.multiselect("Grid fee scenario", grid_options, default=grid_options)
f_seg = st.sidebar.multiselect("Segment", seg_options, default=seg_options)

dff = df.copy()
if market_options:
    dff = dff[dff["market_stack"].astype(str).isin(f_market)]
if grid_options:
    dff = dff[dff["grid_fee_scenario"].astype(str).isin(f_grid)]
if seg_options:
    dff = dff[dff["segment"].astype(str).isin(f_seg)]

# Navigation
tab1, tab2, tab3, tab4 = st.tabs(["➕ Enter Scenarios", "📋 Database", "📊 Insights", "🧠 Value Drivers"])

# =========================
# Tab 1: Entry form
# =========================
with tab1:
    st.subheader("Enter a valuation outcome (one row = one customer type + scenario)")
    colA, colB, colC = st.columns([1.2, 1.1, 1.2])

    with colA:
        label = st.text_input("Label (short name)", placeholder="e.g., Daily_3kWh_18to6")
        kwh_per_session = st.number_input("kWh per session", min_value=0.0, value=3.0, step=0.5)
        frequency_type = st.selectbox(
            "Frequency pattern",
            ["Daily", "X times per week", "Weekdays only", "Weekends only", "Custom days"],
            index=0
        )
        if frequency_type == "Daily":
            sessions_per_week = 7
            weekdays_only = 0
            weekends_only = 0
            custom_days = ""
            st.write("Sessions/week = 7 (fixed for Daily)")
        elif frequency_type == "X times per week":
            sessions_per_week = st.number_input("Sessions per week (1–7)", min_value=1, max_value=7, value=3, step=1)
            weekdays_only = 0
            weekends_only = 0
            custom_days = ""
        elif frequency_type == "Weekdays only":
            sessions_per_week = 5
            weekdays_only = 1
            weekends_only = 0
            custom_days = "Mon,Tue,Wed,Thu,Fri"
            st.write("Sessions/week = 5 (fixed for Weekdays)")
        elif frequency_type == "Weekends only":
            sessions_per_week = 2
            weekdays_only = 0
            weekends_only = 1
            custom_days = "Sat,Sun"
            st.write("Sessions/week = 2 (fixed for Weekends)")
        else:
            custom_days = st.text_input("Custom days (comma separated)", value="Mon,Wed,Fri")
            # try estimate sessions/week from number of days
            days = [d.strip() for d in custom_days.split(",") if d.strip()]
            sessions_per_week = len(days) if len(days) > 0 else 3
            weekdays_only = 0
            weekends_only = 0
            st.write(f"Estimated sessions/week = {sessions_per_week} (based on custom days)")

    with colB:
        st.markdown("**Plug window**")
        plug_in_hour = st.number_input("Plug-in hour (0–23)", min_value=0, max_value=23, value=18, step=1)
        plug_out_hour = st.number_input("Plug-out hour (0–23)", min_value=0, max_value=23, value=6, step=1)
        st.caption("If plug-out < plug-in, we assume it crosses midnight (e.g., 18 → 6 = 12h window).")

        st.markdown("**Scenario setup**")
        market_stack = st.selectbox("Optimisation stack", ["DA", "DA+ID", "DA+ID+aFRR", "DA+ID+FCR", "DA+ID+FCR+aFRR", "Other…"], index=1)
        if market_stack == "Other…":
            market_stack = st.text_input("Enter market stack label", value="DA+ID+...")
        grid_fee_scenario = st.selectbox("Grid fee scenario", ["Standard grid fee", "Time-variable grid fee", "Other…"], index=0)
        if grid_fee_scenario == "Other…":
            grid_fee_scenario = st.text_input("Enter grid fee scenario label", value="My future scenario")

        scenario_tag = st.text_input("Scenario tag (optional)", placeholder="e.g., DE_Westnetz_2025_hist")

    with colC:
        st.markdown("**Valuation output**")
        value_eur_per_year = st.number_input("Flex value (€/year)", value=240.0, step=5.0)
        notes = st.text_area("Notes (optional)", height=160, placeholder="Assumptions, special constraints, etc.")

        st.markdown("---")
        add = st.button("✅ Add entry", use_container_width=True)

    if add:
        new_row = {
            "entry_id": f"E{len(db)+1:06d}",
            "created_ts": now_ts(),
            "label": (label.strip() if label and label.strip() else f"EV_{len(db)+1:06d}"),
            "kwh_per_session": float(kwh_per_session),
            "frequency_type": str(frequency_type),
            "sessions_per_week": int(sessions_per_week),
            "weekdays_only": int(weekdays_only),
            "weekends_only": int(weekends_only),
            "custom_days": str(custom_days),
            "plug_in_hour": int(plug_in_hour),
            "plug_out_hour": int(plug_out_hour),
            "market_stack": str(market_stack),
            "grid_fee_scenario": str(grid_fee_scenario),
            "scenario_tag": str(scenario_tag),
            "value_eur_per_year": float(value_eur_per_year),
            "notes": str(notes),
        }
        db2 = pd.concat([db, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state.db = db2
        save_db(db2)
        st.success("Saved. Go to 'Database' or 'Insights' tabs.")
        st.rerun()

# =========================
# Tab 2: Database view + edit actions
# =========================
with tab2:
    st.subheader("Database (filtered)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entries (filtered)", f"{len(dff):,}")
    c2.metric("Segments", f"{dff['segment'].nunique() if len(dff) else 0}")
    c3.metric("Median value €/y", f"{np.nanmedian(dff['value_eur_per_year']):.1f}" if len(dff) else "—")
    c4.metric("Median kWh/week", f"{np.nanmedian(dff['kwh_per_week']):.1f}" if len(dff) else "—")

    show_cols = [
        "entry_id","created_ts","label",
        "kwh_per_session","sessions_per_week","frequency_type",
        "plug_in_hour","plug_out_hour","window_hours",
        "market_stack","grid_fee_scenario","scenario_tag",
        "value_eur_per_year","kwh_per_week","eur_per_kwh_week",
        "segment","notes"
    ]
    st.dataframe(dff[show_cols].sort_values("created_ts", ascending=False), use_container_width=True, height=420)

    st.markdown("### Export / reset")
    colx, coly = st.columns([1,1])
    with colx:
        csv_bytes = dff.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download filtered CSV", data=csv_bytes, file_name="ev_value_db_filtered.csv", mime="text/csv", use_container_width=True)
    with coly:
        if st.button("🗑️ Delete ALL entries (danger)", type="secondary", use_container_width=True):
            st.session_state.db = pd.DataFrame(columns=BASE_COLUMNS)
            save_db(st.session_state.db)
            st.warning("All entries deleted.")
            st.rerun()

# =========================
# Tab 3: Insights
# =========================
with tab3:
    st.subheader("Insights: segments × scenarios")

    if len(dff) < 3:
        st.info("Add a few scenarios first to see segment insights.")
    else:
        # Segment signature table
        g = dff.groupby(["segment","market_stack","grid_fee_scenario"], dropna=False)
        sig = g.agg(
            n=("value_eur_per_year","size"),
            p10=("value_eur_per_year", lambda x: np.nanpercentile(x,10)),
            p50=("value_eur_per_year", lambda x: np.nanpercentile(x,50)),
            p90=("value_eur_per_year", lambda x: np.nanpercentile(x,90)),
            kwhwk=("kwh_per_week","mean"),
            win=("window_hours","mean"),
            eur_per_kwhwk=("eur_per_kwh_week","mean")
        ).reset_index().sort_values(["p50","n"], ascending=[False, False])

        st.markdown("### Segment signatures (by market stack & grid fee scenario)")
        st.dataframe(sig, use_container_width=True, height=360)

        # Simple “where is it worth targeting” view:
        st.markdown("### Targeting map (median value vs kWh/week)")
        pivot = dff.groupby(["segment"]).agg(
            n=("value_eur_per_year","size"),
            value_p50=("value_eur_per_year", lambda x: np.nanpercentile(x,50)),
            kwh_p50=("kwh_per_week", lambda x: np.nanpercentile(x,50)),
            win_mean=("window_hours","mean"),
        ).reset_index()

        # Use Streamlit native charts (no extra libs)
        st.scatter_chart(
            pivot.rename(columns={"kwh_p50":"kwh_per_week_p50", "value_p50":"value_eur_y_p50"})[["kwh_per_week_p50","value_eur_y_p50"]],
            x="kwh_per_week_p50",
            y="value_eur_y_p50",
            height=300
        )
        st.caption("Interpretation: right-up is best (high energy + high value). Add filters on the left to compare DA vs DA+ID vs AS, Standard vs Time-variable.")

        st.markdown("### Best segments by scenario (top median value)")
        topN = st.slider("Top N rows", 5, 25, 10)
        best = sig.head(topN).copy()
        st.dataframe(best, use_container_width=True)

        st.markdown("### Quick narrative (auto)")
        # Simple narrative: where time-variable helps most, and which market stack adds value
        def median_by(mask):
            x = dff.loc[mask, "value_eur_per_year"]
            return float(np.nanpercentile(x,50)) if x.notna().sum() >= 3 else np.nan

        # time-variable vs standard delta
        tv = dff["grid_fee_scenario"].astype(str).str.lower().str.contains("time")
        std = dff["grid_fee_scenario"].astype(str).str.lower().str.contains("standard")
        tv_med = median_by(tv)
        std_med = median_by(std)
        if not np.isnan(tv_med) and not np.isnan(std_med):
            st.write(f"- Median value under **Time-variable grid fee**: **{tv_med:.1f} €/y** vs **Standard**: **{std_med:.1f} €/y** → delta **{(tv_med-std_med):+.1f} €/y** (within current filters).")

        # DA vs DA+ID vs DA+ID+AS
        def has_stack(s): return dff["market_stack"].astype(str).str.upper().str.replace(" ", "").eq(s)
        da_med = median_by(has_stack("DA"))
        daid_med = median_by(has_stack("DA+ID"))
        daid_afrr_med = median_by(has_stack("DA+ID+AFRR"))
        if not np.isnan(da_med): st.write(f"- **DA** median: **{da_med:.1f} €/y**")
        if not np.isnan(daid_med): st.write(f"- **DA+ID** median: **{daid_med:.1f} €/y** (Δ vs DA: {(daid_med-da_med):+.1f} €/y)" if not np.isnan(da_med) else f"- **DA+ID** median: **{daid_med:.1f} €/y**")
        if not np.isnan(daid_afrr_med): st.write(f"- **DA+ID+aFRR** median: **{daid_afrr_med:.1f} €/y**")

# =========================
# Tab 4: Value driver analysis
# =========================
with tab4:
    st.subheader("Value drivers (what explains €/year?)")

    if len(dff) < 10:
        st.info("Add at least ~10 scenarios for the driver analysis to become meaningful.")
    else:
        # Correlations (simple but useful)
        feats = ["kwh_per_session","sessions_per_week","kwh_per_week","window_hours","weekend_share","availability_quality","is_time_variable_grid_fee"]
        corr_rows = []
        for f in feats:
            corr_rows.append({"feature": f, "corr_with_value": robust_corr(dff, f, "value_eur_per_year")})
        corr_df = pd.DataFrame(corr_rows).sort_values("corr_with_value", ascending=False)
        st.markdown("### Correlation ranking (directional)")
        st.dataframe(corr_df, use_container_width=True, height=260)
        st.caption("Correlation is not causality, but it quickly tells you what moves together with value in your runs.")

        # Binned effects
        st.markdown("### Binned effects (how value changes across ranges)")
        feature_pick = st.selectbox("Pick a feature", ["kwh_per_week","window_hours","sessions_per_week","kwh_per_session","weekend_share"])
        be = binned_effect(dff, feature_pick, "value_eur_per_year", bins=6)
        st.dataframe(be, use_container_width=True, height=240)
        st.line_chart(be.set_index("x_mid")["y_mean"], height=250)
        st.caption("Look for saturation (diminishing returns), thresholds, and non-linear effects.")

        # Scenario deltas by categories
        st.markdown("### Scenario deltas (stack & grid fee)")
        # median by market_stack
        med_stack = dff.groupby("market_stack")["value_eur_per_year"].median().sort_values(ascending=False)
        st.write("**Median €/year by market stack (within filters):**")
        st.dataframe(med_stack.reset_index().rename(columns={"value_eur_per_year":"median_value_eur_y"}), use_container_width=True)

        med_grid = dff.groupby("grid_fee_scenario")["value_eur_per_year"].median().sort_values(ascending=False)
        st.write("**Median €/year by grid fee scenario:**")
        st.dataframe(med_grid.reset_index().rename(columns={"value_eur_per_year":"median_value_eur_y"}), use_container_width=True)

        # Segment x scenario heatmap-style pivot table
        st.markdown("### Segment × Market Stack (median value table)")
        pivot = dff.pivot_table(index="segment", columns="market_stack", values="value_eur_per_year", aggfunc="median")
        st.dataframe(pivot, use_container_width=True, height=320)

st.markdown(
    "<div style='margin-top:16px; font-size:12px; color:#777;'>"
    "Data saved to <code>data/ev_value_db.csv</code> in your Streamlit project folder."
    "</div>",
    unsafe_allow_html=True
)
