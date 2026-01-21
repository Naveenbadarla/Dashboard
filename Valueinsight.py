import os
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
    "frequency_type",
    "sessions_per_week",
    "weekdays_only",
    "weekends_only",
    "custom_days",
    "plug_in_hour",
    "plug_out_hour",
    # Optimization / scenario inputs
    "market_stack",
    "grid_fee_scenario",
    "scenario_tag",
    "value_eur_per_year",
    # Optional
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

def load_db() -> pd.DataFrame:
    if os.path.exists(DB_PATH):
        df = pd.read_csv(DB_PATH)
        for c in BASE_COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        return df[BASE_COLUMNS]
    return pd.DataFrame(columns=BASE_COLUMNS)

def save_db(df: pd.DataFrame):
    df.to_csv(DB_PATH, index=False)

def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ["kwh_per_session", "sessions_per_week", "plug_in_hour", "plug_out_hour", "value_eur_per_year"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # window across midnight
    pin = d["plug_in_hour"]
    pout = d["plug_out_hour"]
    d["window_hours"] = np.where(
        pin.isna() | pout.isna(),
        np.nan,
        np.where(pout >= pin, pout - pin, (24 - pin) + pout)
    )

    d["kwh_per_week"] = d["kwh_per_session"] * d["sessions_per_week"]

    # weekend share heuristic
    wo = d["weekends_only"].fillna(0).astype(int)
    wd = d["weekdays_only"].fillna(0).astype(int)
    s = d["sessions_per_week"]
    d["weekend_share"] = np.where(
        wo == 1, 1.0,
        np.where(wd == 1, 0.0,
                 np.where(s.isna(), np.nan,
                          np.where(s <= 2, 0.7, 0.3)))
    )

    d["is_time_variable_grid_fee"] = d["grid_fee_scenario"].astype(str).str.lower().str.contains("time").astype(int)
    d["market_stack_simple"] = d["market_stack"].astype(str).str.upper().str.replace(" ", "")

    d["availability_quality"] = np.clip((d["window_hours"] - 2) / 12, 0, 1)

    d["eur_per_kwh_week"] = d["value_eur_per_year"] / d["kwh_per_week"].replace(0, np.nan)
    d["eur_per_window_hour"] = d["value_eur_per_year"] / d["window_hours"].replace(0, np.nan)

    return d

def segment_ev_row(r) -> str:
    k = safe_float(r.get("kwh_per_session"))
    s = safe_float(r.get("sessions_per_week"))
    w = safe_float(r.get("window_hours"))
    weekend_share = safe_float(r.get("weekend_share"))

    if any(pd.isna(x) for x in [k, s, w]):
        return "UNCLASSIFIED"

    if (2 <= s <= 3) and (k >= 18) and (w >= 8) and (weekend_share >= 0.5):
        return "WEEKEND_CHARGER"

    if (s >= 5) and (k <= 18) and (w >= 7) and (weekend_share <= 0.4):
        return "DAILY_COMMUTER"

    if (s >= 5) and (k > 18):
        return "HIGH_ENERGY_DAILY"

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
# Bucketing logic
# =========================
def compute_global_cutoffs(dff: pd.DataFrame):
    v = dff["value_eur_per_year"].dropna()
    if len(v) < 5:
        return {"v_p33": 100.0, "v_p66": 200.0}
    return {
        "v_p33": float(np.nanpercentile(v, 33)),
        "v_p66": float(np.nanpercentile(v, 66)),
    }

def value_bucket(row, cutoffs):
    v = safe_float(row.get("value_eur_per_year"))
    if pd.isna(v):
        return "UNKNOWN"
    if v >= cutoffs["v_p66"]:
        return "HIGH"
    if v >= cutoffs["v_p33"]:
        return "MEDIUM"
    return "LOW"

def style_bucket(b):
    if b == "HIGH":
        return "🟢 HIGH"
    if b == "MEDIUM":
        return "🟡 MEDIUM"
    if b == "LOW":
        return "🔴 LOW"
    return "⚪ UNKNOWN"

def why_value(row):
    reasons = []
    kwh_w = safe_float(row.get("kwh_per_week"))
    w = safe_float(row.get("window_hours"))
    s = safe_float(row.get("sessions_per_week"))
    k = safe_float(row.get("kwh_per_session"))
    tv = int(row.get("is_time_variable_grid_fee", 0) or 0)
    stack = str(row.get("market_stack", "")).upper()

    if not pd.isna(kwh_w):
        if kwh_w >= 120:
            reasons.append("High kWh/week → more shiftable volume")
        elif kwh_w >= 60:
            reasons.append("Medium kWh/week supports steady value")
        else:
            reasons.append("Low kWh/week limits total value")

    if not pd.isna(w):
        if w >= 10:
            reasons.append("Long window → high temporal freedom")
        elif w >= 7:
            reasons.append("Good overnight window → easy optimization")
        else:
            reasons.append("Short window → limited flexibility")

    if not pd.isna(s) and not pd.isna(k):
        if s >= 5 and k <= 18:
            reasons.append("Frequent smaller sessions → predictable")
        if s <= 3 and k >= 18:
            reasons.append("Infrequent large sessions → big per-session shift")

    if tv == 1:
        reasons.append("Time-variable grid fee → extra savings windows")

    if "AFRR" in stack or "FCR" in stack:
        reasons.append("AS enabled → added upside if reliable")
    elif "DA+ID" in stack.replace(" ", ""):
        reasons.append("ID enabled → captures intraday spreads")

    if len(reasons) > 4:
        reasons = reasons[:4]

    return " • " + "\n • ".join(reasons) if reasons else "—"

# =========================
# UI
# =========================
st.title("⚡ EV Flex Value Segmentation Lab")
st.caption("Manual scenario logging → segmentation → bucketed targeting view → insights & value-driver analysis.")

if "db" not in st.session_state:
    st.session_state.db = load_db()

db = st.session_state.db
df = compute_derived(db)
df = apply_segmentation(df)

# Sidebar filters
st.sidebar.header("Filters")
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

# Buckets computed on filtered data (reacts to filters)
cutoffs = compute_global_cutoffs(dff) if len(dff) else {"v_p33": 100.0, "v_p66": 200.0}
dff = dff.copy()
dff["value_bucket"] = dff.apply(lambda r: value_bucket(r, cutoffs), axis=1)
dff["value_bucket_label"] = dff["value_bucket"].apply(style_bucket)
dff["why_earns_more"] = dff.apply(why_value, axis=1)

# sort helper for bucket order
bucket_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNKNOWN": 3}
dff["bucket_rank"] = dff["value_bucket"].map(bucket_order).fillna(9).astype(int)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["➕ Enter Scenarios", "📋 Database", "🧺 Buckets", "📊 Insights", "🧠 Value Drivers"]
)

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
            days = [d.strip() for d in custom_days.split(",") if d.strip()]
            sessions_per_week = len(days) if len(days) > 0 else 3
            weekdays_only = 0
            weekends_only = 0
            st.write(f"Estimated sessions/week = {sessions_per_week} (based on custom days)")

    with colB:
        st.markdown("**Plug window**")
        plug_in_hour = st.number_input("Plug-in hour (0–23)", min_value=0, max_value=23, value=18, step=1)
        plug_out_hour = st.number_input("Plug-out hour (0–23)", min_value=0, max_value=23, value=6, step=1)
        st.caption("If plug-out < plug-in, it crosses midnight (e.g., 18 → 6 = 12h).")

        st.markdown("**Scenario setup**")
        market_stack = st.selectbox(
            "Optimisation stack",
            ["DA", "DA+ID", "DA+ID+aFRR", "DA+ID+FCR", "DA+ID+FCR+aFRR", "Other…"],
            index=1
        )
        if market_stack == "Other…":
            market_stack = st.text_input("Enter market stack label", value="DA+ID+...")

        grid_fee_scenario = st.selectbox(
            "Grid fee scenario",
            ["Standard grid fee", "Time-variable grid fee", "Other…"],
            index=0
        )
        if grid_fee_scenario == "Other…":
            grid_fee_scenario = st.text_input("Enter grid fee scenario label", value="My future scenario")

        scenario_tag = st.text_input("Scenario tag (optional)", placeholder="e.g., DE_Westnetz_2025_hist")

    with colC:
        st.markdown("**Valuation output**")
        value_eur_per_year = st.number_input("Flex value (€/year)", value=240.0, step=5.0)
        notes = st.text_area("Notes (optional)", height=160)
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
        st.success("Saved. Check the Buckets tab.")
        st.rerun()

# =========================
# Tab 2: Database
# =========================
with tab2:
    st.subheader("Database (filtered)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entries", f"{len(dff):,}")
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
# Tab 3: Buckets (fixed)
# =========================
with tab3:
    st.subheader("🧺 Easy Bucketing: highlight high-value customers & explain why")

    if len(dff) < 3:
        st.info("Add a few scenarios first to see bucketing.")
    else:
        st.markdown("### Bucket thresholds (auto from current filtered data)")
        st.write(
            f"- **LOW**: < **{cutoffs['v_p33']:.1f} €/y**  \n"
            f"- **MEDIUM**: **{cutoffs['v_p33']:.1f} – {cutoffs['v_p66']:.1f} €/y**  \n"
            f"- **HIGH**: ≥ **{cutoffs['v_p66']:.1f} €/y**"
        )
        st.caption("Thresholds adapt as you add rows. Use filters to bucket within a scenario.")

        # Counts
        bucket_counts = dff["value_bucket"].value_counts(dropna=False).reindex(["HIGH","MEDIUM","LOW","UNKNOWN"]).fillna(0).astype(int)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 HIGH", int(bucket_counts.get("HIGH", 0)))
        c2.metric("🟡 MEDIUM", int(bucket_counts.get("MEDIUM", 0)))
        c3.metric("🔴 LOW", int(bucket_counts.get("LOW", 0)))
        c4.metric("⚪ UNKNOWN", int(bucket_counts.get("UNKNOWN", 0)))

        st.markdown("### Bucket map by Segment")
        seg_bucket = dff.pivot_table(index="segment", columns="value_bucket", values="entry_id", aggfunc="count", fill_value=0)
        for col in ["HIGH","MEDIUM","LOW","UNKNOWN"]:
            if col not in seg_bucket.columns:
                seg_bucket[col] = 0
        seg_bucket = seg_bucket[["HIGH","MEDIUM","LOW","UNKNOWN"]].sort_values(["HIGH","MEDIUM","LOW"], ascending=False)
        st.dataframe(seg_bucket, use_container_width=True, height=260)

        st.markdown("### High-value customer types (with 'why' explanation)")
        top_high = dff[dff["value_bucket"] == "HIGH"].sort_values("value_eur_per_year", ascending=False)
        if len(top_high) == 0:
            st.info("No HIGH rows under current filters. Try removing filters or add more entries.")
        else:
            show = [
                "value_bucket_label","label","segment",
                "value_eur_per_year","kwh_per_week","window_hours",
                "sessions_per_week","kwh_per_session",
                "market_stack","grid_fee_scenario",
                "why_earns_more"
            ]
            st.dataframe(top_high[show].head(30), use_container_width=True, height=420)

        st.markdown("### Why do HIGH customers earn more? (quick comparison vs LOW)")
        high = dff[dff["value_bucket"] == "HIGH"]
        low = dff[dff["value_bucket"] == "LOW"]

        def stat_line(name, series_high, series_low):
            if series_high.notna().sum() < 2 or series_low.notna().sum() < 2:
                return None
            return f"- **{name}**: HIGH median **{np.nanmedian(series_high):.1f}** vs LOW median **{np.nanmedian(series_low):.1f}**"

        lines = []
        lines.append(stat_line("kWh/week", high["kwh_per_week"], low["kwh_per_week"]))
        lines.append(stat_line("Window hours", high["window_hours"], low["window_hours"]))
        lines.append(stat_line("Sessions/week", high["sessions_per_week"], low["sessions_per_week"]))
        lines.append(stat_line("kWh/session", high["kwh_per_session"], low["kwh_per_session"]))

        if len(high) >= 2 and len(low) >= 2:
            tv_high = high["is_time_variable_grid_fee"].mean()
            tv_low = low["is_time_variable_grid_fee"].mean()
            lines.append(f"- **Time-variable grid fee share**: HIGH **{tv_high*100:.0f}%** vs LOW **{tv_low*100:.0f}%**")

        lines = [l for l in lines if l is not None]
        st.write("\n".join(lines) if lines else "Add more variety of entries to make this comparison meaningful.")

        st.markdown("### Bucketed table (all rows)")
        bucketed_cols = [
            "value_bucket_label","segment","label","value_eur_per_year",
            "kwh_per_week","window_hours","sessions_per_week","kwh_per_session",
            "market_stack","grid_fee_scenario","why_earns_more"
        ]

        # ✅ FIX: sort on full dff (has value_bucket), then select columns
        sorted_all = dff.sort_values(["bucket_rank","value_eur_per_year"], ascending=[True, False])
        st.dataframe(sorted_all[bucketed_cols], use_container_width=True, height=520)

# =========================
# Tab 4: Insights
# =========================
with tab4:
    st.subheader("Insights: segments × scenarios")

    if len(dff) < 3:
        st.info("Add a few scenarios first to see segment insights.")
    else:
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

        st.markdown("### Best rows by scenario (top median value)")
        topN = st.slider("Top N rows", 5, 25, 10)
        st.dataframe(sig.head(topN), use_container_width=True)

# =========================
# Tab 5: Drivers
# =========================
with tab5:
    st.subheader("Value drivers (what explains €/year?)")

    if len(dff) < 10:
        st.info("Add at least ~10 scenarios for the driver analysis to become meaningful.")
    else:
        feats = ["kwh_per_session","sessions_per_week","kwh_per_week","window_hours","weekend_share","availability_quality","is_time_variable_grid_fee"]
        corr_rows = [{"feature": f, "corr_with_value": robust_corr(dff, f, "value_eur_per_year")} for f in feats]
        corr_df = pd.DataFrame(corr_rows).sort_values("corr_with_value", ascending=False)

        st.markdown("### Correlation ranking (directional)")
        st.dataframe(corr_df, use_container_width=True, height=260)

        st.markdown("### Binned effects (non-linear patterns)")
        feature_pick = st.selectbox("Pick a feature", ["kwh_per_week","window_hours","sessions_per_week","kwh_per_session","weekend_share"])
        be = binned_effect(dff, feature_pick, "value_eur_per_year", bins=6)
        st.dataframe(be, use_container_width=True, height=240)
        st.line_chart(be.set_index("x_mid")["y_mean"], height=250)

        st.markdown("### Segment × Market Stack (median value table)")
        pivot = dff.pivot_table(index="segment", columns="market_stack", values="value_eur_per_year", aggfunc="median")
        st.dataframe(pivot, use_container_width=True, height=320)

st.markdown(
    "<div style='margin-top:16px; font-size:12px; color:#777;'>"
    "Data saved to <code>data/ev_value_db.csv</code> in your Streamlit project folder."
    "</div>",
    unsafe_allow_html=True
)
