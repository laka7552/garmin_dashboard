"""
garmin_app.py — Streamlit Dashboard
=====================================
Reads garmin_history.json and renders the full training &
recovery dashboard as an interactive web app.

Run locally:
    streamlit run garmin_app.py
"""

import json
import io
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Garmin Training Dashboard",
    page_icon  = "🚴",
    layout     = "wide",
)

# ── Dark theme override (matches matplotlib colours) ──────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0f0f1a; color: #f1f5f9; }
  section[data-testid="stSidebar"] { background-color: #1c1c2e; }
  .metric-card {
      background: #1c1c2e;
      border-radius: 10px;
      padding: 18px 24px;
      text-align: center;
  }
  .metric-value { font-size: 2.2rem; font-weight: 700; margin: 0; }
  .metric-label { font-size: 0.75rem; color: #64748b;
                  text-transform: uppercase; letter-spacing: 1px; margin: 0; }
  .metric-sub   { font-size: 0.8rem; color: #94a3b8; margin: 4px 0 0 0; }
  .status-card {
      background: #1c1c2e;
      border-radius: 10px;
      padding: 16px 20px;
  }
  .status-title { font-size: 0.7rem; color: #64748b;
                  text-transform: uppercase; letter-spacing: 1px; }
  .status-label { font-size: 1.1rem; font-weight: 700; margin: 4px 0; }
  .status-detail{ font-size: 0.8rem; color: #94a3b8; font-style: italic; }
  hr.divider { border: none; border-top: 1px solid #2d2d44; margin: 16px 0; }
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
DATA_FILE = Path(__file__).parent / "garmin_history.json"

@st.cache_data(ttl=300)   # re-read file at most every 5 minutes
def load_data():
    if not DATA_FILE.exists():
        return None
    with open(DATA_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    df["date"]        = pd.to_datetime(df["date"])
    df                = df.sort_values("date").reset_index(drop=True)
    df["sleep_hours"] = pd.to_numeric(df["sleep_seconds"], errors="coerce") / 3600
    df["daily_tss"]   = pd.to_numeric(df["daily_tss"],     errors="coerce").fillna(0)
    df["hrv_avg"]     = pd.to_numeric(df["hrv_weekly_avg"],errors="coerce")
    df["resting_hr"]  = pd.to_numeric(df["resting_hr"],    errors="coerce")
    return df


raw_df = load_data()

if raw_df is None:
    st.error("garmin_history.json not found. Run garmin_fetch_history.py first.")
    st.stop()


# ── Sidebar — date range ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/garmin.png", width=60)
    st.title("Garmin Dashboard")
    st.markdown("---")

    min_date = raw_df["date"].min().date()
    max_date = raw_df["date"].max().date()

    days_options = {"2 Weeks": 14, "4 Weeks": 28, "8 Weeks": 56, "All": 999}
    selected_range = st.selectbox("Date range", list(days_options.keys()), index=1)
    n_days = days_options[selected_range]

    cutoff = max_date - timedelta(days=n_days - 1)
    cutoff = max(cutoff, min_date)

    st.markdown("---")
    st.caption(f"Data from {min_date.strftime('%d %b %Y')} "
               f"to {max_date.strftime('%d %b %Y')}")
    st.caption(f"Last updated: {max_date.strftime('%d %b %Y')}")

    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()


# ── Filter to selected range ───────────────────────────────────────────────────
df = raw_df[raw_df["date"].dt.date >= cutoff].copy().reset_index(drop=True)


# ── Compute ATL / CTL / TSB over full history, then slice ─────────────────────
ATL_DAYS  = 7
CTL_DAYS  = 42
atl_alpha = 2 / (ATL_DAYS + 1)
ctl_alpha = 2 / (CTL_DAYS + 1)

atl_vals, ctl_vals = [0.0], [0.0]
for tss in raw_df["daily_tss"]:
    atl_vals.append(atl_vals[-1] + atl_alpha * (tss - atl_vals[-1]))
    ctl_vals.append(ctl_vals[-1] + ctl_alpha * (tss - ctl_vals[-1]))

raw_df["atl"] = atl_vals[1:]
raw_df["ctl"] = ctl_vals[1:]
raw_df["tsb"] = raw_df["ctl"] - raw_df["atl"]

# Adaptive bands (full history for stable rolling stats)
raw_df["hrv_band_mid"]  = raw_df["hrv_avg"].rolling(7, min_periods=3).mean()
raw_df["hrv_band_std"]  = raw_df["hrv_avg"].rolling(7, min_periods=3).std()
raw_df["hrv_band_high"] = raw_df["hrv_band_mid"] + 1.5 * raw_df["hrv_band_std"]
raw_df["hrv_band_low"]  = (raw_df["hrv_band_mid"] - 1.5 * raw_df["hrv_band_std"]).clip(lower=0)

raw_df["rhr_band_mid"]  = raw_df["resting_hr"].rolling(7, min_periods=3).mean()
raw_df["rhr_band_std"]  = raw_df["resting_hr"].rolling(7, min_periods=3).std()
raw_df["rhr_band_high"] = raw_df["rhr_band_mid"] + 1.5 * raw_df["rhr_band_std"]
raw_df["rhr_band_low"]  = (raw_df["rhr_band_mid"] - 1.5 * raw_df["rhr_band_std"]).clip(lower=0)

df = raw_df[raw_df["date"].dt.date >= cutoff].copy().reset_index(drop=True)


# ── Colours ───────────────────────────────────────────────────────────────────
BG     = "#0f0f1a"
CARD   = "#1c1c2e"
HRV_C  = "#a78bfa"
HR_C   = "#f87171"
SLP_C  = "#60a5fa"
TSS_C  = "#34d399"
ATL_C  = "#fb923c"
CTL_C  = "#38bdf8"
WHITE  = "#f1f5f9"
MUTED  = "#64748b"


# ── Helper: latest non-null value ─────────────────────────────────────────────
def last(series):
    v = series.dropna()
    return v.iloc[-1] if not v.empty else None

def rolling7(series):
    return series.rolling(7, min_periods=1).mean()


# ══════════════════════════════════════════════════════════════════════════════
#  TOP: metric cards
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🚴 Training & Recovery Dashboard")
st.markdown(f"*{df['date'].min().strftime('%d %b')} – {df['date'].max().strftime('%d %b %Y')}*")
st.markdown('<hr class="divider">', unsafe_allow_html=True)

atl_now   = df["atl"].iloc[-1]
ctl_now   = df["ctl"].iloc[-1]
tsb_now   = df["tsb"].iloc[-1]
hrv_now   = last(df["hrv_avg"])
rhr_now   = last(df["resting_hr"])
slp_now   = last(df["sleep_hours"])
tsb_col   = "#4ade80" if tsb_now >= 0 else "#f87171"

def metric_card(label, value, color, sub=""):
    val_str = f"{value:.0f}" if value is not None else "—"
    return f"""
    <div class="metric-card">
      <p class="metric-label">{label}</p>
      <p class="metric-value" style="color:{color};">{val_str}</p>
      <p class="metric-sub">{sub}</p>
    </div>"""

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.markdown(metric_card("ATL · Fatigue", atl_now, ATL_C, f"{ATL_DAYS}-day avg"), unsafe_allow_html=True)
c2.markdown(metric_card("CTL · Fitness", ctl_now, CTL_C, f"{CTL_DAYS}-day avg"), unsafe_allow_html=True)
c3.markdown(metric_card("TSB · Form",    tsb_now, tsb_col, "CTL − ATL"), unsafe_allow_html=True)
c4.markdown(metric_card("HRV",  hrv_now, HRV_C,  "ms · weekly avg"), unsafe_allow_html=True)
c5.markdown(metric_card("Resting HR", rhr_now, HR_C, "bpm"), unsafe_allow_html=True)
c6.markdown(metric_card("Sleep", slp_now, SLP_C, "hours last night"), unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── Interpretation ────────────────────────────────────────────────────────────
def interpret(tsb, atl, ctl, hrv_today, hrv_high, hrv_low, rhr_today, rhr_high, rhr_low):
    if tsb > 15:
        ts = ("Peak Form 🟢", "#4ade80",
              "Well rested with solid fitness base. Ideal for a key effort or race.") \
             if atl >= ctl * 0.6 else \
             ("Tapering / Undertraining ⚠️", "#fbbf24",
              "TSB high but ATL well below CTL — fitness may be declining. Add load.")
    elif tsb >= 0:
        ts = ("Maintaining 🟢", "#4ade80",
              "Load balanced, fitness stable. Good for consistent steady training.")
    elif tsb >= -10:
        ts = ("Productive Training 🔵", CTL_C,
              "Moderate fatigue, fitness building. Normal and healthy training block.")
    elif tsb >= -20:
        ts = ("Accumulating Fatigue 🟡", "#fbbf24",
              "Significant fatigue building up. Monitor sleep and recovery closely.")
    elif tsb >= -30:
        ts = ("Heavy Training Block 🟠", "#f97316",
              "High fatigue load. Prioritise sleep, nutrition and easy sessions.")
    else:
        ts = ("Overreaching Risk 🔴", HR_C,
              "Very high fatigue. Take 1–2 rest or very easy days immediately.")

    signals = []
    if hrv_today and hrv_high and hrv_low:
        if hrv_today > hrv_high:
            signals.append(("↑ HRV above normal range — nervous system well recovered", True))
        elif hrv_today < hrv_low:
            signals.append(("↓ HRV below normal range — still recovering", False))
        else:
            signals.append(("→ HRV within normal range", True))

    if rhr_today and rhr_high and rhr_low:
        if rhr_today > rhr_high:
            signals.append(("↑ Resting HR above normal range — fatigue or stress", False))
        elif rhr_today < rhr_low:
            signals.append(("↓ Resting HR below normal range — very well recovered", True))
        else:
            signals.append(("→ Resting HR within normal range", True))

    good  = sum(1 for _, ok in signals if ok)
    total = len(signals)
    if total == 0:
        rs = ("No recovery data", MUTED, [])
    elif good == total:
        rs = ("Good Recovery 🟢", "#4ade80", signals)
    elif good >= total / 2:
        rs = ("Moderate Recovery 🟡", "#fbbf24", signals)
    else:
        rs = ("Poor Recovery 🔴", HR_C, signals)

    return ts, rs

hrv_hi  = last(df["hrv_band_high"])
hrv_lo  = last(df["hrv_band_low"])
rhr_hi  = last(df["rhr_band_high"])
rhr_lo  = last(df["rhr_band_low"])

(t_label, t_color, t_detail), (r_label, r_color, r_signals) = interpret(
    tsb_now, atl_now, ctl_now, hrv_now, hrv_hi, hrv_lo, rhr_now, rhr_hi, rhr_lo)

icol1, icol2 = st.columns(2)
with icol1:
    st.markdown(f"""
    <div class="status-card">
      <p class="status-title">Training Status</p>
      <p class="status-label" style="color:{t_color};">{t_label}</p>
      <p class="status-detail">{t_detail}</p>
    </div>""", unsafe_allow_html=True)

with icol2:
    sigs_html = "".join(
        f'<p class="status-detail" style="color:{"#4ade80" if ok else HR_C};">'
        f'{s}</p>'
        for s, ok in r_signals
    )
    st.markdown(f"""
    <div class="status-card">
      <p class="status-title">Recovery Status</p>
      <p class="status-label" style="color:{r_color};">{r_label}</p>
      {sigs_html}
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS — reuse the matplotlib figure from garmin_visualize_weekly.py
# ══════════════════════════════════════════════════════════════════════════════

DATE_FMT = mdates.DateFormatter("%d %b")

def style_ax(ax, title, ylabel, color):
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_color("#2d2d44")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(title, color=color, fontsize=12, fontweight="bold", pad=6, loc="left")
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    ax.xaxis.set_major_formatter(DATE_FMT)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, color=MUTED, fontsize=8)
    ax.grid(axis="y", color="#2d2d44", linewidth=0.5)
    ax.set_xlim(df["date"].min(), df["date"].max())

def latest_label(ax, x, y, color, fmt="{:.0f}"):
    if pd.isna(y): return
    ax.annotate(fmt.format(y), xy=(x, y), xytext=(6, 0),
                textcoords="offset points", color=color,
                fontsize=9, fontweight="bold", va="center")

def no_data(ax, msg="No data"):
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            color=MUTED, transform=ax.transAxes, fontsize=10)

def rolling(series, w=7):
    return series.rolling(w, min_periods=1).mean()

@st.cache_data(ttl=300)
def build_figure(df_json: str) -> bytes:
    """Build the matplotlib figure and return as PNG bytes."""
    df = pd.read_json(io.StringIO(df_json))
    df["date"] = pd.to_datetime(df["date"])

    fig = plt.figure(figsize=(14, 24), facecolor=BG)
    gs  = gridspec.GridSpec(7, 1, figure=fig, hspace=0.6,
                            left=0.08, right=0.97, top=0.97, bottom=0.03)

    # 1 · HRV ─────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    style_ax(ax1, "💜  HRV — Heart Rate Variability", "ms", HRV_C)

    if df["hrv_avg"].notna().any():
        bv = df["hrv_band_mid"].notna()
        ax1.fill_between(df["date"][bv], df["hrv_band_low"][bv], df["hrv_band_high"][bv],
                         alpha=0.18, color=HRV_C, label="Normal range (±1.5 SD)")
        ax1.plot(df["date"][bv], df["hrv_band_high"][bv], color=HRV_C, lw=0.9, ls="--", alpha=0.55)
        ax1.plot(df["date"][bv], df["hrv_band_low"][bv],  color=HRV_C, lw=0.9, ls="--", alpha=0.55)
        hv = df["hrv_avg"].notna()
        ax1.plot(df["date"][hv], df["hrv_avg"][hv], color=HRV_C, lw=2, label="HRV weekly avg")
        for _, row in df[hv].iterrows():
            v, mid = row["hrv_avg"], row["hrv_band_mid"]
            sd = row["hrv_band_std"] if pd.notna(row["hrv_band_std"]) else 0
            pt = HRV_C if pd.isna(mid) else ("#4ade80" if v > mid + 1.5*sd else (HR_C if v < mid - 1.5*sd else HRV_C))
            ax1.plot(row["date"], v, "o", color=pt, markersize=5, zorder=5)
        latest_label(ax1, df["date"][hv].iloc[-1], df["hrv_avg"][hv].iloc[-1], HRV_C)
        if df["hrv_band_high"].notna().any():
            hi, lo = df["hrv_band_high"].dropna().iloc[-1], df["hrv_band_low"].dropna().iloc[-1]
            xmax = df["date"].max()
            ax1.text(xmax, hi, f" ↑{hi:.0f}", color=HRV_C, fontsize=8, va="bottom", ha="left", alpha=0.8)
            ax1.text(xmax, lo, f" ↓{lo:.0f}", color=HRV_C, fontsize=8, va="top",    ha="left", alpha=0.8)
        ax1.legend(fontsize=9, labelcolor=MUTED, facecolor=CARD, edgecolor="none", loc="upper left")
    else:
        no_data(ax1, "No HRV data")

    # 2 · Resting HR ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    style_ax(ax2, "❤️   Resting Heart Rate", "bpm", HR_C)

    if df["resting_hr"].notna().any():
        bv = df["rhr_band_mid"].notna()
        ax2.fill_between(df["date"][bv], df["rhr_band_low"][bv], df["rhr_band_high"][bv],
                         alpha=0.18, color=HR_C, label="Normal range (±1.5 SD)")
        ax2.plot(df["date"][bv], df["rhr_band_high"][bv], color=HR_C, lw=0.9, ls="--", alpha=0.55)
        ax2.plot(df["date"][bv], df["rhr_band_low"][bv],  color=HR_C, lw=0.9, ls="--", alpha=0.55)
        rv = df["resting_hr"].notna()
        ax2.plot(df["date"][rv], df["resting_hr"][rv], color=HR_C, lw=2, label="Resting HR")
        for _, row in df[rv].iterrows():
            v, mid = row["resting_hr"], row["rhr_band_mid"]
            sd = row["rhr_band_std"] if pd.notna(row["rhr_band_std"]) else 0
            pt = HR_C if pd.isna(mid) else ("#f97316" if v > mid + 1.5*sd else ("#4ade80" if v < mid - 1.5*sd else HR_C))
            ax2.plot(row["date"], v, "o", color=pt, markersize=5, zorder=5)
        latest_label(ax2, df["date"][rv].iloc[-1], df["resting_hr"][rv].iloc[-1], HR_C, "{:.0f} bpm")
        if df["rhr_band_high"].notna().any():
            hi, lo = df["rhr_band_high"].dropna().iloc[-1], df["rhr_band_low"].dropna().iloc[-1]
            xmax = df["date"].max()
            ax2.text(xmax, hi, f" ↑{hi:.0f}", color=HR_C, fontsize=8, va="bottom", ha="left", alpha=0.8)
            ax2.text(xmax, lo, f" ↓{lo:.0f}", color=HR_C, fontsize=8, va="top",    ha="left", alpha=0.8)
        ax2.legend(fontsize=9, labelcolor=MUTED, facecolor=CARD, edgecolor="none", loc="upper left")
    else:
        no_data(ax2)

    # 3 · Sleep ───────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    style_ax(ax3, "😴  Sleep Duration", "hours", SLP_C)
    if df["sleep_hours"].notna().any():
        bc = df["sleep_hours"].apply(
            lambda v: SLP_C if (not pd.isna(v) and v >= 7)
                      else "#fbbf24" if (not pd.isna(v) and v >= 6)
                      else HR_C if not pd.isna(v) else MUTED)
        ax3.bar(df["date"], df["sleep_hours"], width=0.8, color=bc, alpha=0.85)
        ax3.plot(df["date"][df["sleep_hours"].notna()],
                 rolling(df["sleep_hours"].dropna()),
                 color=SLP_C, lw=2, label="7-day avg")
        ax3.axhline(8, color=SLP_C,     ls="--", lw=0.8, alpha=0.5, label="Target 8 h")
        ax3.axhline(7, color="#fbbf24", ls="--", lw=0.8, alpha=0.5, label="Min 7 h")
        ax3.set_ylim(5, 10)
        ax3.legend(fontsize=9, labelcolor=MUTED, facecolor=CARD, edgecolor="none", loc="upper left")
    else:
        no_data(ax3)

    # 4 · Daily TSS ───────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    style_ax(ax4, "🚴  Daily TSS", "TSS", TSS_C)

    def tss_bar_color(row):
        m = row.get("tss_methods")
        if isinstance(m, list):
            if "TSS"   in m: return TSS_C
            if "hrTSS" in m: return "#fbbf24"
        return MUTED

    ax4.bar(df["date"], df["daily_tss"], width=0.8,
            color=df.apply(tss_bar_color, axis=1), alpha=0.8)
    thresh = df["daily_tss"].quantile(0.75)
    for _, row in df[df["daily_tss"] > thresh].iterrows():
        acts = row.get("activities") or []
        if acts and isinstance(acts, list):
            lbl = (acts[0].get("type") or acts[0].get("name") or "")[:10]
            ax4.text(row["date"], row["daily_tss"] + df["daily_tss"].max() * 0.02,
                     lbl, ha="center", va="bottom", color=WHITE, fontsize=7, alpha=0.7)
    ax4.legend(handles=[
        mpatches.Patch(color=TSS_C,    label="TSS (power)"),
        mpatches.Patch(color="#fbbf24", label="hrTSS (HR)"),
        mpatches.Patch(color=MUTED,    label="Rest day"),
    ], fontsize=9, labelcolor=MUTED, facecolor=CARD, edgecolor="none", loc="upper left")

    # 5 · ATL ─────────────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[4])
    style_ax(ax5, f"🔥  ATL — Fatigue  ({ATL_DAYS}-day avg)", "load", ATL_C)
    ax5.fill_between(df["date"], df["atl"], alpha=0.2, color=ATL_C)
    ax5.plot(df["date"], df["atl"], color=ATL_C, lw=2)
    latest_label(ax5, df["date"].iloc[-1], df["atl"].iloc[-1], ATL_C)
    ax5.text(0.01, 0.92, "High = recently training hard.  Drops fast with rest.",
             transform=ax5.transAxes, color=MUTED, fontsize=8, va="top")

    # 6 · CTL ─────────────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[5])
    style_ax(ax6, f"📈  CTL — Fitness  ({CTL_DAYS}-day avg)", "load", CTL_C)
    ax6.fill_between(df["date"], df["ctl"], alpha=0.2, color=CTL_C)
    ax6.plot(df["date"], df["ctl"], color=CTL_C, lw=2)
    latest_label(ax6, df["date"].iloc[-1], df["ctl"].iloc[-1], CTL_C)
    ax6.text(0.01, 0.92, "Rises slowly with consistent training.  Falls slowly with inactivity.",
             transform=ax6.transAxes, color=MUTED, fontsize=8, va="top")

    # 7 · TSB ─────────────────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[6])
    style_ax(ax7, "⚡  TSB — Form  (CTL − ATL)", "TSB", WHITE)
    tsb = df["tsb"]
    ax7.fill_between(df["date"], tsb, 0, where=(tsb >= 0), alpha=0.25,
                     color="#4ade80", interpolate=True, label="Fresh")
    ax7.fill_between(df["date"], tsb, 0, where=(tsb <  0), alpha=0.25,
                     color=HR_C,     interpolate=True, label="Fatigued")
    ax7.plot(df["date"], tsb, color=WHITE, lw=2)
    ax7.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax7.axhspan( 15,  25, alpha=0.06, color="#4ade80")
    ax7.axhspan(-30, -20, alpha=0.06, color=HR_C)
    ax7.text(df["date"].max(), 20,  "Peak form",   color="#4ade80", fontsize=8, va="center", ha="right")
    ax7.text(df["date"].max(), -25, "Heavy block", color=HR_C,     fontsize=8, va="center", ha="right")
    latest_label(ax7, df["date"].iloc[-1], tsb.iloc[-1], WHITE, "{:+.0f}")
    ax7.legend(fontsize=9, labelcolor=MUTED, facecolor=CARD, edgecolor="none", loc="upper left")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# Render charts
chart_bytes = build_figure(df.to_json())
st.image(chart_bytes, use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.caption("🚴 Garmin Training Dashboard · Data via Garmin Connect API")
