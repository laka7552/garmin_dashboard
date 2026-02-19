"""
Garmin – Training & Recovery Dashboard (Individual Panels)
===========================================================
8 individual panels:
  1. HRV  (with adaptive ±1 SD bands)
  2. Resting Heart Rate
  3. Sleep Duration
  4. Daily TSS
  5. ATL – Fatigue
  6. CTL – Fitness
  7. TSB – Form
  8. Today's Snapshot (today vs 7-day avg + ATL/CTL/TSB + interpretation)

Requirements:
    pip install matplotlib pandas numpy
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

# ── Load ──────────────────────────────────────────────────────────────────────
DATA_FILE = "garmin_history.json"
if not os.path.exists(DATA_FILE):
    raise SystemExit(f"'{DATA_FILE}' not found – run garmin_fetch_history.py first.")

with open(DATA_FILE, encoding="utf-8") as f:
    raw = json.load(f)

df = pd.DataFrame(raw)
df["date"]        = pd.to_datetime(df["date"])
df                = df.sort_values("date").reset_index(drop=True)
df["sleep_hours"] = pd.to_numeric(df["sleep_seconds"],  errors="coerce") / 3600
df["daily_tss"]   = pd.to_numeric(df["daily_tss"],      errors="coerce").fillna(0)
df["hrv"]         = pd.to_numeric(df["hrv_last_night"],  errors="coerce")
df["hrv_avg"]     = pd.to_numeric(df["hrv_weekly_avg"],  errors="coerce")
df["sleep_hr"]    = pd.to_numeric(df["sleep_avg_hr"],    errors="coerce")
df["resting_hr"]  = pd.to_numeric(df["resting_hr"],      errors="coerce")

# ── ATL / CTL / TSB ───────────────────────────────────────────────────────────
ATL_DAYS  = 7
CTL_DAYS  = 42
atl_alpha = 2 / (ATL_DAYS + 1)
ctl_alpha = 2 / (CTL_DAYS + 1)

atl, ctl = [0.0], [0.0]
for tss in df["daily_tss"]:
    atl.append(atl[-1] + atl_alpha * (tss - atl[-1]))
    ctl.append(ctl[-1] + ctl_alpha * (tss - ctl[-1]))

df["atl"] = atl[1:]
df["ctl"] = ctl[1:]
df["tsb"] = df["ctl"] - df["atl"]

# Adaptive HRV bands (rolling 7-day mean ± 1.5 SD) ───────────────────────────
HRV_WINDOW          = 7
df["hrv_band_mid"]  = df["hrv_avg"].rolling(HRV_WINDOW, min_periods=3).mean()
df["hrv_band_std"]  = df["hrv_avg"].rolling(HRV_WINDOW, min_periods=3).std()
df["hrv_band_high"] = df["hrv_band_mid"] + 1.5 * df["hrv_band_std"]
df["hrv_band_low"]  = (df["hrv_band_mid"] - 1.5 * df["hrv_band_std"]).clip(lower=0)

# Adaptive Resting HR bands (rolling 7-day mean ± 1.5 SD) ────────────────────
RHR_WINDOW           = 7
df["rhr_band_mid"]   = df["resting_hr"].rolling(RHR_WINDOW, min_periods=3).mean()
df["rhr_band_std"]   = df["resting_hr"].rolling(RHR_WINDOW, min_periods=3).std()
df["rhr_band_high"]  = df["rhr_band_mid"] + 1.5 * df["rhr_band_std"]
df["rhr_band_low"]   = (df["rhr_band_mid"] - 1.5 * df["rhr_band_std"]).clip(lower=0)

# ── Style ─────────────────────────────────────────────────────────────────────
BG      = "#0f0f1a"
CARD    = "#1c1c2e"
HRV_C   = "#a78bfa"
HR_C    = "#f87171"
SLP_C   = "#60a5fa"
TSS_C   = "#34d399"
ATL_C   = "#fb923c"
CTL_C   = "#38bdf8"
WHITE   = "#f1f5f9"
MUTED   = "#64748b"
DATE_FMT = mdates.DateFormatter("%d %b")

def rolling(series, w=7):
    return series.rolling(w, min_periods=1).mean()

def style_ax(ax, title, ylabel, color):
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_color("#2d2d44")
    ax.tick_params(colors=MUTED, labelsize=10)          # CHANGE 1 – larger ticks
    ax.set_title(title, color=color, fontsize=13, fontweight="bold", pad=6, loc="left")  # CHANGE 1
    ax.set_ylabel(ylabel, color=MUTED, fontsize=11)     # CHANGE 1
    ax.xaxis.set_major_formatter(DATE_FMT)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, color=MUTED, fontsize=9)  # CHANGE 1
    ax.grid(axis="y", color="#2d2d44", linewidth=0.5)
    ax.set_xlim(df["date"].min(), df["date"].max())

def latest_label(ax, x, y, color, fmt="{:.0f}"):
    if pd.isna(y):
        return
    ax.annotate(fmt.format(y), xy=(x, y),
                xytext=(6, 0), textcoords="offset points",
                color=color, fontsize=10, fontweight="bold", va="center")  # CHANGE 1

def no_data(ax, msg="No data available"):
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            color=MUTED, transform=ax.transAxes, fontsize=11)  # CHANGE 1

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 28), facecolor=BG)        # CHANGE 1 – slightly wider/taller
fig.suptitle("Training & Recovery – Last 4 Weeks",
             fontsize=18, color=WHITE, fontweight="bold", y=0.99)  # CHANGE 1

gs = gridspec.GridSpec(8, 1, figure=fig, hspace=0.6,
                       left=0.08, right=0.97, top=0.97, bottom=0.03)


# ── 1 · HRV ──────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
style_ax(ax1, "💜  HRV — Heart Rate Variability", "ms", HRV_C)

if df["hrv_avg"].notna().any():
    band_valid = df["hrv_band_mid"].notna()

    # CHANGE – shaded ±2 SD band
    ax1.fill_between(df["date"][band_valid],
                     df["hrv_band_low"][band_valid],
                     df["hrv_band_high"][band_valid],
                     alpha=0.18, color=HRV_C, label="Normal range (±1.5 SD)")

    # Band edge dashed lines
    ax1.plot(df["date"][band_valid], df["hrv_band_high"][band_valid],
             color=HRV_C, linewidth=0.9, linestyle="--", alpha=0.55)
    ax1.plot(df["date"][band_valid], df["hrv_band_low"][band_valid],
             color=HRV_C, linewidth=0.9, linestyle="--", alpha=0.55)

    # Main HRV line
    hrv_valid = df["hrv_avg"].notna()
    ax1.plot(df["date"][hrv_valid], df["hrv_avg"][hrv_valid],
             color=HRV_C, linewidth=2, label="HRV weekly avg")

    # Colour individual dots by zone (outside ±2 SD)
    for _, row in df[hrv_valid].iterrows():
        v   = row["hrv_avg"]
        mid = row["hrv_band_mid"]
        sd  = row["hrv_band_std"] if pd.notna(row["hrv_band_std"]) else 0
        if pd.isna(mid):
            pt_col = HRV_C
        elif v > mid + 2 * sd:
            pt_col = "#4ade80"   # above ±2 SD → green
        elif v < mid - 2 * sd:
            pt_col = HR_C        # below ±2 SD → red
        else:
            pt_col = HRV_C       # within normal → purple
        ax1.plot(row["date"], v, "o", color=pt_col, markersize=5, zorder=5)

    latest_label(ax1, df["date"][hrv_valid].iloc[-1],
                 df["hrv_avg"][hrv_valid].iloc[-1], HRV_C)

    # CHANGE 3 – current band values on right edge
    if df["hrv_band_high"].notna().any():
        hi   = df["hrv_band_high"].dropna().iloc[-1]
        lo   = df["hrv_band_low"].dropna().iloc[-1]
        xmax = df["date"].max()
        ax1.text(xmax, hi, f" ↑{hi:.0f}", color=HRV_C, fontsize=9,
                 va="bottom", ha="left", alpha=0.8)
        ax1.text(xmax, lo, f" ↓{lo:.0f}", color=HRV_C, fontsize=9,
                 va="top",    ha="left", alpha=0.8)

    ax1.legend(fontsize=10, labelcolor=MUTED, facecolor=CARD, edgecolor="none", loc="upper left")
else:
    no_data(ax1, "No HRV data — device may not support it")


# ── 2 · Resting Heart Rate ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
style_ax(ax2, "❤️   Resting Heart Rate", "bpm", HR_C)

if df["resting_hr"].notna().any():
    rhr_band_valid = df["rhr_band_mid"].notna()

    # Shaded ±1 SD band
    ax2.fill_between(df["date"][rhr_band_valid],
                     df["rhr_band_low"][rhr_band_valid],
                     df["rhr_band_high"][rhr_band_valid],
                     alpha=0.18, color=HR_C, label="Normal range (±1.5 SD)")

    # Band edge dashed lines
    ax2.plot(df["date"][rhr_band_valid], df["rhr_band_high"][rhr_band_valid],
             color=HR_C, linewidth=0.9, linestyle="--", alpha=0.55)
    ax2.plot(df["date"][rhr_band_valid], df["rhr_band_low"][rhr_band_valid],
             color=HR_C, linewidth=0.9, linestyle="--", alpha=0.55)

    # Main resting HR line with coloured dots by zone
    rhr_valid = df["resting_hr"].notna()
    ax2.plot(df["date"][rhr_valid], df["resting_hr"][rhr_valid],
             color=HR_C, linewidth=2, label="Resting HR")

    for _, row in df[rhr_valid].iterrows():
        v   = row["resting_hr"]
        mid = row["rhr_band_mid"]
        sd  = row["rhr_band_std"] if pd.notna(row["rhr_band_std"]) else 0
        if pd.isna(mid):
            pt_col = HR_C
        elif v > mid + 2 * sd:
            pt_col = "#f97316"   # above ±2 SD → elevated (orange warning)
        elif v < mid - 2 * sd:
            pt_col = "#4ade80"   # below ±2 SD → very well recovered (green)
        else:
            pt_col = HR_C        # within normal → red
        ax2.plot(row["date"], v, "o", color=pt_col, markersize=5, zorder=5)

    latest_label(ax2,
                 df["date"][rhr_valid].iloc[-1],
                 df["resting_hr"][rhr_valid].iloc[-1], HR_C, "{:.0f} bpm")

    # Current band values on right edge
    if df["rhr_band_high"].notna().any():
        hi   = df["rhr_band_high"].dropna().iloc[-1]
        lo   = df["rhr_band_low"].dropna().iloc[-1]
        xmax = df["date"].max()
        ax2.text(xmax, hi, f" ↑{hi:.0f}", color=HR_C, fontsize=9,
                 va="bottom", ha="left", alpha=0.8)
        ax2.text(xmax, lo, f" ↓{lo:.0f}", color=HR_C, fontsize=9,
                 va="top",    ha="left", alpha=0.8)

    ax2.legend(fontsize=10, labelcolor=MUTED, facecolor=CARD, edgecolor="none", loc="upper left")
else:
    no_data(ax2)


# ── 3 · Sleep Duration ───────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
style_ax(ax3, "😴  Sleep Duration", "hours", SLP_C)

if df["sleep_hours"].notna().any():
    bar_colors = df["sleep_hours"].apply(
        lambda v: SLP_C if (not pd.isna(v) and v >= 7)
                  else "#fbbf24" if (not pd.isna(v) and v >= 6)
                  else HR_C if not pd.isna(v) else MUTED)
    ax3.bar(df["date"], df["sleep_hours"], width=0.8, color=bar_colors, alpha=0.85)
    ax3.plot(df["date"][df["sleep_hours"].notna()],
             rolling(df["sleep_hours"].dropna()),
             color=SLP_C, linewidth=2, label="7-day avg")
    ax3.axhline(8, color=SLP_C,    linestyle="--", linewidth=0.8, alpha=0.5, label="Target 8 h")
    ax3.axhline(7, color="#fbbf24", linestyle="--", linewidth=0.8, alpha=0.5, label="Min 7 h")
    ax3.set_ylim(5, 10)
    ax3.legend(fontsize=10, labelcolor=MUTED, facecolor=CARD, edgecolor="none", loc="upper left")
else:
    no_data(ax3)


# ── 4 · Daily TSS ────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[3])
style_ax(ax4, "🚴  Daily TSS", "TSS", TSS_C)

def bar_color(row):
    methods = row.get("tss_methods")
    if isinstance(methods, list):
        if "TSS"   in methods: return TSS_C
        if "hrTSS" in methods: return "#fbbf24"
    return MUTED

bar_colors = df.apply(bar_color, axis=1)
ax4.bar(df["date"], df["daily_tss"], width=0.8, color=bar_colors, alpha=0.8)

tss_threshold = df["daily_tss"].quantile(0.75)
for _, row in df[df["daily_tss"] > tss_threshold].iterrows():
    acts = row.get("activities") or []
    if acts and isinstance(acts, list):
        label = (acts[0].get("type") or acts[0].get("name") or "")[:10]
        ax4.text(row["date"], row["daily_tss"] + df["daily_tss"].max() * 0.02,
                 label, ha="center", va="bottom",
                 color=WHITE, fontsize=8, alpha=0.7)

tss_patch   = mpatches.Patch(color=TSS_C,    label="TSS (power)")
hrtss_patch = mpatches.Patch(color="#fbbf24", label="hrTSS (HR)")
rest_patch  = mpatches.Patch(color=MUTED,    label="Rest day")
ax4.legend(handles=[tss_patch, hrtss_patch, rest_patch],
           fontsize=10, labelcolor=MUTED, facecolor=CARD, edgecolor="none", loc="upper left")


# ── 5 · ATL – Fatigue ────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[4])
style_ax(ax5, f"🔥  ATL — Fatigue  ({ATL_DAYS}-day avg)", "load", ATL_C)

ax5.fill_between(df["date"], df["atl"], alpha=0.2, color=ATL_C)
ax5.plot(df["date"], df["atl"], color=ATL_C, linewidth=2)
latest_label(ax5, df["date"].iloc[-1], df["atl"].iloc[-1], ATL_C)

ax5.text(0.01, 0.92,
         "High = recently training hard.  Drops fast with rest.",
         transform=ax5.transAxes, color=MUTED, fontsize=9, va="top")


# ── 6 · CTL – Fitness ────────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[5])
style_ax(ax6, f"📈  CTL — Fitness  ({CTL_DAYS}-day avg)", "load", CTL_C)

ax6.fill_between(df["date"], df["ctl"], alpha=0.2, color=CTL_C)
ax6.plot(df["date"], df["ctl"], color=CTL_C, linewidth=2)
latest_label(ax6, df["date"].iloc[-1], df["ctl"].iloc[-1], CTL_C)

ax6.text(0.01, 0.92,
         "Rises slowly with consistent training.  Falls slowly with inactivity.",
         transform=ax6.transAxes, color=MUTED, fontsize=9, va="top")


# ── 7 · TSB – Form ───────────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[6])
style_ax(ax7, "⚡  TSB — Form  (CTL − ATL)", "TSB", WHITE)

tsb = df["tsb"]
ax7.fill_between(df["date"], tsb, 0,
                 where=(tsb >= 0), alpha=0.25, color="#4ade80", interpolate=True, label="Fresh")
ax7.fill_between(df["date"], tsb, 0,
                 where=(tsb < 0),  alpha=0.25, color=HR_C,     interpolate=True, label="Fatigued")
ax7.plot(df["date"], tsb, color=WHITE, linewidth=2)
ax7.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")

ax7.axhspan( 15,  25, alpha=0.06, color="#4ade80")
ax7.axhspan(-30, -20, alpha=0.06, color=HR_C)

ymin, ymax = ax7.get_ylim()
ax7.text(df["date"].max(), 20,  "Peak form",    color="#4ade80", fontsize=9, va="center", ha="right")
ax7.text(df["date"].max(), -25, "Heavy block",  color=HR_C,     fontsize=9, va="center", ha="right")

latest_label(ax7, df["date"].iloc[-1], tsb.iloc[-1], WHITE, "{:+.0f}")
ax7.legend(fontsize=10, labelcolor=MUTED, facecolor=CARD, edgecolor="none", loc="upper left")


# ── 8 · Today's Snapshot + Interpretation ────────────────────────────────────
ax8 = fig.add_subplot(gs[7])
ax8.set_facecolor(CARD)
for spine in ax8.spines.values():
    spine.set_color("#2d2d44")
ax8.set_title("📊  Current vs Normal Range  ·  Current Load  ·  Interpretation",
              color=WHITE, fontsize=13, fontweight="bold", pad=8, loc="left")
ax8.tick_params(colors=MUTED, labelsize=10)
ax8.set_xlim(0, 10)
ax8.set_ylim(0, 10)
ax8.set_xticks([])
ax8.set_yticks([])

# ── Helper ────────────────────────────────────────────────────────────────────
def val(series):
    v = series.dropna()
    return v.iloc[-1] if not v.empty else None

def avg7(series):
    v = series.iloc[-7:].dropna()
    return round(v.mean(), 1) if not v.empty else None

ax8_gauge = ax8.inset_axes([0.01, 0.05, 0.36, 0.92])
ax8_gauge.set_facecolor(CARD)
for spine in ax8_gauge.spines.values():
    spine.set_visible(False)
ax8_gauge.set_xticks([])
ax8_gauge.set_yticks([])
ax8_gauge.set_xlim(0, 1)
ax8_gauge.set_ylim(0, 1)

def draw_gauge(ax, y_center, label, current, band_lo, band_mid, band_hi, color, unit=""):
    """
    Horizontal gauge:  grey track  |  coloured band  |  diamond marker
    y_center, band values and current are all in their own units.
    The axes x range is always 0–1 (normalised per gauge row).
    Labels are placed at fixed small offsets from y_center.
    """
    TRACK_H  = 0.07   # height of the bar in y-data units (axes ylim 0–1)
    VAL_OFF  = 0.09   # y offset above bar for value label
    BND_OFF  = 0.09   # y offset below bar for band labels

    if current is None or band_lo is None or band_hi is None:
        ax.text(0.5, y_center, "No data", ha="center", va="center",
                color=MUTED, fontsize=9)
        return

    # Build a per-row x normalisation with padding
    all_vals = [current, band_lo, band_hi]
    span     = max(all_vals) - min(all_vals)
    pad      = max(span * 0.45, 2)
    x_min    = min(all_vals) - pad
    x_max    = max(all_vals) + pad

    def norm(v):
        return (v - x_min) / (x_max - x_min)

    # ── Track ──────────────────────────────────────────────────────
    ax.barh(y_center, 1.0, left=0.0, height=TRACK_H,
            color="#2d2d44", align="center", zorder=1)

    # ── Coloured band zone ─────────────────────────────────────────
    bw = norm(band_hi) - norm(band_lo)
    ax.barh(y_center, bw, left=norm(band_lo), height=TRACK_H,
            color=color, alpha=0.32, align="center", zorder=2)

    # ── Mid-line ───────────────────────────────────────────────────
    ax.plot([norm(band_mid), norm(band_mid)],
            [y_center - TRACK_H * 0.65, y_center + TRACK_H * 0.65],
            color=color, linewidth=1.2, alpha=0.65, zorder=3)

    # ── Diamond marker ─────────────────────────────────────────────
    cx = norm(current)
    if   current > band_hi: marker_col = "#f97316" if color == HR_C else "#4ade80"
    elif current < band_lo: marker_col = "#4ade80" if color == HR_C else HR_C
    else:                   marker_col = color

    ax.plot(cx, y_center, "D", color=marker_col, markersize=8,
            markeredgecolor=WHITE, markeredgewidth=0.7, zorder=5)

    # ── Metric name (left of track) ────────────────────────────────
    ax.text(-0.01, y_center, label, ha="right", va="center",
            color=WHITE, fontsize=9, fontweight="bold")

    # ── Current value (above marker, small fixed offset) ──────────
    # Clamp x so label stays inside axes
    lx = min(max(cx, 0.05), 0.95)
    ax.text(lx, y_center + VAL_OFF,
            f"{current:.0f} {unit}", ha="center", va="bottom",
            color=marker_col, fontsize=9, fontweight="bold")

    # ── Band boundary labels (below track, at band edges) ─────────
    ax.text(norm(band_lo), y_center - BND_OFF,
            f"{band_lo:.0f}", ha="center", va="top",
            color=MUTED, fontsize=7.5)
    ax.text(norm(band_hi), y_center - BND_OFF,
            f"{band_hi:.0f}", ha="center", va="top",
            color=MUTED, fontsize=7.5)

hrv_cur  = val(df["hrv_avg"])
hrv_lo   = val(df["hrv_band_low"])
hrv_mid  = val(df["hrv_band_mid"])
hrv_hi   = val(df["hrv_band_high"])

rhr_cur  = val(df["resting_hr"])
rhr_lo   = val(df["rhr_band_low"])
rhr_mid  = val(df["rhr_band_mid"])
rhr_hi   = val(df["rhr_band_high"])

slp_cur  = val(df["sleep_hours"])

draw_gauge(ax8_gauge, y_center=0.75,
           label="HRV",     current=hrv_cur, band_lo=hrv_lo, band_mid=hrv_mid,
           band_hi=hrv_hi,  color=HRV_C, unit="ms")

draw_gauge(ax8_gauge, y_center=0.42,
           label="Rest HR", current=rhr_cur, band_lo=rhr_lo, band_mid=rhr_mid,
           band_hi=rhr_hi,  color=HR_C, unit="bpm")

# Sleep: fixed reference band (5h–9.5h, target 8h) — wide enough to always contain current
slp_lo, slp_mid, slp_hi = 6.0, 8.0, 9.0
draw_gauge(ax8_gauge, y_center=0.10,
           label="Sleep",   current=slp_cur if slp_cur is not None else 0,
           band_lo=slp_lo,  band_mid=slp_mid,
           band_hi=slp_hi,  color=SLP_C, unit="h")

# ── Middle: ATL / CTL / TSB cards ─────────────────────────────────────────────
atl_now   = df["atl"].iloc[-1]
ctl_now   = df["ctl"].iloc[-1]
tsb_now   = df["tsb"].iloc[-1]
tsb_color = "#4ade80" if tsb_now >= 0 else HR_C

cards = [
    (f"{atl_now:.0f}", "ATL", "Fatigue", ATL_C,    0.41),
    (f"{ctl_now:.0f}", "CTL", "Fitness", CTL_C,    0.54),
    (f"{tsb_now:+.0f}", "TSB", "Form",  tsb_color, 0.67),
]
for val_str, abbr, label, col, cx in cards:
    ax8.text(cx, 0.78, val_str, transform=ax8.transAxes,
             ha="center", va="center", fontsize=22, fontweight="bold", color=col)
    ax8.text(cx, 0.58, abbr,   transform=ax8.transAxes,
             ha="center", va="center", fontsize=10,  fontweight="bold", color=col)  # CHANGE 1
    ax8.text(cx, 0.46, label,  transform=ax8.transAxes,
             ha="center", va="center", fontsize=9,  color=MUTED)                    # CHANGE 1

ax8.axvline(x=3.8, color="#2d2d44", linewidth=1)
ax8.axvline(x=7.2, color="#2d2d44", linewidth=1)

# ── Right: Interpretation ─────────────────────────────────────────────────────
def interpret(tsb, atl, ctl, hrv_today, hrv_high, hrv_low, rhr_today, rhr_high, rhr_low):
    if tsb > 15:
        if atl < ctl * 0.6:
            ts = ("Tapering / Undertraining", "#fbbf24",
                  "TSB high but ATL well below CTL.\nFitness may be declining — add load.")
        else:
            ts = ("Peak Form", "#4ade80",
                  "Well rested with solid fitness base.\nIdeal for a key effort or race.")
    elif tsb >= 0:
        ts = ("Maintaining", "#4ade80",
              "Load balanced, fitness stable.\nGood for consistent steady training.")
    elif tsb >= -10:
        ts = ("Productive Training", CTL_C,
              "Moderate fatigue, fitness building.\nNormal and healthy training block.")
    elif tsb >= -20:
        ts = ("Accumulating Fatigue", "#fbbf24",
              "Significant fatigue building up.\nMonitor sleep and recovery closely.")
    elif tsb >= -30:
        ts = ("Heavy Training Block", "#f97316",
              "High fatigue load. Prioritise\nsleep, nutrition and easy sessions.")
    else:
        ts = ("Overreaching Risk", HR_C,
              "Very high fatigue. Take 1–2 rest\nor very easy days immediately.")

    signals = []
    # Use band boundaries instead of simple 7-day avg ratio
    if hrv_today and pd.notna(hrv_high) and pd.notna(hrv_low) and hrv_high > 0:
        if hrv_today > hrv_high:
            signals.append(("HRV above normal range — nervous system well recovered", True))
        elif hrv_today < hrv_low:
            signals.append(("HRV below normal range — still recovering", False))
        else:
            signals.append(("HRV within normal range", True))

    if rhr_today and pd.notna(rhr_high) and pd.notna(rhr_low) and rhr_high > 0:
        if rhr_today > rhr_high:
            signals.append(("Resting HR above normal range — fatigue or stress", False))
        elif rhr_today < rhr_low:
            signals.append(("Resting HR below normal range — very well recovered", True))
        else:
            signals.append(("Resting HR within normal range", True))

    good  = sum(1 for _, ok in signals if ok)
    total = len(signals)
    if total == 0:
        rl, rc = "No data", MUTED
    elif good == total:
        rl, rc = "Good Recovery", "#4ade80"
    elif good >= total / 2:
        rl, rc = "Moderate Recovery", "#fbbf24"
    else:
        rl, rc = "Poor Recovery", HR_C

    return ts, (rl, rc, signals)

hrv_t    = val(df["hrv_avg"])
hrv_hi   = val(df["hrv_band_high"])
hrv_lo   = val(df["hrv_band_low"])
rhr_t    = val(df["resting_hr"])
rhr_hi   = val(df["rhr_band_high"])
rhr_lo   = val(df["rhr_band_low"])

(t_label, t_color, t_detail), (r_label, r_color, r_signals) = interpret(
    tsb_now, atl_now, ctl_now, hrv_t, hrv_hi, hrv_lo, rhr_t, rhr_hi, rhr_lo)

# CHANGE 2 – interpretation column shifted further right
rx = 0.84

ax8.text(rx, 0.92, "Training Status", transform=ax8.transAxes,
         ha="center", va="center", fontsize=10, color=MUTED)           # CHANGE 1
ax8.text(rx, 0.78, t_label, transform=ax8.transAxes,
         ha="center", va="center", fontsize=11, fontweight="bold", color=t_color)  # CHANGE 1
for line_i, line in enumerate(t_detail.split("\n")):
    ax8.text(rx, 0.63 - line_i * 0.13, line, transform=ax8.transAxes,
             ha="center", va="center", fontsize=9, color=MUTED, style="italic")    # CHANGE 1

ax8.text(rx, 0.37, "Recovery Status", transform=ax8.transAxes,
         ha="center", va="center", fontsize=10, color=MUTED)           # CHANGE 1
ax8.text(rx, 0.25, r_label, transform=ax8.transAxes,
         ha="center", va="center", fontsize=11, fontweight="bold", color=r_color)  # CHANGE 1
for sig_i, (sig_text, ok) in enumerate(r_signals):
    sig_col = "#4ade80" if ok else HR_C
    ax8.text(rx, 0.12 - sig_i * 0.13, f"{'↑' if ok else '↓'} {sig_text}",
             transform=ax8.transAxes,
             ha="center", va="center", fontsize=9, color=sig_col, style="italic")  # CHANGE 1


out     = "garmin_weekly_dashboard.png"
out_pdf = "garmin_weekly_dashboard.pdf"
plt.savefig(out,     dpi=150, bbox_inches="tight", facecolor=BG)
plt.savefig(out_pdf, bbox_inches="tight", facecolor=BG)
print(f"✓ Saved → '{out}'")
print(f"✓ Saved → '{out_pdf}'")
plt.show()
