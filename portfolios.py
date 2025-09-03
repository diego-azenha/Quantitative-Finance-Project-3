# mean_variance_two_plots_with_labels.py
# -----------------------------------------------------------------------------
# Build FACTOR covariance (FF5+MOM) and SAMPLE covariance, compute:
#   • Global Minimum-Variance (GMV) and Tangency portfolios for BOTH
# Make TWO separate charts (one per model) with:
#   • Efficient Frontier in BLACK (solid)
#   • CAL dashed in model color
#   • GMV/Tangency markers with DATA LABELS (σ, μ, Sharpe)
#   • Frontier extends at least to the Tangency volatility on the x-axis
# Print a compact (annualized) GMV performance comparison.
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os

# --------------------- Paths & constants ---------------------
ROOT = Path(__file__).resolve().parent
CDIR = ROOT / "clean_data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FACTORS_CSV = CDIR / "ff_factors_daily_clean.csv"      # MKT_RF, SMB, HML, RMW, CMA, RF, MOM (decimals)
PRICES_PARQUET = CDIR / "clean_stock_prices.parquet"   # wide: index=Date or column "Date"
START = pd.Timestamp("2010-01-01")
K_FACTORS = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
N_STOCKS = 30
TRADING_DAYS = 252

# --------------------- Plot aesthetics ----------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#dddddd",
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.grid.axis": "both",
    "font.family": "DejaVu Sans",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.facecolor": "white",
    "legend.edgecolor": "#dddddd",
})

# Model colors (used for CAL and markers)
COL_SAMPLE = "#1f77b4"   # blue
COL_FACTOR = "#ff7f0e"   # orange
COL_RF     = "#9e9e9e"

# ===================== Data & factor covariance =====================

def load_factors(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    f = f.loc[f.index >= START, ["MKT_RF","SMB","HML","RMW","CMA","RF","MOM"]].astype(float)
    return f

def load_prices(path: Path) -> pd.DataFrame:
    px = pd.read_parquet(path)
    if "Date" in px.columns:
        px = px.set_index("Date")
    px.index = pd.to_datetime(px.index)
    return px.loc[px.index >= START].sort_index()

def to_returns(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna(how="all")

def pick_tickers(px: pd.DataFrame, n: int) -> list[str]:
    avail = px.notna().sum().sort_values(ascending=False)
    return avail.index.tolist()[: min(n, len(avail))]

def regress_excess_on_factors(Rex: pd.DataFrame, F: pd.DataFrame):
    X = F[K_FACTORS].values
    X = np.c_[X, np.ones(len(F))]  # intercept
    k = len(K_FACTORS)

    tickers = Rex.columns.tolist()
    B = np.full((len(tickers), k), np.nan)
    resid_var = np.full(len(tickers), np.nan)

    for i, c in enumerate(tickers):
        y = Rex[c].values
        mask = np.isfinite(y)
        X_i, y_i = X[mask], y[mask]
        if len(y_i) <= k + 5:
            continue
        beta_i, *_ = np.linalg.lstsq(X_i, y_i, rcond=None)  # (k+1,)
        B[i, :] = beta_i[:k]
        resid = y_i - X_i @ beta_i
        dof = max(len(y_i) - (k + 1), 1)
        resid_var[i] = (resid @ resid) / dof

    Bdf = pd.DataFrame(B, index=tickers, columns=K_FACTORS).dropna()
    rvar = pd.Series(resid_var, index=tickers).loc[Bdf.index]
    return Bdf, rvar

def build_factor_cov(B: pd.DataFrame, F: pd.DataFrame, rvar: pd.Series) -> pd.DataFrame:
    Sigma_F = np.cov(F[K_FACTORS].T, ddof=1)        # KxK
    D = np.diag(rvar.values)                         # NxN
    Sigma = B.values @ Sigma_F @ B.values.T + D      # NxN
    return pd.DataFrame(Sigma, index=B.index, columns=B.index)

# ===================== Mean–Variance helpers =====================

def _regularize_cov(S: pd.DataFrame, ridge: float = 1e-8) -> pd.DataFrame:
    S = (S + S.T) / 2.0
    return S + np.eye(S.shape[0]) * ridge

def gmv_weights(S: pd.DataFrame) -> pd.Series:
    S = _regularize_cov(S)
    invS = np.linalg.inv(S.values)
    ones = np.ones(S.shape[0])
    A = ones @ invS @ ones
    w = invS @ ones / A
    return pd.Series(w, index=S.index)

def tangency_weights(mu: pd.Series, S: pd.DataFrame, rf_daily: float) -> pd.Series:
    S = _regularize_cov(S)
    mu_ex = mu.values - rf_daily
    invS = np.linalg.inv(S.values)
    w_unnorm = invS @ mu_ex
    w = w_unnorm / (np.ones(len(mu)) @ w_unnorm)
    return pd.Series(w, index=mu.index)

def efficient_frontier_cover(mu: pd.Series, S: pd.DataFrame,
                             r_gmv: float, vol_target: float,
                             npts: int = 180) -> tuple[np.ndarray, np.ndarray]:
    """
    Efficient frontier that **guarantees** the x-extent reaches at least
    'vol_target' by expanding the target-return grid if needed.
    """
    S = _regularize_cov(S)
    mu_v = mu.values
    invS = np.linalg.inv(S.values)
    ones = np.ones(len(mu_v))

    A = ones @ invS @ ones
    B = ones @ invS @ mu_v
    C = mu_v @ invS @ mu_v
    D = A*C - B*B

    def _frontier(r_min, r_max):
        r_grid = np.linspace(r_min, r_max, npts)
        vols = []
        for r in r_grid:
            lam = (C - B*r) / D
            gam = (A*r - B) / D
            w = invS @ (lam*ones + gam*mu_v)
            vols.append(np.sqrt(w @ S.values @ w))
        return np.array(vols), r_grid

    r_min = r_gmv
    r_max = max(mu_v.max(), r_gmv)
    vols, rets = _frontier(r_min, r_max)

    # Expand r_max until the frontier reaches the target volatility
    it = 0
    while np.max(vols) < vol_target * 1.02 and it < 12:
        r_max *= 1.25
        vols, rets = _frontier(r_min, r_max)
        it += 1
    return vols, rets

def ann_stats(w: pd.Series, mu_daily: pd.Series, S_daily: pd.DataFrame, rf_daily: float):
    r_d = float(w @ mu_daily)
    v_d = float(np.sqrt(w.values @ S_daily.values @ w.values))
    r_a = r_d * TRADING_DAYS
    v_a = v_d * np.sqrt(TRADING_DAYS)
    rf_a = rf_daily * TRADING_DAYS
    sharpe = (r_a - rf_a) / v_a if v_a > 0 else np.nan
    return r_a, v_a, sharpe

def _p(x, d=1):  # percent string helper for labels
    return f"{x*100:.{d}f}%"

# ===================== Plot (two separate figures with labels) =====================

def plot_single_model(model_name: str, color: str, ef, gmv, tan, rf_ann: float,
                      data_note: str, outpath: Path):
    ef_vol, ef_ret = ef
    gmv_vol, gmv_ret = gmv
    tan_vol, tan_ret = tan
    sharpe = (tan_ret - rf_ann) / tan_vol if tan_vol > 0 else np.nan

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    # Efficient frontier: BLACK, solid
    ax.plot(ef_vol, ef_ret, color="black", linewidth=2.6, label="Efficient Frontier")

    # CAL (dashed, model color)
    x = np.linspace(0, max(ef_vol.max(), tan_vol)*1.03, 200)
    slope = (tan_ret - rf_ann) / tan_vol if tan_vol > 0 else 0.0
    ax.plot(x, rf_ann + slope*x, linestyle="--", linewidth=1.8, color=color, alpha=0.95, label="CAL")

    # Risk-free level + point
    ax.axhline(rf_ann, color=COL_RF, linestyle=":", linewidth=1.0)
    ax.scatter([0], [rf_ann], color=COL_RF, s=40, zorder=5)
    ax.annotate(f"RF {_p(rf_ann)}", (0, rf_ann), xytext=(8, 8),
                textcoords="offset points", ha="left", va="bottom", fontsize=10, color="#555555")

    # GMV marker + label
    ax.scatter([gmv_vol], [gmv_ret], marker="D", s=110, color=color,
               edgecolor="white", linewidth=0.7, zorder=6, label="GMV")
    ax.annotate(f"GMV\nσ={_p(gmv_vol)}  μ={_p(gmv_ret)}",
                (gmv_vol, gmv_ret), xytext=(-80, 18),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=10, color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.9))

    # Tangency marker + label
    ax.scatter([tan_vol], [tan_ret], marker="*", s=220, color=color,
               edgecolor="white", linewidth=0.7, zorder=6, label="Tangency")
    ax.annotate(f"Tangency\nσ={_p(tan_vol)}  μ={_p(tan_ret)}\nSR={sharpe:.2f}",
                (tan_vol, tan_ret), xytext=(14, -4),
                textcoords="offset points", ha="left", va="top",
                fontsize=10, color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.9))

    # Axis formatting
    ax.set_title(f"Mean–Variance Frontier (annualized) — {model_name}")
    ax.set_xlabel("Volatility (σ)")
    ax.set_ylabel("Expected Return (μ)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    # Limits with padding (ensure EF covers CAL extent)
    xmax = max(ef_vol.max(), tan_vol) * 1.05
    ymax = max(ef_ret.max(), tan_ret, rf_ann)
    ymin = min(ef_ret.min(), gmv_ret, rf_ann)
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin - 0.03*(ymax - ymin), ymax + 0.05*(ymax - ymin))

    # Legend (compact)
    ax.legend(loc="upper left")

    # Context note
    ax.text(
        0.99, 0.02, data_note,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="#444444",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#dddddd", alpha=0.95)
    )

    fig.savefig(outpath, dpi=300)
    plt.show()

# ===================== Main =====================

def main():
    # Data
    factors = load_factors(FACTORS_CSV)
    prices  = load_prices(PRICES_PARQUET)
    tickers = pick_tickers(prices, N_STOCKS)
    prices  = prices[tickers]

    R = to_returns(prices)
    data = R.join(factors, how="inner")
    Rex = data[tickers].subtract(data["RF"], axis=0)

    # Covariances
    B, rvar = regress_excess_on_factors(Rex, data)
    tickers = B.index.tolist()
    R   = R[tickers]
    Rex = Rex[tickers]

    Sigma_factor = build_factor_cov(B, data, rvar)
    Sigma_sample = Rex.cov()

    # Expected returns & RF
    mu_daily = R.mean()
    rf_daily = float(factors["RF"].mean())

    # GMV
    w_gmv_sample = gmv_weights(Sigma_sample)
    w_gmv_factor = gmv_weights(Sigma_factor.loc[tickers, tickers])

    # Tangency
    w_tan_sample = tangency_weights(mu_daily, Sigma_sample, rf_daily)
    w_tan_factor = tangency_weights(mu_daily, Sigma_factor.loc[tickers, tickers], rf_daily)

    # Daily metrics (ret, vol)
    def port_point(w, S):
        ret_d = float(w @ mu_daily)
        vol_d = float(np.sqrt(w.values @ S.values @ w.values))
        return ret_d, vol_d

    ret_gmv_s_d, vol_gmv_s_d = port_point(w_gmv_sample, Sigma_sample)
    ret_gmv_f_d, vol_gmv_f_d = port_point(w_gmv_factor, Sigma_factor.loc[tickers, tickers])
    ret_tan_s_d, vol_tan_s_d = port_point(w_tan_sample,  Sigma_sample)
    ret_tan_f_d, vol_tan_f_d = port_point(w_tan_factor,  Sigma_factor.loc[tickers, tickers])

    # Efficient frontiers — ensure they reach tangency volatility on x-axis
    vols_s_d, rets_s_d = efficient_frontier_cover(mu_daily, Sigma_sample,
                                                  r_gmv=ret_gmv_s_d, vol_target=vol_tan_s_d, npts=220)
    vols_f_d, rets_f_d = efficient_frontier_cover(mu_daily, Sigma_factor.loc[tickers, tickers],
                                                  r_gmv=ret_gmv_f_d, vol_target=vol_tan_f_d, npts=220)

    # Annualize
    rf_ann   = rf_daily * TRADING_DAYS
    ef_s     = (vols_s_d * np.sqrt(TRADING_DAYS), rets_s_d * TRADING_DAYS)
    ef_f     = (vols_f_d * np.sqrt(TRADING_DAYS), rets_f_d * TRADING_DAYS)
    gmv_s    = (vol_gmv_s_d * np.sqrt(TRADING_DAYS), ret_gmv_s_d * TRADING_DAYS)
    gmv_f    = (vol_gmv_f_d * np.sqrt(TRADING_DAYS), ret_gmv_f_d * TRADING_DAYS)
    tan_s    = (vol_tan_s_d * np.sqrt(TRADING_DAYS), ret_tan_s_d * TRADING_DAYS)
    tan_f    = (vol_tan_f_d * np.sqrt(TRADING_DAYS), ret_tan_f_d * TRADING_DAYS)

    # Context note
    date_min = data.index.min().date()
    date_max = data.index.max().date()
    data_note = f"{len(tickers)} stocks | Daily | {date_min} → {date_max} | RF≈{rf_ann:.2%} p.a."

    # --- TWO SEPARATE PLOTS WITH DATA LABELS ---
    plot_single_model(
        model_name="Sample Covariance",
        color=COL_SAMPLE,
        ef=ef_s, gmv=gmv_s, tan=tan_s, rf_ann=rf_ann,
        data_note=data_note,
        outpath=RESULTS_DIR / "mv_frontier_sample_labeled.png"
    )

    plot_single_model(
        model_name="Factor-model Covariance",
        color=COL_FACTOR,
        ef=ef_f, gmv=gmv_f, tan=tan_f, rf_ann=rf_ann,
        data_note=data_note,
        outpath=RESULTS_DIR / "mv_frontier_factor_labeled.png"
    )

    # Performance comparison (annualized, GMV)
    gmv_s_stats = ann_stats(w_gmv_sample, mu_daily, Sigma_sample, rf_daily)
    gmv_f_stats = ann_stats(w_gmv_factor,  mu_daily, Sigma_factor.loc[tickers, tickers], rf_daily)

    comp = pd.DataFrame({
        "GMV – Sample Σ": {"Ann. Return": gmv_s_stats[0], "Ann. Vol": gmv_s_stats[1], "Ann. Sharpe": gmv_s_stats[2]},
        "GMV – Factor Σ": {"Ann. Return": gmv_f_stats[0], "Ann. Vol": gmv_f_stats[1], "Ann. Sharpe": gmv_f_stats[2]},
    })
    comp["Difference (Factor − Sample)"] = comp["GMV – Factor Σ"] - comp["GMV – Sample Σ"]
    comp = comp.round(5)
    out_csv = RESULTS_DIR / "gmv_performance_comparison.csv"
    comp.to_csv(out_csv, float_format="%.5f")

    print("\n=== GMV Performance (annualized) ===")
    print(comp.to_string())
    print(f"\nSaved charts:\n  - {RESULTS_DIR / 'mv_frontier_sample_labeled.png'}\n  - {RESULTS_DIR / 'mv_frontier_factor_labeled.png'}")
    print(f"Saved comparison CSV: {out_csv}")

if __name__ == "__main__":
    main()
