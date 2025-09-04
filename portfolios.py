# mean_variance_two_plots_with_labels_longonly.py
# -----------------------------------------------------------------------------
# Build FACTOR covariance (FF5+MOM) and SAMPLE covariance, compute:
#   • Global Minimum-Variance (GMV) and Tangency portfolios for BOTH (LONG-ONLY)
# Make TWO separate charts (one per model) with:
#   • Efficient Frontier in BLACK (solid)
#   • CAL dashed in model color
#   • GMV/Tangency markers with DATA LABELS (σ, μ, Sharpe)
#   • Frontier extends at least to the Tangency volatility on the x-axis
# Print a compact (annualized) GMV performance comparison and Top-7 allocations.
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.optimize import minimize  # <-- long-only solver (SLSQP)
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

# ===================== Mean–Variance helpers (LONG-ONLY) =====================

def _regularize_cov(S: pd.DataFrame, ridge: float = 1e-8) -> pd.DataFrame:
    S = (S + S.T) / 2.0
    return S + np.eye(S.shape[0]) * ridge

def _solve_long_only(objective, grad, w0, bounds, cons):
    res = minimize(objective, w0, method="SLSQP", jac=grad, bounds=bounds, constraints=cons,
                   options=dict(maxiter=1000, ftol=1e-12, disp=False))
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    # clip tiny negatives from numerical noise
    w = np.clip(res.x, 0, None)
    w = w / w.sum()
    return w

def gmv_weights_long_only(S: pd.DataFrame) -> pd.Series:
    S = _regularize_cov(S)
    N = S.shape[0]
    ones = np.ones(N)
    bounds = [(0.0, 1.0)] * N
    cons = ({'type':'eq', 'fun': lambda w: w.sum() - 1.0,
             'jac': lambda w: np.ones_like(w)})
    def obj(w): return w @ (S.values @ w)
    def grad(w): return 2.0 * (S.values @ w)
    w0 = np.full(N, 1.0/N)
    w = _solve_long_only(obj, grad, w0, bounds, cons)
    return pd.Series(w, index=S.index)

def tangency_weights_long_only(mu: pd.Series, S: pd.DataFrame, rf_daily: float) -> pd.Series:
    S = _regularize_cov(S)
    N = S.shape[0]
    mu_ex = mu.values - rf_daily
    bounds = [(0.0, 1.0)] * N
    cons = ({'type':'eq', 'fun': lambda w: w.sum() - 1.0,
             'jac': lambda w: np.ones_like(w)})
    def obj(w):
        a = mu_ex @ w
        b = np.sqrt(w @ (S.values @ w))
        if b <= 0: return 1e6
        return -a / b  # maximize Sharpe -> minimize negative
    def grad(w):
        Sw = S.values @ w
        b = np.sqrt(w @ Sw)
        a = mu_ex @ w
        if b <= 0: return np.zeros_like(w)
        # d(-a/b)/dw = -mu_ex/b + a*(Sw)/b^3
        return -mu_ex / b + a * Sw / (b**3)
    w0 = np.full(N, 1.0/N)
    w = _solve_long_only(obj, grad, w0, bounds, cons)
    return pd.Series(w, index=mu.index)

def efficient_frontier_long_only(mu: pd.Series, S: pd.DataFrame,
                                 rf_daily: float,
                                 vol_target: float | None = None,
                                 npts: int = 140):
    """
    Generate EF by solving long-only mean–variance problems over a lambda grid:
        min  w'Sw - λ * (mu'w)
        s.t. sum(w)=1, w>=0
    Expand λ until EF reaches vol_target (if provided).
    """
    S = _regularize_cov(S)
    N = S.shape[0]
    mu_v = mu.values
    bounds = [(0.0, 1.0)] * N
    cons = ({'type':'eq', 'fun': lambda w: w.sum() - 1.0,
             'jac': lambda w: np.ones_like(w)})

    def solve_lambda(lmbda, w_start):
        def obj(w): return w @ (S.values @ w) - lmbda * (mu_v @ w)
        def grad(w): return 2.0 * (S.values @ w) - lmbda * mu_v
        return _solve_long_only(obj, grad, w_start, bounds, cons)

    lambdas = np.logspace(-3, 3, npts)
    vols, rets = [], []
    w_prev = np.full(N, 1.0/N)

    for lmbda in lambdas:
        w_prev = solve_lambda(lmbda, w_prev)
        Sw = S.values @ w_prev
        vols.append(np.sqrt(w_prev @ Sw))
        rets.append(mu_v @ w_prev)

    vols = np.array(vols)
    rets = np.array(rets)

    # extend if needed to cover at least the tangency volatility
    if vol_target is not None and (np.max(vols) < 1.02 * vol_target):
        lambdas2 = np.logspace(3, 5, npts//3)
        for lmbda in lambdas2:
            w_prev = solve_lambda(lmbda, w_prev)
            Sw = S.values @ w_prev
            vols = np.append(vols, np.sqrt(w_prev @ Sw))
            rets = np.append(rets, mu_v @ w_prev)

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
    ax.set_title(f"Mean–Variance Frontier (annualized) — {model_name} (Long-only)")
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

    # LONG-ONLY GMV & Tangency
    w_gmv_sample = gmv_weights_long_only(Sigma_sample)
    w_gmv_factor = gmv_weights_long_only(Sigma_factor.loc[tickers, tickers])
    w_tan_sample = tangency_weights_long_only(mu_daily, Sigma_sample, rf_daily)
    w_tan_factor = tangency_weights_long_only(mu_daily, Sigma_factor.loc[tickers, tickers], rf_daily)

    # Daily metrics (ret, vol)
    def port_point(w, S):
        ret_d = float(w @ mu_daily)
        vol_d = float(np.sqrt(w.values @ S.values @ w.values))
        return ret_d, vol_d

    ret_gmv_s_d, vol_gmv_s_d = port_point(w_gmv_sample, Sigma_sample)
    ret_gmv_f_d, vol_gmv_f_d = port_point(w_gmv_factor, Sigma_factor.loc[tickers, tickers])
    ret_tan_s_d, vol_tan_s_d = port_point(w_tan_sample,  Sigma_sample)
    ret_tan_f_d, vol_tan_f_d = port_point(w_tan_factor,  Sigma_factor.loc[tickers, tickers])

    # Efficient frontiers (LONG-ONLY) — ensure they reach tangency volatility
    vols_s_d, rets_s_d = efficient_frontier_long_only(mu_daily, Sigma_sample,
                                                      rf_daily, vol_target=vol_tan_s_d, npts=220)
    vols_f_d, rets_f_d = efficient_frontier_long_only(mu_daily, Sigma_factor.loc[tickers, tickers],
                                                      rf_daily, vol_target=vol_tan_f_d, npts=220)

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
        outpath=RESULTS_DIR / "mv_frontier_sample_labeled_longonly.png"
    )

    plot_single_model(
        model_name="Factor-model Covariance",
        color=COL_FACTOR,
        ef=ef_f, gmv=gmv_f, tan=tan_f, rf_ann=rf_ann,
        data_note=data_note,
        outpath=RESULTS_DIR / "mv_frontier_factor_labeled_longonly.png"
    )

    # Performance comparison (annualized, GMV)
    gmv_s_stats = ann_stats(w_gmv_sample, mu_daily, Sigma_sample, rf_daily)
    gmv_f_stats = ann_stats(w_gmv_factor,  mu_daily, Sigma_factor.loc[tickers, tickers], rf_daily)

    comp = pd.DataFrame({
        "GMV – Sample Σ (LO)": {"Ann. Return": gmv_s_stats[0], "Ann. Vol": gmv_s_stats[1], "Ann. Sharpe": gmv_s_stats[2]},
        "GMV – Factor Σ (LO)": {"Ann. Return": gmv_f_stats[0], "Ann. Vol": gmv_f_stats[1], "Ann. Sharpe": gmv_f_stats[2]},
    })
    comp["Difference (Factor − Sample)"] = comp["GMV – Factor Σ (LO)"] - comp["GMV – Sample Σ (LO)"]
    comp = comp.round(5)
    out_csv = RESULTS_DIR / "gmv_performance_comparison_longonly.csv"
    comp.to_csv(out_csv, float_format="%.5f")

    print("\n=== GMV Performance (annualized, LONG-ONLY) ===")
    print(comp.to_string())
    print(f"\nSaved charts:\n  - {RESULTS_DIR / 'mv_frontier_sample_labeled_longonly.png'}\n  - {RESULTS_DIR / 'mv_frontier_factor_labeled_longonly.png'}")
    print(f"Saved comparison CSV: {out_csv}")

    # --- Step 3: Top-7 allocations (in-sample GMV), simple & clear ---
    def top_allocations(w: pd.Series, k: int = 7):
        order = w.abs().sort_values(ascending=False).index
        top = w.loc[order].head(k)
        table = (100 * top).round(2).to_frame("Weight (%)")  # sign preserved (will be ≥0 here)
        largest = float(top.abs().iloc[0])                   # e.g., 0.18 = 18%
        topk_share = float(top.abs().sum())                  # <= 1.0 for long-only
        return table, largest, topk_share

    tbl_s, largest_s, share7_s = top_allocations(w_gmv_sample, k=7)
    tbl_f, largest_f, share7_f = top_allocations(w_gmv_factor, k=7)

    print("\n=== GMV — Sample Σ (LO) : Top 7 positions ===")
    print(tbl_s.to_string())
    print(f"Largest single position: {largest_s*100:.2f}%")
    print(f"Sum of weights in top 7: {share7_s*100:.2f}%")

    print("\n=== GMV — Factor Σ (LO) : Top 7 positions ===")
    print(tbl_f.to_string())
    print(f"Largest single position: {largest_f*100:.2f}%")
    print(f"Sum of weights in top 7: {share7_f*100:.2f}%")

    # Optional: save the small tables
    (tbl_s.to_csv(RESULTS_DIR / "gmv_sample_top7_longonly.csv"))
    (tbl_f.to_csv(RESULTS_DIR / "gmv_factor_top7_longonly.csv"))

if __name__ == "__main__":
    main()
