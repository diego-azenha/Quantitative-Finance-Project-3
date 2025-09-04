# oos_backtest_longonly.py
# ---------------------------------------------------------------------
# Out-of-sample GMV backtest (MONTHLY, LONG-ONLY) comparing:
#   • Sample covariance   vs   • Factor-model covariance (FF5+MOM)
# Uses the same clean_data files and functions you already have.
# Outputs CSVs and a simple cumulative return plot.
# ---------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse functions/constants from your long-only plotting module
from portfolios import (
    CDIR, RESULTS_DIR, START, K_FACTORS,
    load_factors, load_prices, regress_excess_on_factors, build_factor_cov,
    gmv_weights_long_only
)

TRADING_MONTHS = 12

# ---------- helpers ----------

def to_monthly_prices(px_daily: pd.DataFrame) -> pd.DataFrame:
    return px_daily.resample("M").last()

def returns_from_prices(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna(how="all")

def compound_daily_to_monthly(f_daily: pd.DataFrame) -> pd.DataFrame:
    return (1.0 + f_daily).resample("M").prod() - 1.0

def max_drawdown(ret: pd.Series) -> float:
    curve = (1.0 + ret).cumprod()
    roll_max = curve.cummax()
    dd = (curve / roll_max) - 1.0
    return float(dd.min())

def ann_summary(r: pd.Series, rf: pd.Series) -> dict:
    # monthly stats -> annualized
    r = r.dropna()
    if len(r) == 0:
        return {"Ann.Return": np.nan, "Ann.Vol": np.nan, "Ann.Sharpe": np.nan, "MaxDD": np.nan}
    mu_m = r.mean()
    sd_m = r.std(ddof=1)
    rf_m = rf.reindex_like(r).fillna(0.0).mean()
    ann_ret = (1.0 + r).prod() ** (TRADING_MONTHS / len(r)) - 1.0
    ann_vol = sd_m * np.sqrt(TRADING_MONTHS)
    ann_sh = ((mu_m - rf_m) * TRADING_MONTHS) / ann_vol if ann_vol > 0 else np.nan
    mdd = max_drawdown(r)
    return {"Ann.Return": ann_ret, "Ann.Vol": ann_vol, "Ann.Sharpe": ann_sh, "MaxDD": mdd}

# ---------- core OOS backtest ----------

def run_oos_backtest_long_only(window_months: int = 60, top_n: int = 30, min_total_obs: int = 72):
    # Load daily, then convert to monthly
    factors_d = load_factors(CDIR / "ff_factors_daily_clean.csv")
    prices_d  = load_prices(CDIR / "clean_stock_prices.parquet")

    px_m = to_monthly_prices(prices_d)
    R_m  = returns_from_prices(px_m)
    F_m  = compound_daily_to_monthly(factors_d)

    # Align and pick the universe (stable set with enough data)
    R_m = R_m.loc[R_m.index >= START].sort_index()
    F_m = F_m.loc[F_m.index >= START].sort_index()
    common_idx = R_m.index.intersection(F_m.index)
    R_m = R_m.loc[common_idx]
    F_m = F_m.loc[common_idx, ["RF"] + K_FACTORS]

    # Choose tickers with sufficient history
    valid_counts = R_m.notna().sum()
    universe = valid_counts[valid_counts >= min_total_obs].sort_values(ascending=False).index.tolist()[:top_n]
    R_m = R_m[universe].dropna(how="all")

    # Storage
    dates = R_m.index
    start_i = window_months
    ret_sample, ret_factor = [], []
    rf_m_series = F_m["RF"].copy()
    w_sample_hist = []
    w_factor_hist = []

    for i in range(start_i, len(dates)):
        est_idx = dates[i - window_months:i]   # window (t-window ... t-1)
        hold_dt = dates[i]                     # hold on month t

        R_win  = R_m.loc[est_idx]
        F_win  = F_m.loc[est_idx]
        RF_win = F_win["RF"]

        # Align excess returns for the window
        Rex_win = R_win.subtract(RF_win, axis=0).dropna(axis=1, how="any")
        if Rex_win.shape[1] < 5:  # need at least a few names
            continue
        tick = Rex_win.columns.tolist()

        # --- Sample covariance (window) ---
        Sigma_sample = Rex_win.cov()

        # --- Factor covariance (window): betas + Σ_F + idio ---
        B, rvar = regress_excess_on_factors(Rex_win, F_win)
        tick_f = B.index.tolist()
        Sigma_factor = build_factor_cov(B, F_win, rvar)

        # Compute long-only GMV weights for both models
        w_s = gmv_weights_long_only(Sigma_sample.loc[tick, tick])
        w_f = gmv_weights_long_only(Sigma_factor.loc[tick_f, tick_f])

        # Store weights (expanded to full universe with zeros)
        w_s_full = pd.Series(0.0, index=universe); w_s_full.loc[w_s.index] = w_s.values
        w_f_full = pd.Series(0.0, index=universe); w_f_full.loc[w_f.index] = w_f.values
        w_sample_hist.append(w_s_full.rename(hold_dt))
        w_factor_hist.append(w_f_full.rename(hold_dt))

        # Realized return next month (hold_dt)
        r_hold = R_m.loc[hold_dt]
        rs = float(w_s.reindex_like(r_hold).fillna(0.0) @ r_hold.fillna(0.0))
        rf = float(w_f.reindex_like(r_hold).fillna(0.0) @ r_hold.fillna(0.0))
        ret_sample.append(pd.Series({"date": hold_dt, "ret": rs}))
        ret_factor.append(pd.Series({"date": hold_dt, "ret": rf}))

    # Assemble outputs
    ret_s = pd.DataFrame(ret_sample).set_index("date")["ret"]
    ret_f = pd.DataFrame(ret_factor).set_index("date")["ret"]
    oos_returns = pd.concat([ret_s.rename("GMV_Sample"), ret_f.rename("GMV_Factor")], axis=1)

    w_sample_df = pd.DataFrame(w_sample_hist)
    w_factor_df = pd.DataFrame(w_factor_hist)

    # Save CSVs
    RESULTS_DIR.mkdir(exist_ok=True)
    oos_returns.to_csv(RESULTS_DIR / "oos_returns.csv", float_format="%.6f")
    w_sample_df.to_csv(RESULTS_DIR / "oos_weights_sample.csv", float_format="%.6f")
    w_factor_df.to_csv(RESULTS_DIR / "oos_weights_factor.csv", float_format="%.6f")

    # Summary stats (annualized from monthly)
    summ_s = ann_summary(oos_returns["GMV_Sample"], rf=rf_m_series)
    summ_f = ann_summary(oos_returns["GMV_Factor"], rf=rf_m_series)
    summary = pd.DataFrame({"GMV_Sample (LO)": summ_s, "GMV_Factor (LO)": summ_f})
    summary["Diff (Factor − Sample)"] = summary["GMV_Factor (LO)"] - summary["GMV_Sample (LO)"]
    summary = summary.round(5)
    summary.to_csv(RESULTS_DIR / "oos_summary.csv", float_format="%.5f")

    # Simple cumulative performance plot
    fig, ax = plt.subplots(figsize=(9, 5))
    (1 + oos_returns["GMV_Sample"]).cumprod().plot(ax=ax, lw=2.0, label="GMV — Sample Σ (LO)")
    (1 + oos_returns["GMV_Factor"]).cumprod().plot(ax=ax, lw=2.0, label="GMV — Factor Σ (LO)")
    ax.set_title("OOS Cumulative Return (Monthly, Long-only GMV)")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, axis="y", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "oos_cumulative.png", dpi=300)
    plt.close(fig)

    print("\n=== OOS Summary (annualized) ===")
    print(summary.to_string())
    print(f"\nSaved: {RESULTS_DIR / 'oos_returns.csv'}")
    print(f"Saved: {RESULTS_DIR / 'oos_weights_sample.csv'}")
    print(f"Saved: {RESULTS_DIR / 'oos_weights_factor.csv'}")
    print(f"Saved: {RESULTS_DIR / 'oos_summary.csv'}")
    print(f"Saved: {RESULTS_DIR / 'oos_cumulative.png'}")

if __name__ == "__main__":
    run_oos_backtest_long_only(window_months=60, top_n=30, min_total_obs=72)
