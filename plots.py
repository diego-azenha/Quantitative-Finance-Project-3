import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure results folder exists
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Global Economist-like style
plt.rcParams.update({
    "axes.edgecolor": "white",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "grid.color": "#e0e0e0",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "font.family": "DejaVu Sans",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "legend.frameon": False,
})

def plot_mean_variance_frontier(
    cloud_interior,              # (V_in, R_in)
    cloud_edge,                  # (V_edge, R_edge) — sparse, frontier-hugging
    rf_rate,
    opt_point=None,
    mv_point=None,
    ef_curve=None,               # (vols, rets)
    filename="mean_variance_frontier.png"
):
    """
    Mean-variance figure with two clouds (no legend entries),
    efficient frontier curve, CAL, Optimal (green star), Min Var (blue diamond).
    Labels appear to the left of the markers, close to the icons.
    """
    plt.figure(figsize=(10, 6))

    V_in, R_in = map(np.asarray, cloud_interior)
    V_ed, R_ed = map(np.asarray, cloud_edge)

    sharpe_in = (R_in - rf_rate) / V_in
    sharpe_ed = (R_ed - rf_rate) / V_ed

    # Interior cloud (no legend entry)
    plt.scatter(
        V_in, R_in,
        c=sharpe_in,
        cmap="YlOrRd_r",
        marker="o",
        edgecolor="none",
        alpha=0.45,
        s=10,
        label=None
    )

    # Edge cloud (no legend entry)
    sc2 = plt.scatter(
        V_ed, R_ed,
        c=sharpe_ed,
        cmap="YlOrRd_r",
        marker="o",
        edgecolor="none",
        alpha=0.85,
        s=14,
        label=None
    )

    cbar = plt.colorbar(sc2)
    cbar.set_label("Sharpe Ratio")

    # Risk-free (no legend entry)
    plt.axhline(rf_rate, color="#999999", linestyle=":", linewidth=1.0, label=None)

    # Efficient frontier (in legend)
    if ef_curve is not None and len(ef_curve[0]) > 1:
        ef_vols, ef_rets = ef_curve
        plt.plot(ef_vols, ef_rets, linewidth=2.0, linestyle="-", color="#303030", label="Efficient Frontier")

    # CAL + Optimal
    if opt_point is not None:
        opt_vol, opt_ret = opt_point
        if opt_vol > 0 and np.isfinite(opt_ret):
            x = np.linspace(0, max(np.max(V_in), np.max(V_ed), opt_vol) * 1.05, 200)
            slope = (opt_ret - rf_rate) / opt_vol
            y = rf_rate + slope * x
            plt.plot(x, y, linestyle="--", linewidth=1.6, color="#2ca25f", label="CAL (Optimal)")
        plt.scatter([opt_vol], [opt_ret], marker="*", s=180, zorder=5, color="#2ca25f", label="Optimal (Sharpe-max)")
        plt.annotate("Optimal", (opt_vol, opt_ret), xytext=(-25, 0), textcoords="offset points",
                     ha="right", va="center")

    # Minimum variance (blue)
    if mv_point is not None:
        mv_vol, mv_ret = mv_point
        plt.scatter([mv_vol], [mv_ret], marker="D", s=90, zorder=5, color="#08519c", label="Minimum Variance")
        plt.annotate("Min Var", (mv_vol, mv_ret), xytext=(-25, 0), textcoords="offset points",
                     ha="right", va="center")

    plt.title("Mean–Variance Frontier (annualized)")
    plt.xlabel("Portfolio Volatility (Risk)")
    plt.ylabel("Portfolio Return")
    plt.legend(loc="best")
    plt.tight_layout()

    outpath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
