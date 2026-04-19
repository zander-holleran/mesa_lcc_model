import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

PARAM_LABELS = {
    "v_f": "Free-Flow Speed (v_f)",
    "v_0": "Speed Constant (v_0)",
    "k_j": "Jam Density (k_j)",
    "k_0": "Optimal Density (k_0)",
}


#===================================================================
# Model Functions
#===================================================================
def greenshields(k, v_f, k_j):
    """Greenshields (1935): linear speed-density.
    k, k_j in veh/mi; v_f in mph. Returns speed in mph.
    """
    return np.clip(v_f * (1.0 - k / k_j), 0, None)


def greenberg(k, v_0, k_j):
    """Greenberg (1959): logarithmic speed-density.
    Undefined at k=0 — clipped to 1e-6 to avoid log singularity.
    Also clipped to k_j to avoid log of negative inside region k > k_j.
    k, k_j in veh/mi; v_0 in mph. Returns speed in mph.
    """
    k_safe = np.clip(np.asarray(k, dtype=float), 1e-6, k_j)
    return v_0 * np.log(k_j / k_safe)


def underwood(k, v_f, k_0):
    """Underwood (1961): exponential speed-density.
    k, k_0 in veh/mi; v_f in mph. Returns speed in mph.
    """
    return v_f * np.exp(-np.asarray(k, dtype=float) / k_0)

#===================================================================
# Unit conversions
#===================================================================

KMH_TO_MPH    = 0.621371       # km/h → mph
VEHKM_TO_VEHMI = 1.60934       # veh/km → veh/mi (multiply by 1.60934)
METERS_PER_MILE = 1609.34

def kmh_to_mph(v):   return v * KMH_TO_MPH
def vehkm_to_vehmi(k): return k * VEHKM_TO_VEHMI

def convert_to_imperial(entry):
    """Return a copy of entry with speed in mph and density in veh/mi."""
    e = dict(entry)
    if e.get("units") == "metric":
        for key in ("v_f", "v_0"):
            if key in e:
                e[key] = kmh_to_mph(e[key])
        for key in ("k_j", "k_0"):
            if key in e:
                e[key] = vehkm_to_vehmi(e[key])
        e["units"] = "imperial_converted"
    return e


def convert_literature_entries(LITERATURE_RAW):
    """Convert a list of raw literature entries to imperial units."""
    LIT = {
        model: [convert_to_imperial(e) for e in entries]
        for model, entries in LITERATURE_RAW.items()
    }
    return LIT


def display_entries(LIT):
    for model_name, entries in LIT.items():
        rows = []
        for e in entries:
            row = {"Source": e["source"]}
            for p in ("v_f", "v_0", "k_j", "k_0"):
                if p in e:
                    row[PARAM_LABELS.get(p, p)] = round(e[p], 2)
            if e.get("outlier"):
                row["note"] = "OUTLIER"
            rows.append(row)
        df = pd.DataFrame(rows)
        print(f"\n{'='*60}")
        print(f"  {model_name.upper()}  (all values in mph / veh/mi)")
        print(f"{'='*60}")
        print(df.to_string(index=False))


#===================================================================
# Plotting
#===================================================================

def plot_param_dists(dists, grid_alpha=0.3, fig_w_per_panel=3.5, fig_h=4):
    """Plot PDFs for a collection of parameter distributions.

    Parameters
    ----------
    dists : list of (label, dist, entries)
        label   — axis title, e.g. "v_f  (mph)"
        dist    — GaussianDist or UniformDist
        entries — list of dicts with at minimum a value key matching the
                  parameter and optionally "n_obs" and "source". Can also be
                  a plain list of floats (legacy) or None to skip the rug.

    Returns the (fig, axes) pair.
    """
    from scipy.stats import norm as scipy_norm

    n = len(dists)
    fig, axes = plt.subplots(1, n, figsize=(fig_w_per_panel * n, fig_h))
    if n == 1:
        axes = [axes]

    for ax, (label, dist, raw) in zip(axes, dists):
        color = "#2c7bb6"

        # infer param key from label (e.g. "k_j  (veh/mi)" -> "k_j")
        param_key = label.split()[0]

        # --- parse raw into parallel lists: values, n_obs ---
        vals, n_obs_list = [], []
        if raw is not None:
            for item in raw:
                if isinstance(item, dict):
                    if param_key in item:
                        vals.append(item[param_key])
                    else:
                        # fallback: first matching known key
                        for k in ("v_f", "v_0", "k_j", "k_0"):
                            if k in item:
                                vals.append(item[k])
                                break
                    n_obs_list.append(item.get("n_obs"))
                else:
                    vals.append(float(item))
                    n_obs_list.append(None)

        has_nobs = all(w is not None for w in n_obs_list) and len(n_obs_list) > 0

        # --- PDF curve ---
        if hasattr(dist, 'lo'):
            pad  = (dist.hi - dist.lo) * 0.4
            x_lo = max(0.0, dist.lo - pad)
            x_hi = dist.hi + pad
            height = 1.0 / (dist.hi - dist.lo)

            ax.hlines(height, dist.lo, dist.hi, color=color, lw=2.2)
            ax.fill_between([dist.lo, dist.hi], [height, height], alpha=0.35, color=color,
                            label=f"Uniform  [{dist.lo:.1f}, {dist.hi:.1f}]")
            ax.vlines([dist.lo, dist.hi], 0, height, color=color, lw=2.2)
            ax.axvline(dist.mid, color="#d73027", lw=1.6, ls="--",
                       label=f"mid = {dist.mid:.1f}")
            y_max = height

        else:  # GaussianDist
            mu, sigma = dist.mu, dist.sigma
            x_lo = max(0.0, mu - 3.5 * sigma)
            x_hi = mu + 3.5 * sigma
            x    = np.linspace(x_lo, x_hi, 400)
            y    = scipy_norm.pdf(x, mu, sigma)

            ax.plot(x, y, lw=2.2, color=color)
            ax.fill_between(x, y, alpha=0.18, color=color)

            ci_lo_v = max(scipy_norm.ppf(0.025, mu, sigma), 0.0)
            ci_hi_v = scipy_norm.ppf(0.975, mu, sigma)
            mask    = (x >= ci_lo_v) & (x <= ci_hi_v)
            ax.fill_between(x[mask], y[mask], alpha=0.32, color=color,
                            label=f"95% CI  [{ci_lo_v:.1f}, {ci_hi_v:.1f}]")
            ax.axvline(mu, color="#d73027", lw=1.6, ls="--", label=f"μ = {mu:.1f}")
            y_max = y.max()

        # --- rug / heap marks ---
        if vals:
            rug_color = "#555"
            heap_zone = 0.22 * y_max  # reserve bottom 22% for heaps

            if has_nobs:
                nobs_arr  = np.array(n_obs_list, dtype=float)
                max_nobs  = nobs_arr.max()
                # bar heights proportional to n_obs, scaled into heap_zone
                bar_heights = (nobs_arr / max_nobs) * heap_zone

                bar_width = (x_hi - x_lo) * 0.025
                for v, h, nob in zip(vals, bar_heights, n_obs_list):
                    ax.bar(v, h, width=bar_width, bottom=0, color=rug_color,
                           alpha=0.45, zorder=5, edgecolor="white", lw=0.5)
                    ax.text(v, h + 0.01 * y_max, f"n={int(nob)}",
                            ha="center", va="bottom", fontsize=6, color=rug_color)

                ax.bar([], [], color=rug_color, alpha=0.45,
                       label=f"studies (bar height ~ n_obs)")
            else:
                for v in vals:
                    ax.axvline(v, color=rug_color, lw=1.0, alpha=0.5, ymax=0.12)
                ax.scatter(vals, np.zeros(len(vals)) - 0.003 * y_max,
                           marker="|", s=120, color=rug_color, zorder=5,
                           label=f"lit. values (n={len(vals)})")

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("value", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=grid_alpha)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(bottom=-0.008 * y_max)

    fig.tight_layout()
    return fig, axes


def plot_engineering_dists(engineering, grid_alpha=0.3, fig_w_per_panel=3.5, fig_h=4):
    """Plot uniform PDF panels for engineering estimate ranges.

    Parameters
    ----------
    engineering : dict
        {param_name: (lo, hi)}, e.g. {"v_f": (34, 37), "k_j": (100, 160), ...}.
        Only v_f, k_j, v_0, k_0 are plotted (in that order, if present).

    Returns (fig, axes).
    """
    units = {"v_f": "mph", "k_j": "veh/mi", "v_0": "mph", "k_0": "veh/mi"}
    param_order = [k for k in ("v_f", "k_j", "v_0", "k_0") if k in engineering]
    n = len(param_order)

    fig, axes = plt.subplots(1, n, figsize=(fig_w_per_panel * n, fig_h))
    if n == 1:
        axes = [axes]

    color = "#2c7bb6"
    for ax, key in zip(axes, param_order):
        lo, hi = engineering[key]
        mid = (lo + hi) / 2.0
        pad = (hi - lo) * 0.4
        x_lo = max(0.0, lo - pad)
        x_hi = hi + pad
        height = 1.0 / (hi - lo)

        ax.hlines(height, lo, hi, color=color, lw=2.2)
        ax.fill_between([lo, hi], [height, height], alpha=0.35, color=color,
                        label=f"Uniform  [{lo:.1f}, {hi:.1f}]")
        ax.vlines([lo, hi], 0, height, color=color, lw=2.2)
        ax.axvline(mid, color="#d73027", lw=1.6, ls="--",
                   label=f"mid = {mid:.1f}")

        label = f"{key}  ({units.get(key, '')})"
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("value", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=grid_alpha)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(bottom=-0.008 * height)

    fig.tight_layout()
    return fig, axes


def sensitivity_plot_1param(ax, model_fn, param_grid, fixed_kwargs,
                             swept_param_name, param_units="",
                             cmap_name="viridis", sr210_range=None,
                             xlabel="Density (k) (veh/mi)",
                             ylabel="Speed v (mph)", title="",
                             k_range=None, grid_alpha=0.3):
    """Plot speed-density curves sweeping one parameter."""
    if k_range is None:
        k_range = np.linspace(0.5, 160, 500)
    cmap = plt.get_cmap(cmap_name, len(param_grid))
    for i, val in enumerate(param_grid):
        kwargs = {**fixed_kwargs, swept_param_name: val}
        v = model_fn(k_range, **kwargs)
        v = np.clip(v, 0, None)
        ax.plot(k_range, v, color=cmap(i / max(len(param_grid) - 1, 1)),
                lw=1.8, label=f"{val:.0f}{param_units}")
    if sr210_range:
        lo, hi = sr210_range
        ax.axhspan(lo, hi, alpha=0.08, color="orange",
                   label="SR-210 Free-Flow Speed est. (35–50 mph)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, loc="upper right", title=PARAM_LABELS.get(swept_param_name, swept_param_name),
              title_fontsize=8)
    ax.grid(True, alpha=grid_alpha)
    ax.set_ylim(0, 95)
    ax.set_xlim(0, k_range.max())


def ci_bands_from_ranges(ranges, k_range=None, n_mc=2000, seed=42):
    """Compute CI bands for all three models from engineering-style range dicts.

    Parameters
    ----------
    ranges : dict
        {param_name: (lo, hi)}, e.g. {"v_f": (34, 40), "k_j": (50, 250), ...}.
    k_range : array-like or None
        Density values. Defaults to linspace(0.5, 250, 500).

    Returns dict: {"greenshields": (mean, lo, hi), "greenberg": ..., "underwood": ...}
    """
    if k_range is None:
        k_range = np.linspace(0.5, 250, 500)

    def _dist(key):
        lo, hi = ranges[key]
        return UniformDist(lo, hi)

    models = [
        ("greenshields", greenshields, [_dist("v_f"), _dist("k_j")]),
        ("greenberg",    greenberg,    [_dist("v_0"), _dist("k_j")]),
        ("underwood",    underwood,    [_dist("v_f"), _dist("k_0")]),
    ]
    return {
        key: mc_ci_band(fn, dists, k_range, n_mc=n_mc, seed=seed)
        for key, fn, dists in models
    }


def ci_bands_from_dists(param_dists, k_range=None, n_mc=2000, seed=42):
    """Compute CI bands from a param_dists dict (output of lit_param_dists).

    Parameters
    ----------
    param_dists : dict
        {model_key: {param_name: dist_object}}, e.g. from lit_param_dists().
    k_range : array-like or None
        Density values. Defaults to linspace(0.5, 250, 500).

    Returns dict: {"greenshields": (mean, lo, hi), ...}
    """
    if k_range is None:
        k_range = np.linspace(0.5, 250, 500)

    models = [
        ("greenshields", greenshields, ["v_f", "k_j"]),
        ("greenberg",    greenberg,    ["v_0", "k_j"]),
        ("underwood",    underwood,    ["v_f", "k_0"]),
    ]
    return {
        key: mc_ci_band(fn, [param_dists[key][p] for p in pnames],
                        k_range, n_mc=n_mc, seed=seed)
        for key, fn, pnames in models
    }


def plot_kv_ci_figure(param_dists, k_range=None, n_mc=2000, seed=42,
                      ci_alpha=0.20, grid_alpha=0.3, save_path=None):
    """Plot 3-panel k-v CI figure from parameter distributions.

    Parameters
    ----------
    param_dists : dict
        {model_key: {param_name: dist_object}}, e.g. from lit_param_dists().
        Each dist must have a .sample(rng) method (UniformDist, GaussianDist).
    k_range : array-like or None
        Density values (veh/mi). Defaults to linspace(0.5, 250, 500).
    n_mc, seed : int
        Monte Carlo parameters passed to mc_ci_band().

    Returns (fig, axes).
    """
    if k_range is None:
        k_range = np.linspace(0.5, 250, 500)

    panels = [
        ("greenshields", greenshields, ["v_f", "k_j"],
         "Greenshields (1935)", "#2196F3", (0, 95)),
        ("greenberg", greenberg, ["v_0", "k_j"],
         "Greenberg (1959)", "#E91E63", (0, 110)),
        ("underwood", underwood, ["v_f", "k_0"],
         "Underwood (1961)", "#4CAF50", (0, 95)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax, (key, model_fn, pnames, title, color, ylim) in zip(axes, panels):
        dists = [param_dists[key][p] for p in pnames]
        v_mean, v_lo, v_hi = mc_ci_band(model_fn, dists, k_range,
                                         n_mc=n_mc, seed=seed)
        ax.fill_between(k_range, v_lo, v_hi, alpha=ci_alpha, color=color,
                        label="95% CI")
        ax.plot(k_range, v_mean, color=color, lw=2.2, label="Mean curve")
        ax.set_xlabel("Density (k) (veh/mi)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=grid_alpha)
        ax.set_xlim(0, k_range.max())
        ax.set_ylim(*ylim)

    axes[0].set_ylabel("Speed v (mph)", fontsize=10)
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    fig.suptitle(
        "Speed-Density (k\u2013v) Fundamental Diagram\n"
        "Literature Parameter Uncertainty \u2014 95% CI",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes


def plot_kv_ci_from_engineering(engineering, **kwargs):
    """Thin wrapper: convert engineering ranges to dists, call plot_kv_ci_figure."""
    param_dists = {
        "greenshields": {"v_f": UniformDist(*engineering["v_f"]),
                         "k_j": UniformDist(*engineering["k_j"])},
        "greenberg":    {"v_0": UniformDist(*engineering["v_0"]),
                         "k_j": UniformDist(*engineering["k_j"])},
        "underwood":    {"v_f": UniformDist(*engineering["v_f"]),
                         "k_0": UniformDist(*engineering["k_0"])},
    }
    return plot_kv_ci_figure(param_dists, **kwargs)


def plot_kv_ci_comparison(k, lit_ci_bands, sim_ci_bands,
                          ci_alpha=0.18, grid_alpha=0.3, save_path=None):
    """Overlay literature and simulation CI bands on a 3-panel k-v figure.

    Parameters
    ----------
    k : array-like
        Density values (veh/mi).
    lit_ci_bands, sim_ci_bands : dict
        Keys: "greenshields", "greenberg", "underwood".
        Values: (v_mean, v_lo, v_hi) arrays from mc_ci_band().

    Returns (fig, axes).
    """
    panels = [
        ("greenshields", "Greenshields (1935)", "#2196F3", "#FF9800", (0, 95)),
        ("greenberg",    "Greenberg (1959)",    "#E91E63", "#FF9800", (0, 110)),
        ("underwood",    "Underwood (1961)",     "#4CAF50", "#FF9800", (0, 95)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax, (key, title, lit_color, sim_color, ylim) in zip(axes, panels):
        # literature band
        lit_mean, lit_lo, lit_hi = lit_ci_bands[key]
        ax.fill_between(k, lit_lo, lit_hi, alpha=ci_alpha, color=lit_color,
                        label="Literature 95% CI")
        ax.plot(k, lit_mean, color=lit_color, lw=2.0, ls="--",
                label="Literature mean")

        # simulation curve (or band)
        sim_val = sim_ci_bands[key]
        if isinstance(sim_val, np.ndarray):
            # Line only — no CI band
            ax.plot(k, sim_val, color=sim_color, lw=2.2, label="ABM fitted")
        else:
            sim_mean, sim_lo, sim_hi = sim_val
            ax.fill_between(k, sim_lo, sim_hi, alpha=ci_alpha + 0.08,
                            color=sim_color, label="ABM 95% CI")
            ax.plot(k, sim_mean, color=sim_color, lw=2.2, label="ABM mean")

        ax.set_xlabel("Density (k) (veh/mi)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=grid_alpha)
        ax.set_xlim(0, k.max())
        ax.set_ylim(*ylim)

    axes[0].set_ylabel("Speed v (mph)", fontsize=10)
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    fig.suptitle(
        "Speed-Density (k\u2013v) Fundamental Diagram\n"
        "Literature vs ABM Simulation \u2014 95% CI Comparison",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes


def plot_kv_scatter_ci(lit_ci_bands, kv_df, n_bins=30,
                       scatter_alpha=0.02, ci_alpha=0.18, grid_alpha=0.3,
                       save_path=None):
    """3-panel figure: literature CI bands with raw sim data and binned mean line.

    Parameters
    ----------
    lit_ci_bands : dict
        Keys: "greenshields", "greenberg", "underwood".
        Values: (v_mean, v_lo, v_hi) arrays from ci_bands_from_ranges().
    kv_df : DataFrame
        Output of process_sweep_validation() with columns k_vehmi, v_mph.
    n_bins : int
        Number of equal-width density bins for the mean speed line.

    Returns (fig, axes).
    """
    k_obs = kv_df["k_vehmi"].values
    v_obs = kv_df["v_mph"].values

    # bin-average for mean line
    k_ba, v_ba = bin_average_kv(k_obs, v_obs, n_bins=n_bins)

    # infer k_range from lit_ci_bands (all share same length)
    first_key = next(iter(lit_ci_bands))
    n_k = len(lit_ci_bands[first_key][0])
    k_max = k_obs.max()
    k_range = np.linspace(0.5, k_max, n_k)

    panels = [
        ("greenshields", "Greenshields (1935)", "#2196F3", (0, 95)),
        ("greenberg",    "Greenberg (1959)",    "#E91E63", (0, 110)),
        ("underwood",    "Underwood (1961)",    "#4CAF50", (0, 95)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax, (key, title, color, ylim) in zip(axes, panels):
        # literature CI band
        lit_mean, lit_lo, lit_hi = lit_ci_bands[key]
        ax.fill_between(k_range, lit_lo, lit_hi, alpha=ci_alpha, color=color,
                        label="Literature 95% CI")
        ax.plot(k_range, lit_mean, color=color, lw=2.0, ls="--",
                label="Literature mean")

        # raw sim scatter
        ax.scatter(k_obs, v_obs, s=2, alpha=scatter_alpha, color="#888",
                   edgecolors="none", rasterized=True, label="Sim observations")

        # binned mean line
        ax.plot(k_ba, v_ba, color="#FF6F00", lw=2.5, zorder=5,
                label=f"Sim mean ({n_bins} bins)")

        ax.set_xlabel("Density (k) (veh/mi)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=grid_alpha)
        ax.set_xlim(0, k_max)
        ax.set_ylim(*ylim)

    axes[0].set_ylabel("Speed v (mph)", fontsize=10)
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    fig.suptitle(
        "Speed\u2013Density (k\u2013v) Fundamental Diagram\n"
        "Literature CI vs Raw ABM Simulation Data",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes


def plot_kv_scatter_fit(kv_df, k_range=None, n_bins=50, n_mc=2000, seed=42,
                        scatter_alpha=0.05, ci_alpha=0.20, grid_alpha=0.3,
                        save_path=None):
    """3-panel k-v figure with raw observations and NLS-fitted curves with CIs.

    Parameters
    ----------
    kv_df : DataFrame
        Output of process_sweep_validation() with columns k_vehmi, v_mph.
    k_range : array-like, optional
        Density values for fitted curves. Default np.linspace(0.5, 250, 500).
    n_mc : int
        Monte Carlo samples for CI bands from fitted pcov.
    seed : int
        RNG seed.
    scatter_alpha : float
        Alpha for observation scatter points.
    ci_alpha : float
        Alpha for CI band fill.
    grid_alpha : float
        Alpha for grid lines.
    save_path : str or None
        If set, save figure to this path.

    Returns
    -------
    fig, axes
    """
    if k_range is None:
        k_range = np.linspace(0.5, 250, 500)

    k_obs = kv_df["k_vehmi"].values
    v_obs = kv_df["v_mph"].values

    models = [
        ("Greenshields (1935)", greenshields, [45, 200], "#2196F3"),
        ("Greenberg (1959)",    greenberg,    [25, 200], "#E91E63"),
        ("Underwood (1961)",    underwood,    [45, 30],  "#4CAF50"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    rng = np.random.default_rng(seed)

    for ax, (title, model_fn, p0, color) in zip(axes, models):
        # Scatter raw observations
        ax.scatter(k_obs, v_obs, s=4, alpha=scatter_alpha, color=color,
                   edgecolors="none", label="Observations", rasterized=True)

        # Bin-average for balanced fitting
        k_ba, v_ba = bin_average_kv(k_obs, v_obs, n_bins=n_bins)

        # Fit model to bin-averaged data
        result = fit_model(model_fn, k_ba, v_ba, p0=p0, model_name=title)

        if result["converged"]:
            popt = result["params"]
            pcov = result["pcov"]
            r2 = result["r2"]

            # Fitted mean curve
            v_mean = model_fn(k_range, *popt)
            ax.plot(k_range, v_mean, color=color, lw=2.2, label="Fitted mean")

            # CI bands from pcov via MC
            param_samples = rng.multivariate_normal(popt, pcov, n_mc)
            param_samples = np.clip(param_samples, 0.1, None)
            curves = np.array([model_fn(k_range, *p) for p in param_samples])
            v_lo, v_hi = np.percentile(curves, [2.5, 97.5], axis=0)
            ax.fill_between(k_range, v_lo, v_hi, color=color, alpha=ci_alpha,
                            label="95% CI")

            ax.set_title(f"{title}\n$R^2$ = {r2:.3f}", fontsize=11,
                         fontweight="bold")
        else:
            ax.set_title(f"{title}\n(no convergence)", fontsize=11,
                         fontweight="bold")

        ax.set_xlabel("Density (k) (veh/mi)", fontsize=10)
        ax.set_xlim(0, k_range.max())
        ax.grid(True, alpha=grid_alpha)
        ax.legend(fontsize=7, loc="upper right")

    ylims = [(0, 95), (0, 110), (0, 95)]
    for ax, yl in zip(axes, ylims):
        ax.set_ylim(yl)
    axes[0].set_ylabel("Speed v (mph)", fontsize=10)

    fig.suptitle(
        "Speed\u2013Density (k\u2013v) Fundamental Diagram\n"
        "ABM Simulation Data \u2014 NLS Fit with 95% CI",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes


#===================================================================
# Probabilistic Helpers
#===================================================================

class UniformDist:
    """Uniform distribution over [lo, hi] for use in mc_ci_band."""
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi
        self.mid = (lo + hi) / 2.0

    def sample(self, rng):
        return rng.uniform(self.lo, self.hi)

    def __repr__(self):
        return f"UniformDist({self.lo}, {self.hi})"


class GaussianDist:
    """Gaussian distribution (mu, sigma) for use in mc_ci_band."""
    def __init__(self, mu, sigma):
        self.mu, self.sigma = mu, sigma
        self.mid = mu

    def sample(self, rng):
        return max(rng.normal(self.mu, self.sigma), 0.1)

    def __repr__(self):
        return f"GaussianDist({self.mu:.2f}, {self.sigma:.2f})"


def estimate_gaussian_params(values, exclude_outliers_z=2.5):
    """Estimate mean and sigma for MC sampling from a list of literature estimates.
    Optionally excludes values beyond z-score threshold.
    Returns (mu, sigma). Enforces minimum 5% CV.
    """
    arr = np.array(values, dtype=float)
    if len(arr) < 2:
        mu    = arr[0]
        sigma = mu * 0.10
        return mu, sigma
    mu    = np.mean(arr)
    sigma = np.std(arr, ddof=1)
    if exclude_outliers_z and sigma > 0:
        z   = np.abs((arr - mu) / sigma)
        arr = arr[z <= exclude_outliers_z]
        mu  = np.mean(arr)
        sigma = np.std(arr, ddof=1) if len(arr) > 1 else mu * 0.10
    sigma = max(sigma, mu * 0.05)   # minimum 5% CV prevents degenerate zero-width bands
    return GaussianDist(mu, sigma)


def estimate_weighted_gaussian_params(values, weights, min_cv=0.05):
    """Weighted Gaussian parameter estimate for meta-analytic literature synthesis.

    Uses w_i = weight_i as an inverse-variance proxy (suitable when individual
    parameter standard errors are unavailable and within-study variance scales
    as ~1/n). Sigma captures between-study heterogeneity via the weighted
    sample standard deviation with a reliability-weights Bessel correction.

    Parameters
    ----------
    values : array-like
        Parameter point estimates from literature studies.
    weights : array-like
        Weights for each study (typically n_obs). Larger = more influence.
    min_cv : float
        Minimum coefficient of variation (sigma/mu). Default 0.05 (5%).
    """
    vals = np.asarray(values, dtype=float)
    wts  = np.asarray(weights, dtype=float)

    if len(vals) != len(wts):
        raise ValueError(f"values length {len(vals)} != weights length {len(wts)}")

    if len(vals) < 2:
        mu    = vals[0]
        sigma = mu * 0.10
        return GaussianDist(mu, max(sigma, mu * min_cv))

    # normalised weights
    w = wts / wts.sum()

    # weighted mean
    mu = np.dot(w, vals)

    # weighted sample variance (reliability-weights Bessel correction)
    # V2 = sum(w_i^2); unbiased sigma^2 = sum(w_i*(x_i - mu)^2) / (1 - V2)
    v2 = np.dot(w, w)
    weighted_var = np.dot(w, (vals - mu) ** 2) / (1.0 - v2)

    sigma = np.sqrt(max(weighted_var, 0.0))
    sigma = max(sigma, mu * min_cv)

    return GaussianDist(mu, sigma)


def estimate_gaussian_params_from_entries(entries, param_key, n_obs_key="n_obs",
                                          exclude_outliers=True, min_cv=0.05):
    """Extract parameter values and n_obs from literature entries, return weighted GaussianDist.

    Parameters
    ----------
    entries : list of dict
        Literature entries (post unit-conversion). Each must have param_key.
    param_key : str
        Key to extract parameter values (e.g., "k_j", "v_0").
    n_obs_key : str
        Key for sample size weights. Falls back to unweighted if any entry lacks it.
    exclude_outliers : bool
        If True, skip entries where entry.get("outlier") is truthy.
    min_cv : float
        Minimum coefficient of variation passed to the estimator.
    """
    filtered = [e for e in entries if param_key in e]
    if exclude_outliers:
        filtered = [e for e in filtered if not e.get("outlier")]

    values = [e[param_key] for e in filtered]

    if all(n_obs_key in e for e in filtered):
        weights = [e[n_obs_key] for e in filtered]
        return estimate_weighted_gaussian_params(values, weights, min_cv=min_cv)
    else:
        return estimate_gaussian_params(values)


def lit_param_dists(lit):
    """Build parameter distributions from literature entries.

    Parameters
    ----------
    lit : dict
        Keys: "greenshields", "greenberg", "underwood".
        Values: lists of literature entry dicts (post unit-conversion).

    Returns
    -------
    dict
        {"greenshields": {"v_f": UniformDist, "k_j": GaussianDist},
         "greenberg":    {"v_0": GaussianDist, "k_j": GaussianDist},
         "underwood":    {"v_f": UniformDist, "k_0": GaussianDist}}
    """
    return {
        "greenshields": {
            "v_f": estimate_gaussian_params_from_entries(lit["greenshields"], "v_f"),
            "k_j": estimate_gaussian_params_from_entries(lit["greenshields"], "k_j"),
        },
        "greenberg": {
            "v_0": estimate_gaussian_params_from_entries(lit["greenberg"], "v_0"),
            "k_j": estimate_gaussian_params_from_entries(lit["greenberg"], "k_j"),
        },
        "underwood": {
            "v_f": estimate_gaussian_params_from_entries(lit["underwood"], "v_f"),
            "k_0": estimate_gaussian_params_from_entries(lit["underwood"], "k_0"),
        },
    }


def mc_ci_band(model_fn, param_distributions, k_range, n_mc=2000, seed=42,
               ci_lo=2.5, ci_hi=97.5):
    """Monte Carlo CI band.

    param_distributions: list of GaussianDist or UniformDist objects.
    Returns: (v_mean, v_lo, v_hi) each shape (len(k_range),)
    """
    rng    = np.random.default_rng(seed)
    curves = np.zeros((n_mc, len(k_range)))
    for i in range(n_mc):
        params = [d.sample(rng) for d in param_distributions]
        v = model_fn(k_range, *params)
        curves[i] = np.clip(v, 0, None)
    v_mean = np.mean(curves, axis=0)
    v_lo   = np.percentile(curves, ci_lo, axis=0)
    v_hi   = np.percentile(curves, ci_hi, axis=0)
    return v_mean, v_lo, v_hi


#===================================================================
# Other Stuff
#===================================================================


def compute_snapshot_kv(df, road_length_m=19950, n_spatial=20, min_veh=2,
                        road_parquet="data/roads/hw210_sl_and_curvs.parquet"):
    """Compute instantaneous (k, v) observations from Tier 2 snapshot data.

    At each sampled step, counts vehicles in spatial bins along the road and
    computes the arithmetic mean speed per bin. This avoids the Edie time-area
    denominator issues that arise when most vehicles are queued off-road
    (negative distance_traveled).

    Parameters
    ----------
    df : DataFrame
        Raw Tier 2 spatial data with columns: Step, AgentID, distance_traveled,
        speed (mph), status.
    road_length_m : float
        Total road length in meters.
    n_spatial : int
        Number of spatial bins along the road.
    min_veh : int
        Minimum vehicles in a bin to include (filters noise).
    road_parquet : str or None
        Path to road geometry parquet with speed_limit and distance_traveled
        columns. If provided, each bin gets the speed limit at its midpoint.

    Returns DataFrame with columns: Step, x_bin, n_veh, k_vehmi, v_mph,
    and speed_limit (if road_parquet is provided).
    """
    on_road = df[
        (df["distance_traveled"] >= 0)
        & (df["distance_traveled"] <= road_length_m)
        & (df["status"].isin(["driving", "slowing"]))
    ].copy()

    seg_length_mi = (road_length_m / n_spatial) / METERS_PER_MILE

    x_edges = np.linspace(0, road_length_m, n_spatial + 1)
    on_road["x_bin"] = pd.cut(
        on_road["distance_traveled"], bins=x_edges,
        labels=False, include_lowest=True,
    )
    on_road = on_road.dropna(subset=["x_bin"])
    on_road["x_bin"] = on_road["x_bin"].astype(int)

    grouped = on_road.groupby(["Step", "x_bin"], observed=True).agg(
        n_veh=("AgentID", "nunique"),
        v_mph=("speed", "mean"),
    ).reset_index()

    grouped["k_vehmi"] = grouped["n_veh"] / seg_length_mi

    # Map each spatial bin to its speed limit via bin midpoint
    if road_parquet is not None:
        road = pd.read_parquet(road_parquet, columns=["distance_traveled", "speed_limit"])
        road = road.sort_values("distance_traveled").reset_index(drop=True)
        bin_midpoints = (x_edges[:-1] + x_edges[1:]) / 2.0
        sl_by_bin = np.searchsorted(road["distance_traveled"].values, bin_midpoints, side="right") - 1
        sl_by_bin = np.clip(sl_by_bin, 0, len(road) - 1)
        bin_sl_map = dict(enumerate(road["speed_limit"].values[sl_by_bin]))
        grouped["speed_limit"] = grouped["x_bin"].map(bin_sl_map)

    return grouped[grouped["n_veh"] >= min_veh].copy()


def bin_average_kv(k, v, n_bins=50):
    """Bin-average speed observations by density for balanced model fitting.

    Discretizes k into equal-width bins, computes mean v per bin.
    Returns only bins with at least 1 observation.
    """
    pos = k > 0
    if pos.sum() == 0:
        return np.array([]), np.array([])
    k_edges = np.linspace(k[pos].min(), k.max(), n_bins + 1)
    bin_idx = np.digitize(k, k_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    k_avg, v_avg = [], []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() > 0:
            k_avg.append(k[mask].mean())
            v_avg.append(v[mask].mean())
    return np.array(k_avg), np.array(v_avg)


def fit_kv_models(kv_df, k_range=None, n_bins=50):
    """Fit classical models to pooled k-v data with bin-averaging.

    Parameters
    ----------
    kv_df : DataFrame
        Output of process_sweep_validation() with columns k_vehmi, v_mph.
    k_range : array-like, optional
        Density values for evaluating fitted curves. Default np.linspace(0.5, 250, 500).
    n_bins : int
        Number of density bins for bin-averaging before fit.

    Returns
    -------
    sim_params : dict
        {model_key: {"p0_name": val, "p1_name": val, "r2": float}}
        Fitted parameter point estimates.
    sim_curves : dict
        {model_key: v_mean_array} — fitted curve evaluated at k_range,
        ready for plot_kv_ci_comparison() as (v_mean, v_mean, v_mean).
    """
    if k_range is None:
        k_range = np.linspace(0.5, 250, 500)

    k_obs = kv_df["k_vehmi"].values
    v_obs = kv_df["v_mph"].values
    k_ba, v_ba = bin_average_kv(k_obs, v_obs, n_bins=n_bins)

    models = [
        ("greenshields", greenshields, [45, 200], ["v_f", "k_j"]),
        ("greenberg",    greenberg,    [25, 200], ["v_0", "k_j"]),
        ("underwood",    underwood,    [45, 30],  ["v_f", "k_0"]),
    ]

    sim_params = {}
    sim_curves = {}

    for model_key, model_fn, p0, param_names in models:
        result = fit_model(model_fn, k_ba, v_ba, p0=p0, model_name=model_key)

        if not result["converged"]:
            sim_params[model_key] = {"r2": np.nan, "converged": False}
            sim_curves[model_key] = model_fn(k_range, *p0)
            continue

        popt = result["params"]
        entry = {"r2": result["r2"], "converged": True}
        for name, val in zip(param_names, popt):
            entry[name] = round(float(val), 2)
        sim_params[model_key] = entry
        sim_curves[model_key] = model_fn(k_range, *popt)

    return sim_params, sim_curves


def process_sweep_validation(sweep_results_path, seasons_dir, road_length_m=19950,
                              n_spatial=20):
    """Process all runs from a validation sweep into a combined snapshot k-v DataFrame.

    Parameters
    ----------
    sweep_results_path : str or Path
        Path to sweep_*_results.parquet.
    seasons_dir : str or Path
        Directory containing season output folders.
    road_length_m : float
        Road length in meters.
    n_spatial : int
        Number of spatial bins for snapshot density.

    Returns
    -------
    DataFrame
        Combined snapshot k-v data with columns: season_id, Step, x_bin,
        n_veh, v_mph, k_vehmi.
    """
    results_df = pd.read_parquet(sweep_results_path)
    frames = []

    for _, row in results_df.iterrows():
        sid = row["season_id"]
        path = f"{seasons_dir}/{sid}/day_0_spatial.parquet"

        try:
            df = pd.read_parquet(path)
        except (FileNotFoundError, OSError):
            continue

        kv = compute_snapshot_kv(df, road_length_m=road_length_m, n_spatial=n_spatial)
        if len(kv) == 0:
            continue

        kv["season_id"] = sid
        frames.append(kv)

    if not frames:
        return pd.DataFrame(columns=["season_id", "Step", "x_bin", "n_veh", "v_mph", "k_vehmi", "speed_limit"])

    return pd.concat(frames, ignore_index=True)[
        ["season_id", "Step", "x_bin", "n_veh", "v_mph", "k_vehmi", "speed_limit"]
    ]


# --- NLS model fitting ---

def fit_model(model_fn, k_obs, v_obs, p0, bounds=(0, np.inf), model_name=""):
    """Fit model_fn(k, *params) = v via nonlinear least squares.

    Returns dict with: model, params, pcov, rmse, r2, n_obs, converged.
    """
    mask = (k_obs > 0) & (v_obs >= 0) & np.isfinite(k_obs) & np.isfinite(v_obs)
    k_fit, v_fit = k_obs[mask], v_obs[mask]

    if len(k_fit) < len(p0) + 2:
        return {"converged": False, "model": model_name, "n_obs": len(k_fit),
                "error": "insufficient data"}
    try:
        popt, pcov = curve_fit(model_fn, k_fit, v_fit, p0=p0, bounds=bounds, maxfev=8000)
        v_pred    = model_fn(k_fit, *popt)
        residuals = v_fit - v_pred
        ss_res    = np.sum(residuals**2)
        ss_tot    = np.sum((v_fit - v_fit.mean())**2)
        rmse      = np.sqrt(np.mean(residuals**2))
        r2        = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {
            "model": model_name,
            "params": popt,
            "pcov":   pcov,
            "rmse":   rmse,
            "r2":     r2,
            "n_obs":  len(k_fit),
            "converged": True,
        }
    except (RuntimeError, ValueError) as exc:
        return {"converged": False, "model": model_name,
                "n_obs": len(k_fit), "error": str(exc)}
