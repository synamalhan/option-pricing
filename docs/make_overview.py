"""Builds docs/overview-dark.png and docs/overview-light.png: UI screenshots plus measured model results.

Run from the repo root:  python docs/make_overview.py
Needs matplotlib and the screenshots in docs/screenshots/.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from models import black_scholes, greeks, monte_carlo_option

S, K, T, r, SIG = 100.0, 100.0, 1.0, 0.05, 0.20
THEMES = {  # INK, GREY, LIGHT, ACCENT, RED, PAPER
    "dark": ("#e4e4e7", "#9b9ba4", "#3b3b42", "#c584f7", "#e8735f", "#161618"),
    "light": ("#1d1d1f", "#6e6e73", "#d2d2d7", "#6b3fa0", "#b3412f", "#fbfaf7"),
}


def label(ax, letter, title):
    ax.set_title(f"({letter})  {title}", loc="left", fontweight="bold", pad=8)


def mc(n, seed, k=K, kind="call"):
    np.random.seed(seed)
    price, s_t = monte_carlo_option(S, k, T, r, SIG, n, kind)
    pay = np.exp(-r * T) * (np.maximum(s_t - k, 0) if kind == "call" else np.maximum(k - s_t, 0))
    return price, pay.std(ddof=1) / np.sqrt(n), pay


bs = black_scholes(S, K, T, r, SIG)

# ---- measurements ----
Ns = np.unique(np.logspace(2, 5, 13).astype(int))
rmse = [np.sqrt(np.mean([(mc(n, s)[0] - bs) ** 2 for s in range(300)])) for n in Ns]

coverage = np.mean([abs(p - bs) <= 1.96 * se for p, se, _ in (mc(10_000, s) for s in range(1000))])

strikes = np.arange(70, 135, 5)
acc = {kind: np.array([(mc(100_000, 7, k, kind)[0] - black_scholes(S, k, T, r, SIG, kind), mc(100_000, 7, k, kind)[1])
                       for k in strikes]) for kind in ("call", "put")}

time_N = [10 ** e for e in range(3, 7)]
mc_ms = []
for n in time_N:
    t0 = time.perf_counter()
    for _ in range(7):
        monte_carlo_option(S, K, T, r, SIG, n)
    mc_ms.append((time.perf_counter() - t0) / 7 * 1e3)
spots = np.random.default_rng(0).uniform(50, 150, 1_000_000)
t0 = time.perf_counter()
black_scholes(spots, K, T, r, SIG)
bs_ms = (time.perf_counter() - t0) * 1e3

h = 1e-4
fd_err = {}
for kind in ("call", "put"):
    g = greeks(S, K, T, r, SIG, kind)
    f = lambda **kw: black_scholes(**{"S": S, "K": K, "T": T, "r": r, "sigma": SIG, **kw}, option_type=kind)
    fd = {"Delta": (f(S=S + h) - f(S=S - h)) / (2 * h),
          "Gamma": (f(S=S + 1e-2) - 2 * f() + f(S=S - 1e-2)) / 1e-4,
          "Vega": (f(sigma=SIG + h) - f(sigma=SIG - h)) / (2 * h) / 100,
          "Theta": -(f(T=T + h) - f(T=T - h)) / (2 * h) / 365,
          "Rho": (f(r=r + h) - f(r=r - h)) / (2 * h) / 100}
    fd_err[kind] = {n: (g[n], abs(g[n] - fd[n])) for n in g}

_, _, pay = mc(200_000, 42)
pnl = pay - bs
var95 = -np.percentile(pnl, 5)


# ---- figure ----
def draw(theme):
    INK, GREY, LIGHT, ACCENT, RED, PAPER = THEMES[theme]

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    })
    fig = plt.figure(figsize=(15, 13), dpi=170)
    gs = GridSpec(4, 4, figure=fig, height_ratios=[0.2, 1.42, 1, 1], hspace=0.42, wspace=0.32,
                  left=0.05, right=0.97, top=0.975, bottom=0.045)

    head = fig.add_subplot(gs[0, :])
    head.axis("off")
    head.text(0, 1.0, "Option Pricing & Risk Lab", fontsize=20, fontweight="bold", va="top")
    head.text(0, 0.12, "Black-Scholes and Monte Carlo pricing of European options: interface and validation results.  "
              f"Reference contract: at-the-money call, $S_0$ = {S:.0f}, K = {K:.0f}, T = {T:.0f}y, r = {r:.0%}, "
              f"σ = {SIG:.0%}, Black-Scholes price {bs:.4f}.", fontsize=10, color=GREY, va="bottom", transform=head.transAxes)
    head.axhline(0, color=INK, lw=0.8)

    for col, (name, letter, title, crop) in enumerate([
        ("pricing", "a", "Interface: pricing and Greeks", 0.86),
        ("montecarlo", "b", "Interface: Monte Carlo simulation", 0.80),
    ]):
        ax = fig.add_subplot(gs[1, col * 2:col * 2 + 2])
        img = plt.imread(f"docs/screenshots/{name}.png")
        ax.imshow(img[: int(img.shape[0] * crop)])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(LIGHT); s.set_linewidth(0.8)
        label(ax, letter, title)

    # (c) convergence
    ax = fig.add_subplot(gs[2, 0])
    ax.loglog(Ns, rmse, "o", ms=3.5, color=ACCENT, label="measured RMSE (300 seeds)")
    ref = rmse[0] * np.sqrt(Ns[0] / Ns)
    ax.loglog(Ns, ref, "--", color=GREY, lw=0.9, label=r"$\propto N^{-1/2}$")
    slope = np.polyfit(np.log(Ns), np.log(rmse), 1)[0]
    ax.text(0.97, 0.93, f"fitted slope {slope:.3f}", transform=ax.transAxes, ha="right", fontsize=8.5)
    ax.set_xlabel("simulated paths N"); ax.set_ylabel("RMSE vs Black-Scholes")
    ax.legend(loc="lower left", fontsize=8)
    label(ax, "c", "Monte Carlo convergence")

    # (d) accuracy across strikes
    ax = fig.add_subplot(gs[2, 1])
    for kind, color, off in (("call", ACCENT, -0.7), ("put", RED, 0.7)):
        err, se = acc[kind].T
        ax.errorbar(strikes + off, err * 100, yerr=1.96 * se * 100, fmt="o", ms=3, lw=0.8, capsize=2,
                    color=color, label=kind)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel("strike K"); ax.set_ylabel("MC − BS (cents), ±95% CI")
    ax.set_ylim(top=22)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.text(0.98, 0.95, "one seed for all strikes,\nso errors move together", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color=GREY)
    label(ax, "d", "Pricing error across strikes, N = 100k")

    # (e) runtime
    ax = fig.add_subplot(gs[2, 2])
    ax.loglog(time_N, mc_ms, "s-", ms=3.5, lw=0.9, color=INK, label="Monte Carlo, 1 option")
    ax.axhline(bs_ms, color=ACCENT, lw=0.9, ls="--")
    ax.text(time_N[0], bs_ms * 1.25, f"Black-Scholes, 10⁶ options vectorised: {bs_ms:.0f} ms",
            color=ACCENT, fontsize=8)
    for n, ms in zip(time_N, mc_ms):
        ax.annotate(f"{ms:.2f} ms" if ms < 1 else f"{ms:.0f} ms", (n, ms), textcoords="offset points",
                    xytext=(5, -3), fontsize=7.5, color=GREY)
    ax.set_xlabel("simulated paths N"); ax.set_ylabel("wall time (ms)")
    ax.set_ylim(top=bs_ms * 6)
    ax.legend(loc="lower right", fontsize=8)
    label(ax, "e", "Runtime (NumPy, single thread)")

    # (f) CI coverage
    ax = fig.add_subplot(gs[2, 3])
    z = np.array([(p - bs) / se for p, se, _ in (mc(10_000, s) for s in range(1000, 2000))])
    ax.hist(z, bins=40, density=True, color=LIGHT, edgecolor=GREY, lw=0.4)
    xs = np.linspace(-4, 4, 200)
    ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color=ACCENT, lw=1.1, label="N(0, 1)")
    for x in (-1.96, 1.96):
        ax.axvline(x, color=INK, lw=0.6, ls=":")
    ax.text(0.03, 0.93, f"95% CI coverage\n{coverage:.1%} of 1,000 runs", transform=ax.transAxes, fontsize=8.5, va="top")
    ax.set_xlabel("(MC − BS) / standard error"); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper right", fontsize=8)
    label(ax, "f", "Standard-error calibration")

    # (g) Greek profiles
    ax = fig.add_subplot(gs[3, 0])
    xs = np.linspace(60, 140, 300)
    for frac, ls in ((1.0, "-"), (0.25, "--"), (0.04, ":")):
        ax.plot(xs, greeks(xs, K, T * frac, r, SIG, "call")["Delta"], ls=ls, color=INK, lw=1,
                label=f"T = {T * frac:.2f}y")
    ax.axvline(K, color=LIGHT, lw=0.8)
    ax.set_xlabel("spot S"); ax.set_ylabel("call delta")
    ax.legend(loc="lower right", fontsize=8)
    label(ax, "g", "Delta steepens toward expiry")

    # (h) risk
    ax = fig.add_subplot(gs[3, 1])
    bins = np.linspace(-bs - 0.5, np.percentile(pnl, 99.5), 70)
    ax.hist(pnl[pnl < 0], bins=bins, color=RED, alpha=0.85, label="loss")
    ax.hist(pnl[pnl >= 0], bins=bins, color=GREY, alpha=0.6, label="profit")
    ax.axvline(-var95, color=INK, lw=0.8, ls="--")
    ax.text(4, ax.get_ylim()[1] * 0.25, r"$\leftarrow$" + f" VaR 95% = {var95:.2f}\n    (the full premium)", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("P&L of long call held to expiry"); ax.set_ylabel("count (log)")
    ax.legend(loc="upper right", fontsize=8)
    label(ax, "h", f"P&L, 200k paths: {np.mean(pnl < 0):.1%} lose money")

    # (i) Greeks table
    ax = fig.add_subplot(gs[3, 2:])
    ax.axis("off")
    label(ax, "i", "Analytic Greeks vs central finite differences")
    rows = [[n, f"{fd_err['call'][n][0]:+.5f}", f"{fd_err['call'][n][1]:.1e}",
             f"{fd_err['put'][n][0]:+.5f}", f"{fd_err['put'][n][1]:.1e}"] for n in fd_err["call"]]
    pc = black_scholes(S, K, T, r, SIG, "call") - black_scholes(S, K, T, r, SIG, "put") - (S - K * np.exp(-r * T))
    rows.append(["Put-call parity", "", f"{abs(pc):.1e}", "", ""])
    tbl = ax.table(cellText=rows, colLabels=["", "call value", "|abs. error|", "put value", "|abs. error|"],
                   loc="upper left", cellLoc="right", colLoc="right", bbox=[0, 0.02, 1, 0.9])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (i, j), cell in tbl.get_celld().items():
        cell.set_facecolor(PAPER); cell.set_linewidth(0)
        if i == 0:
            cell.set_text_props(fontweight="bold"); cell.visible_edges = "B"; cell.set_linewidth(0.6)
        if j == 0:
            cell._loc = "left"
        if i == len(rows):
            cell.visible_edges = "T"; cell.set_linewidth(0.6)
    ax.text(0, -0.06, "Vega and rho per 1 percentage point, theta per calendar day.  Step h = 1e-4 (1e-2 for gamma).",
            transform=ax.transAxes, fontsize=8, color=GREY)

    fig.savefig(f"docs/overview-{theme}.png")
    plt.close(fig)


for theme in THEMES:
    draw(theme)
print(f"saved docs/overview-{{dark,light}}.png  coverage={coverage:.3f} bs_1M={bs_ms:.0f}ms")
