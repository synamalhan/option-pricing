import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from models import black_scholes, greeks, monte_carlo_option, simulate_paths

st.set_page_config(page_title="Options Pricing", layout="wide")

ACCENT, BLUE, RED, GREEN, MUTED = "#c584f7", "#5fa8f5", "#f26d6d", "#4fd1a1", "#8b8b9a"
st.markdown(
    """
    <style>
    .block-container {padding-top: 3.5rem;}
    [data-testid="stMetric"] {background: rgba(197,132,247,0.06); border: 1px solid rgba(197,132,247,0.25);
        border-radius: 12px; padding: 14px 18px;}
    [data-testid="stMetricLabel"] p {font-size: 0.85rem; opacity: 0.8;}
    </style>
    """,
    unsafe_allow_html=True,
)


def style(fig, title, x, y, height=420):
    fig.update_layout(
        title=title, xaxis_title=x, yaxis_title=y, height=height, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


def vline(fig, x, label, color, dash="dash", pos="top right"):
    fig.add_vline(x=x, line_dash=dash, line_color=color, annotation_position=pos,
                  annotation_text=label, annotation_font_color=color)


# ---------------- Sidebar inputs ----------------
with st.sidebar:
    st.header("Option Parameters")
    option_type = st.segmented_control("Option type", ["call", "put"], default="call",
                                       format_func=str.capitalize) or "call"
    S = st.number_input("Spot price  S₀", min_value=0.01, value=100.0, step=1.0)
    K = st.number_input("Strike price  K", min_value=0.01, value=100.0, step=1.0)
    T = st.slider("Time to maturity  T (years)", 0.01, 5.0, 1.0, 0.01)
    r = st.slider("Risk-free rate  r", 0.0, 0.20, 0.05, 0.005, format="%.3f")
    sigma = st.slider("Volatility  σ", 0.01, 1.50, 0.20, 0.01)

    st.header("Simulation")
    simulations = st.slider("Monte Carlo paths", 1_000, 100_000, 20_000, 1_000)
    confidence = st.select_slider("Risk confidence level", [0.90, 0.95, 0.975, 0.99], value=0.95,
                                  format_func=lambda c: f"{c:.1%}")
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

# ---------------- Core computations ----------------
bs_price = black_scholes(S, K, T, r, sigma, option_type)
g = greeks(S, K, T, r, sigma, option_type)

np.random.seed(seed)
mc_price, S_T = monte_carlo_option(S, K, T, r, sigma, simulations, option_type)
payoffs = np.maximum(S_T - K, 0) if option_type == "call" else np.maximum(K - S_T, 0)
disc_payoffs = np.exp(-r * T) * payoffs
std_err = disc_payoffs.std(ddof=1) / np.sqrt(simulations)

intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
prob_itm = norm.cdf(d2) if option_type == "call" else norm.cdf(-d2)

# P&L of buying one option at the Black-Scholes price and holding to expiry (in today's dollars)
pnl = disc_payoffs - bs_price
VaR = -np.percentile(pnl, (1 - confidence) * 100)
ES = -pnl[pnl <= -VaR].mean()

# ---------------- Header & KPIs ----------------
st.title("Option Pricing & Risk Lab")
st.caption(f"European **{option_type}** · S₀ = {S:,.2f} · K = {K:,.2f} · "
           f"T = {T:.2f}y · r = {r:.2%} · σ = {sigma:.0%}")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Black-Scholes price", f"${bs_price:,.4f}", help="Closed-form analytic price.")
k2.metric("Monte Carlo price", f"${mc_price:,.4f}", f"{mc_price - bs_price:+.4f} vs BS",
          delta_color="off", delta_arrow="off", help="Mean of discounted simulated payoffs.")
k3.metric("MC 95% interval", f"± ${1.96 * std_err:,.4f}",
          help="1.96 × standard error. Shrinks with √N — 4× the paths halves it.")
k4.metric("Time value", f"${bs_price - intrinsic:,.4f}", f"intrinsic ${intrinsic:,.2f}",
          delta_color="off", delta_arrow="off",
          help="Intrinsic = value if exercised now. Time value = the rest of the price: what you pay for optionality.")
k5.metric("Prob. finishing ITM", f"{prob_itm:.1%}", help="Risk-neutral N(d₂) (or N(−d₂) for puts).")

st.write("")
tab_price, tab_mc, tab_sens, tab_risk, tab_theory = st.tabs(
    ["Pricing & Greeks", "Monte Carlo", "Sensitivity", "Risk", "Theory"]
)

# ---------------- Pricing & Greeks ----------------
with tab_price:
    cols = st.columns(5)
    greek_help = {
        "Delta": "Change in option price per $1 move in the underlying. Also the hedge ratio.",
        "Gamma": "Change in delta per $1 move. Highest at-the-money near expiry.",
        "Vega": "Change in price per 1 percentage-point rise in volatility.",
        "Theta": "Change in price per calendar day passing (time decay).",
        "Rho": "Change in price per 1 percentage-point rise in interest rates.",
    }
    for col, (name, val) in zip(cols, g.items()):
        col.metric(name, f"{val:+.4f}", help=greek_help[name])

    spots = np.linspace(max(0.01, S * 0.4), S * 1.6, 300)
    expiry_payoff = np.maximum(spots - K, 0) if option_type == "call" else np.maximum(K - spots, 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=expiry_payoff, name="Payoff at expiry",
                             line=dict(color=MUTED, dash="dot")))
    fig.add_trace(go.Scatter(x=spots, y=black_scholes(spots, K, T, r, sigma, option_type),
                             name="Value today (BS)", line=dict(color=ACCENT, width=3),
                             fill="tonexty", fillcolor="rgba(197,132,247,0.10)"))
    fig.add_trace(go.Scatter(x=[S], y=[bs_price], mode="markers", name="Current",
                             marker=dict(size=12, color=ACCENT, line=dict(color="white", width=2))))
    vline(fig, K, "Strike", BLUE)
    st.plotly_chart(style(fig, "Option value vs. spot — the shaded gap is time value", "Spot price", "Option value"),
                    width="stretch")
    st.caption("Before expiry the option is worth more than its payoff because volatility can still move it "
               "into (or deeper into) the money. That premium decays to zero as T → 0.")

# ---------------- Monte Carlo ----------------
with tab_mc:
    left, right = st.columns(2)

    paths = simulate_paths(S, T, r, sigma, n_paths=150, n_steps=200, seed=seed)
    t_grid = np.linspace(0, T, paths.shape[1])
    fig = go.Figure()
    for p in paths:
        fig.add_trace(go.Scattergl(x=t_grid, y=p, mode="lines", line=dict(width=0.7, color=ACCENT),
                                   opacity=0.25, showlegend=False, hoverinfo="skip"))
    for q, dash in [(5, "dot"), (50, "solid"), (95, "dot")]:
        fig.add_trace(go.Scatter(x=t_grid, y=np.percentile(paths, q, axis=0), name=f"{q}th pct",
                                 line=dict(color="white", width=2, dash=dash)))
    fig.add_hline(y=K, line_dash="dash", line_color=BLUE, annotation_text="Strike", annotation_position="bottom left")
    left.plotly_chart(style(fig, "Simulated GBM price paths (150 shown)", "Time (years)", "Price"),
                      width="stretch")

    itm = (S_T > K) if option_type == "call" else (S_T < K)
    fig = go.Figure()
    bins = dict(start=float(S_T.min()), end=float(S_T.max()), size=float(np.ptp(S_T) / 80))
    fig.add_trace(go.Histogram(x=S_T[~itm], xbins=bins, name="Expires worthless", marker_color=MUTED))
    fig.add_trace(go.Histogram(x=S_T[itm], xbins=bins, name="In the money", marker_color=ACCENT))
    vline(fig, K, "Strike", BLUE, pos="top left")
    vline(fig, S * np.exp(r * T), "Forward", GREEN, "dot")
    fig.update_layout(barmode="stack", bargap=0.02)
    right.plotly_chart(style(fig, f"Terminal prices S_T — {itm.mean():.1%} finish ITM", "Price at maturity", "Count"),
                       width="stretch")

    # Running estimate shows the Law of Large Numbers at work
    n = np.arange(1, simulations + 1)
    running = np.cumsum(disc_payoffs) / n
    band = 1.96 * disc_payoffs.std(ddof=1) / np.sqrt(n)
    idx = np.unique(np.geomspace(10, simulations, 400).astype(int)) - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n[idx], y=(running + band)[idx], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=n[idx], y=(running - band)[idx], line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(197,132,247,0.18)", name="95% confidence band"))
    fig.add_trace(go.Scatter(x=n[idx], y=running[idx], name="MC estimate", line=dict(color=ACCENT, width=2)))
    fig.add_hline(y=bs_price, line_dash="dash", line_color=GREEN, annotation_text="Black-Scholes")
    fig.update_xaxes(type="log")
    st.plotly_chart(style(fig, "Convergence: Monte Carlo estimate → Black-Scholes as paths grow",
                          "Number of paths (log scale)", "Price estimate", 380), width="stretch")

# ---------------- Sensitivity ----------------
with tab_sens:
    left, right = st.columns(2)

    spot_axis = np.linspace(S * 0.6, S * 1.4, 25)
    vol_axis = np.linspace(max(0.05, sigma * 0.3), sigma * 2.0, 25)
    grid = black_scholes(spot_axis[None, :], K, T, r, vol_axis[:, None], option_type)
    fig = go.Figure(go.Heatmap(x=spot_axis, y=vol_axis, z=grid, colorscale="Purples",
                               colorbar=dict(title="Price"),
                               hovertemplate="S=%{x:.1f}<br>σ=%{y:.2f}<br>price=%{z:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[S], y=[sigma], mode="markers", name="Current",
                             marker=dict(symbol="x", size=14, color="white")))
    left.plotly_chart(style(fig, "Price surface: spot × volatility", "Spot price", "Volatility σ"),
                      width="stretch")

    greek_name = right.segmented_control("Greek", list(g), default="Delta") or "Delta"
    spots = np.linspace(S * 0.5, S * 1.5, 300)
    fig = go.Figure()
    for frac, color in [(1.0, ACCENT), (0.5, BLUE), (0.1, GREEN)]:
        Tm = max(T * frac, 0.005)
        fig.add_trace(go.Scatter(x=spots, y=greeks(spots, K, Tm, r, sigma, option_type)[greek_name],
                                 name=f"T = {Tm:.2f}y", line=dict(color=color, width=2)))
    vline(fig, K, "Strike", MUTED)
    right.plotly_chart(style(fig, f"{greek_name} across spot, for shrinking maturities", "Spot price", greek_name),
                       width="stretch")
    st.caption("As expiry approaches, Greeks concentrate around the strike: delta becomes a step, "
               "gamma and theta spike. That is why near-dated at-the-money options are hardest to hedge.")

# ---------------- Risk ----------------
with tab_risk:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Premium paid", f"${bs_price:,.2f}")
    c2.metric(f"VaR {confidence:.1%}", f"${VaR:,.2f}", help="Loss not exceeded with this confidence.")
    c3.metric(f"Expected Shortfall {confidence:.1%}", f"${ES:,.2f}", help="Average loss in the worst tail beyond VaR.")
    c4.metric("Prob. of losing money", f"{(pnl < 0).mean():.1%}")

    x_max = np.percentile(pnl, 99.5)
    pbins = dict(start=-bs_price - 0.5, end=x_max, size=(x_max + bs_price) / 70)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pnl[pnl < 0], xbins=pbins, name="Loss", marker_color=RED))
    fig.add_trace(go.Histogram(x=pnl[pnl >= 0], xbins=pbins, name="Profit", marker_color=GREEN))
    vline(fig, -VaR, f"VaR {confidence:.0%}", "white")
    vline(fig, -ES, "ES", ACCENT, "dot", "bottom left")
    fig.update_layout(barmode="overlay", bargap=0.02)
    fig.update_xaxes(range=[-bs_price * 1.15, x_max])
    st.plotly_chart(style(fig, "P&L distribution of a long option held to expiry (discounted)",
                          "Profit / loss ($)", "Count"), width="stretch")
    st.info(
        "For a **bought** option the worst case is losing the whole premium, so VaR and ES are capped at "
        f"${bs_price:,.2f}. The tall bar on the left is every path that expires worthless — "
        "a long option's loss distribution is lumpy, not bell-shaped. When more than (1 − confidence) of paths "
        "expire worthless, VaR and ES both equal the full premium.",
    )

# ---------------- Theory ----------------
with tab_theory:
    a, b = st.columns(2)
    with a:
        st.subheader("Black-Scholes")
        st.latex(r"C = S_0 N(d_1) - K e^{-rT} N(d_2)")
        st.latex(r"P = K e^{-rT} N(-d_2) - S_0 N(-d_1)")
        st.latex(r"d_1 = \frac{\ln(S_0/K) + (r + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}},\quad d_2 = d_1 - \sigma\sqrt{T}")
        st.markdown("Assumes lognormal prices, constant σ and r, no dividends, frictionless markets, "
                    "and European (expiry-only) exercise.")
        st.subheader("Greeks")
        st.latex(r"\Delta = \frac{\partial V}{\partial S},\ \Gamma = \frac{\partial^2 V}{\partial S^2},\ "
                 r"\mathcal{V} = \frac{\partial V}{\partial \sigma},\ \Theta = \frac{\partial V}{\partial t},\ "
                 r"\rho = \frac{\partial V}{\partial r}")
    with b:
        st.subheader("Monte Carlo")
        st.latex(r"S_T = S_0 \exp\!\left((r - \tfrac{1}{2}\sigma^2)T + \sigma\sqrt{T}\,Z\right),\ Z \sim N(0,1)")
        st.latex(r"\hat V = e^{-rT}\,\frac{1}{N}\sum_{i=1}^{N} \text{payoff}(S_T^{(i)}),\quad "
                 r"\text{SE} = \frac{s}{\sqrt{N}}")
        st.markdown("Simulates the risk-neutral terminal price and averages discounted payoffs. "
                    "Slower than closed form, but generalises to path-dependent and exotic payoffs.")
        st.subheader("Risk metrics")
        st.latex(r"\text{VaR}_\alpha = -q_{1-\alpha}(\text{P\&L}),\quad "
                 r"\text{ES}_\alpha = -\mathbb{E}[\text{P\&L} \mid \text{P\&L} \le -\text{VaR}_\alpha]")
        st.markdown("Computed empirically from the simulated P&L of buying the option at its fair price.")
