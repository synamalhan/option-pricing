"""Run with: python test_models.py — checks pricing against put-call parity and Greeks against finite differences."""
import numpy as np

from models import black_scholes, greeks, monte_carlo_option

S, K, T, r, v = 100, 105, 0.75, 0.04, 0.25
args = dict(S=S, K=K, T=T, r=r, sigma=v)

c, p = black_scholes(**args, option_type="call"), black_scholes(**args, option_type="put")
assert abs(c - p - (S - K * np.exp(-r * T))) < 1e-9, "put-call parity"

for t in ("call", "put"):
    g, h = greeks(**args, option_type=t), 1e-4
    f = lambda **kw: black_scholes(**{**args, **kw}, option_type=t)
    assert abs(g["Delta"] - (f(S=S + h) - f(S=S - h)) / (2 * h)) < 1e-6
    assert abs(g["Gamma"] - (f(S=S + 1e-2) - 2 * f() + f(S=S - 1e-2)) / 1e-4) < 1e-5
    assert abs(g["Vega"] - (f(sigma=v + h) - f(sigma=v - h)) / (2 * h) / 100) < 1e-6
    assert abs(g["Rho"] - (f(r=r + h) - f(r=r - h)) / (2 * h) / 100) < 1e-6
    assert abs(g["Theta"] + (f(T=T + h) - f(T=T - h)) / (2 * h) / 365) < 1e-6

    np.random.seed(0)
    mc, _ = monte_carlo_option(**args, simulations=200_000, option_type=t)
    assert abs(mc - f()) < 0.1, f"MC {t} {mc} vs BS {f()}"

print("all checks passed")
