import numpy as np
from scipy.stats import norm 

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return price


def monte_carlo_option(S, K, T, r, sigma, simulations=10000, option_type='call'):
    Z = np.random.standard_normal(simulations)
    S_T = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    price = np.exp(-r * T) * np.mean(payoffs)
    return price, S_T


def greeks(S, K, T, r, sigma, option_type='call'):
    """Analytic Black-Scholes Greeks. Vega/rho per 1% move, theta per calendar day."""
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc = K * np.exp(-r * T)
    gamma = norm.pdf(d1) / (S * sigma * sqrtT)
    vega = S * norm.pdf(d1) * sqrtT / 100
    decay = -S * norm.pdf(d1) * sigma / (2 * sqrtT)
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (decay - r * disc * norm.cdf(d2)) / 365
        rho = T * disc * norm.cdf(d2) / 100
    elif option_type == 'put':
        delta = norm.cdf(d1) - 1
        theta = (decay + r * disc * norm.cdf(-d2)) / 365
        rho = -T * disc * norm.cdf(-d2) / 100
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}


def simulate_paths(S, T, r, sigma, n_paths=100, n_steps=252, seed=None):
    """Full GBM price paths, shape (n_paths, n_steps + 1), for visualisation."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_steps = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    return S * np.exp(np.hstack([np.zeros((n_paths, 1)), np.cumsum(log_steps, axis=1)]))
