import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from models import black_scholes, monte_carlo_option

# App Title
st.title("Options Pricing and Risk Assessment Tool")

# Sidebar for Inputs
st.sidebar.header("Option Parameters")
S = st.sidebar.number_input("Current Stock Price (S)", min_value=0.0, value=100.0)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=100.0)
T = st.sidebar.number_input("Time to Maturity (T, in years)", min_value=0.01, value=1.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, value=0.2, step=0.01)
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
simulations = st.sidebar.slider("Monte Carlo Simulations", 1000, 50000, 10000)

# Black-Scholes Pricing
st.subheader("Black-Scholes Pricing")
bs_price = black_scholes(S, K, T, r, sigma, option_type)
st.write(f"The {option_type.capitalize()} option price using Black-Scholes Model is: **${bs_price:.2f}**")

# Monte Carlo Simulation
st.subheader("Monte Carlo Simulation")
mc_price, S_T = monte_carlo_option(S, K, T, r, sigma, simulations, option_type)
st.write(f"The {option_type.capitalize()} option price using Monte Carlo Simulation is: **${mc_price:.2f}**")

# Plot the Distribution of Simulated Prices
st.subheader("Distribution of Simulated Prices at Maturity")
fig, ax = plt.subplots()
ax.hist(S_T, bins=50, color="skyblue", edgecolor="black")
ax.set_title("Distribution of Simulated Asset Prices")
ax.set_xlabel("Price")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Value at Risk (VaR) and Expected Shortfall (ES)
st.subheader("Risk Assessment Metrics")
VaR_95 = np.percentile(S_T, 5)
ES_95 = S_T[S_T <= VaR_95].mean()

st.write(f"**Value at Risk (VaR 95%)**: ${VaR_95:.2f}")
st.write(f"**Expected Shortfall (ES 95%)**: ${ES_95:.2f}")
