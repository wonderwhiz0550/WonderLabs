import streamlit as st
from ExpectationInvesting_Code import evaluate_stock, config
from PIL import Image

st.set_page_config(page_title="Stock Valuation App", layout="centered")

st.title("📊 Stock Valuation Tool")
st.markdown("Evaluate a stock using multi-stage DCF and Monte Carlo simulation.")

# Input fields
ticker = st.text_input("Stock Ticker", value="MSFT")

col1, col2 = st.columns(2)
with col1:
    risk_free_rate = st.slider("Risk-Free Rate", 0.0, 0.1, 0.035, 0.001)
    market_return = st.slider("Market Return", 0.0, 0.2, 0.095, 0.001)
    terminal_growth_rate = st.slider("Terminal Growth Rate", 0.0, 0.1, 0.035, 0.001)
    default_discount_rate = st.slider("Default Discount Rate", 0.0, 0.2, 0.011, 0.001)
    default_analyst_growth_rate = st.slider("Default Analyst Growth Rate", 0.01, 0.2, 0.07, 0.001)
with col2:
    high_growth_period = st.slider("High Growth Period (Years)", 1, 15, 8)
    transition_period = st.slider("Transition Period (Years)", 1, 15, 5)
    sensitivity_range = st.slider("Sensitivity Range", 0.001, 0.1, 0.04, 0.001)
    num_monte_carlo_sims = st.slider("Monte Carlo Simulations", 100, 20000, 10000, step=100)
    exit_multiple = st.slider("Exit Multiple", 5, 40, 20)

terminal_method = st.radio("Terminal Value Method", ["perpetual_growth", "exit_multiple"], index=1)

# Update config
config.update({
    "risk_free_rate": risk_free_rate,
    "market_return": market_return,
    "terminal_growth_rate": terminal_growth_rate,
    "default_discount_rate": default_discount_rate,
    "default_analyst_growth_rate": default_analyst_growth_rate,
    "high_growth_period": high_growth_period,
    "transition_period": transition_period,
    "sensitivity_range": sensitivity_range,
    "num_monte_carlo_sims": int(num_monte_carlo_sims),
    "terminal_method": terminal_method,
    "exit_multiple": exit_multiple
})

# Run analysis
if st.button("Run Valuation"):
    with st.spinner("Running Monte Carlo Simulation..."):
        result, status, plot_path = evaluate_stock(ticker.upper(), config)
        if status != "Success":
            st.error(status)
        else:
            st.success("Valuation Complete")
            st.metric("Current Stock Price", f"${result['stock_price']:.2f}")
            st.metric("Mean Simulated Price", f"${result['mean_simulated_price']:.2f}")
            st.metric("Valuation Status", result['valuation_status'])
            st.metric("Revenue", f"${result['revenue']:,.0f}")
            st.metric("Free Cash Flow", f"${result['free_cash_flow']:,.0f}")
            st.metric("FCF Margin", f"{result['fcf_margin']:.2%}")
            st.image(plot_path, caption="Monte Carlo Simulation Histogram", use_column_width=True)
