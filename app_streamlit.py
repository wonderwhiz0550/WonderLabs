import streamlit as st
from ExpectationInvesting_Code import evaluate_stock, config

st.set_page_config(page_title="Stock Valuation Tool", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(to right, #232526, #414345);
        color: #ffffff;
    }

    .card {
        background: rgba(255, 255, 255, 0.07);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }

    .stTabs [role="tab"] {
        background-color: #2a2d34;
        color: #ffffff;
        padding: 10px;
        border-radius: 10px 10px 0 0;
        margin-right: 2px;
    }

    .stTabs [role="tab"][aria-selected="true"] {
        background: #1f77b4;
        font-weight: bold;
        color: white;
    }

    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
    }

    .stButton>button:hover {
        background-color: #145374;
    }

    .metric-style div {
        font-size: 1.3rem !important;
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>📊 Stock Valuation Tool</h1>", unsafe_allow_html=True)

# Layout: Inputs | Outputs
left, right = st.columns([2, 3])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    tabs = st.tabs(["🏢 Stock", "📈 Growth", "💸 Discount", "🎲 Simulation"])

    with tabs[0]:  # Stock
        ticker = st.text_input("Stock Ticker", value="MSFT")

    with tabs[1]:  # Growth
        default_analyst_growth_rate = st.slider("Analyst Growth Rate", 0.01, 0.2, 0.07, 0.001)
        high_growth_period = st.slider("High Growth Period (Years)", 1, 15, 8)
        transition_period = st.slider("Transition Period (Years)", 1, 15, 5)
        terminal_growth_rate = st.slider("Terminal Growth Rate", 0.0, 0.1, 0.035, 0.001)

    with tabs[2]:  # Discount
        risk_free_rate = st.slider("Risk-Free Rate", 0.0, 0.1, 0.035, 0.001)
        market_return = st.slider("Market Return", 0.0, 0.2, 0.095, 0.001)
        default_discount_rate = st.slider("Default Discount Rate", 0.0, 0.2, 0.011, 0.001)

    with tabs[3]:  # Simulation
        sensitivity_range = st.slider("Sensitivity Range", 0.001, 0.1, 0.04, 0.001)
        num_monte_carlo_sims = st.slider("Monte Carlo Simulations", 100, 20000, 10000, step=100)
        exit_multiple = st.slider("Exit Multiple", 5, 40, 20)
        terminal_method = st.radio("Terminal Value Method", ["perpetual_growth", "exit_multiple"], index=1)
    st.markdown('</div>', unsafe_allow_html=True)

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

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if st.button("🚀 Run Valuation"):
        with st.spinner("Running Monte Carlo Simulation..."):
            result, status, plot_path = evaluate_stock(ticker.upper(), config)
            if status != "Success":
                st.error(status)
            else:
                st.success("✅ Valuation Complete")
                cols = st.columns(3)
                with cols[0]:
                    st.metric("📌 Current Price", f"${result['stock_price']:.2f}")
                    st.metric("📈 Revenue", f"${result['revenue']:,.0f}")
                with cols[1]:
                    st.metric("💰 Simulated Price", f"${result['mean_simulated_price']:.2f}")
                    st.metric("💸 Free Cash Flow", f"${result['free_cash_flow']:,.0f}")
                with cols[2]:
                    st.metric("📊 Valuation", result['valuation_status'])
                    st.metric("🧮 FCF Margin", f"{result['fcf_margin']:.2%}")
                st.image(plot_path, caption="📉 Monte Carlo Histogram", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
