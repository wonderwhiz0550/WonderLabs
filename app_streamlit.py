import os

import streamlit as st

from ExpectationInvesting_Code import evaluate_stock, config, MARKET_PRESETS

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


def format_large_number(number, sym="$"):
    if abs(number) >= 1_000_000_000_000:
        return f"{sym}{number / 1_000_000_000_000:.2f}T"
    elif abs(number) >= 1_000_000_000:
        return f"{sym}{number / 1_000_000_000:.2f}B"
    elif abs(number) >= 1_000_000:
        return f"{sym}{number / 1_000_000:.2f}M"
    else:
        return f"{sym}{number:,.0f}"


def render_results(result, status, plot_path):
    if status != "Success":
        st.error(status)
        return

    sym = result.get("currency_symbol", "$")
    name = result.get("company_name") or result.get("ticker", "")
    st.success(f"✅ Valuation Complete: {name}")

    intrinsic = result.get("median_simulated_price", result["mean_simulated_price"])

    cols = st.columns(3)
    with cols[0]:
        st.metric("📌 Current Price", f"{sym}{result['stock_price']:,.2f}")
        st.metric("📈 Revenue", format_large_number(result["revenue"], sym))
    with cols[1]:
        st.metric("💰 Intrinsic Value (median)", f"{sym}{intrinsic:,.2f}")
        fcf_label = "💸 FCF (after SBC)" if result.get("sbc_adjusted") else "💸 Free Cash Flow"
        st.metric(fcf_label, format_large_number(result["free_cash_flow"], sym))
    with cols[2]:
        st.metric("📊 Valuation", result["valuation_status"])
        st.metric("🧮 FCF Margin", f"{result['fcf_margin']:.2%}")

    if "prob_undervalued" in result:
        cols2 = st.columns(3)
        with cols2[0]:
            st.metric("🎯 Chance value > price", f"{result['prob_undervalued']:.0%}")
        with cols2[1]:
            mos = result.get("margin_of_safety")
            st.metric("🛡️ Margin of Safety", f"{mos:.0%}" if mos is not None else "n/a")
        with cols2[2]:
            st.metric("📉 Cost of Equity", f"{result['discount_rate']:.2%}")

        cl = int(round(result["confidence_level"] * 100))
        beta = f"{result['beta']:.2f}" if result["beta"] is not None else "n/a"
        st.caption(
            f"{cl}% simulated range: {sym}{result['ci_lower']:,.0f} to {sym}{result['ci_upper']:,.0f}  ·  "
            f"Base case: {sym}{result['base_case_price']:,.0f}  ·  "
            f"Growth used: {result['growth_used']:.1%}  ·  "
            f"Beta: {beta} ({result['beta_source']})  ·  "
            f"Data basis: {result['data_basis']}"
        )

    for w in result.get("warnings", []):
        st.warning(w)

    if plot_path and os.path.exists(plot_path):
        st.image(plot_path, caption="📉 Monte Carlo Histogram", use_container_width=True)

    # GenAI multi-dimensional analysis
    if result.get("genai_analysis"):
        st.markdown("---")
        st.markdown("### 🤖 Strategic Analysis")
        with st.container(border=True):
            # Escape "$" so Streamlit doesn't treat text between dollar signs as LaTeX
            st.markdown(result["genai_analysis"].replace("$", "\\$"))
    elif result.get("genai_error"):
        st.info(f"Strategic analysis unavailable. {result['genai_error']}")


# Title
st.markdown("<h1 style='text-align: center;'>📊 Stock Valuation Tool</h1>", unsafe_allow_html=True)

# Layout: Inputs | Outputs
left, right = st.columns([2, 3])

with left:
    with st.container(border=True):
        tabs = st.tabs(["🏢 Stock", "📈 Growth", "💸 Discount", "🎲 Simulation"])

        with tabs[0]:  # Stock
            market = st.radio(
                "Market", list(MARKET_PRESETS),
                format_func=lambda m: MARKET_PRESETS[m]["label"], horizontal=True)
            preset = MARKET_PRESETS[market]
            if market == "US":
                ticker = st.text_input("Stock Ticker", value="AAPL")
                st.caption("Use Yahoo Finance symbols, e.g. AAPL, MSFT, BRK-B (BRK.B also works).")
            else:
                ticker = st.text_input("Stock Ticker", value="TCS.NS")
                st.caption("NSE symbols; .NS is added automatically if you leave it off.")
            adjust_sbc = st.checkbox(
                "Deduct stock-based compensation from FCF", value=bool(config["adjust_for_sbc"]),
                help="Recommended for US tech and other companies with heavy share-based pay.")

        with tabs[1]:  # Growth
            growth_source = st.radio(
                "Base growth source",
                ["user", "historical"],
                format_func=lambda x: "Slider below" if x == "user" else "Historical revenue CAGR (3y)",
                horizontal=True,
            )
            default_analyst_growth_rate = st.slider(
                "Analyst Growth Rate", 0.01, 0.2, float(preset["default_analyst_growth_rate"]), 0.001)
            high_growth_period = st.slider("High Growth Period (Years)", 1, 15, int(config["high_growth_period"]))
            transition_period = st.slider("Transition Period (Years)", 1, 15, int(config["transition_period"]))
            terminal_growth_rate = st.slider(
                "Terminal Growth Rate", 0.0, 0.1, float(preset["terminal_growth_rate"]), 0.001)

        with tabs[2]:  # Discount
            risk_free_rate = st.slider("Risk-Free Rate", 0.0, 0.1, float(preset["risk_free_rate"]), 0.001)
            market_return = st.slider("Market Return", 0.0, 0.2, float(preset["market_return"]), 0.001)
            default_discount_rate = st.slider(
                "Default Discount Rate (used if beta is unavailable)", 0.0, 0.2,
                float(preset["default_discount_rate"]), 0.001)

        with tabs[3]:  # Simulation
            sensitivity_range = st.slider(
                "Sensitivity Range", 0.001, 0.1, float(config["sensitivity_range"]), 0.001)
            num_monte_carlo_sims = st.slider("Monte Carlo Simulations", 100, 20000, 10000, step=100)
            exit_multiple = st.slider("Exit Multiple (P/FCF)", 5, 40, int(config["exit_multiple"]))
            terminal_method = st.radio("Terminal Value Method", ["perpetual_growth", "exit_multiple"], index=0)

# Per-run settings. A copy is used so concurrent users never overwrite each other's settings.
run_cfg = {
    **config,
    "market": market,
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
    "exit_multiple": exit_multiple,
    "growth_source": growth_source,
    "adjust_for_sbc": adjust_sbc,
}

with right:
    with st.container(border=True):
        if st.button("🚀 Run Valuation"):
            with st.spinner("Running Monte Carlo Simulation..."):
                st.session_state["last_run"] = evaluate_stock(ticker, run_cfg)

        last = st.session_state.get("last_run")
        if last:
            st.caption("Results reflect the settings at the time of the last run.")
            render_results(*last)
