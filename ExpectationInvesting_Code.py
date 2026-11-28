"""
Expectation Investing - valuation engine (US default, India optional).

Method: free-cash-flow-to-equity (FCFE) DCF, discounted at cost of equity (CAPM),
with a Monte Carlo layer over growth, discount rate, terminal growth, FCF margin
and exit multiple. Because FCFE is already after interest and debt service, the
result is an equity value and no net-debt bridge is applied.

Markets: set config["market"] to "US" (default) or "IN". Each market has a preset
(risk-free rate, equity return, terminal growth, benchmark index, currency, ticker
conventions) in MARKET_PRESETS.

Public API: evaluate_stock(ticker, config) -> (result | None, status, plot_path | None)
"""
import glob
import os
import re
import tempfile
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")  # headless backend, safe on servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# --------------------------------------------------------------------------- #
# Market presets. The front end uses these as slider defaults.
# --------------------------------------------------------------------------- #
MARKET_PRESETS = {
    "US": {
        "label": "United States",
        "currency_symbol": "$",
        "benchmark": "^GSPC",
        "benchmark_name": "S&P 500",
        "suffix": "",                        # US tickers need no exchange suffix
        "timezone": "America/New_York",
        "risk_free_rate": 0.05,              # US 10Y Treasury ~5.0% (Nov 2026)
        "market_return": 0.095,              # ~4.5% equity risk premium over the risk-free rate
        "terminal_growth_rate": 0.03,        # long-run nominal growth, below the risk-free rate
        "default_discount_rate": 0.09,       # fallback cost of equity if beta unavailable
        "default_analyst_growth_rate": 0.07,
        "cite_example": "[Q3 2026 10-Q]",
        "research_note": ("Use SEC filings (10-K, 10-Q, 8-K), earnings calls and consensus estimates. "
                          "Frame macro around the Fed, US inflation and tariffs, and legal risk around "
                          "DOJ/FTC/SEC where relevant. Quote all figures in USD."),
    },
    "IN": {
        "label": "India",
        "currency_symbol": "₹",
        "benchmark": "^NSEI",
        "benchmark_name": "Nifty 50",
        "suffix": ".NS",                     # appended when the ticker has no "."
        "timezone": "Asia/Kolkata",
        "risk_free_rate": 0.072,             # India 10Y G-Sec ~7.2%
        "market_return": 0.12,
        "terminal_growth_rate": 0.05,
        "default_discount_rate": 0.11,
        "default_analyst_growth_rate": 0.08,
        "cite_example": "[Q2 FY27 results]",
        "research_note": ("Use BSE/NSE filings, results press releases and consensus estimates. "
                          "Frame macro around RBI policy, Indian inflation and the INR, and legal risk around "
                          "CCI/SEBI/RBI where relevant. Quote all figures in INR."),
    },
}

# --------------------------------------------------------------------------- #
# Configuration. The front end may override any key.
# --------------------------------------------------------------------------- #
config = {
    "market": "US",
    # market-dependent defaults (taken from the US preset)
    "risk_free_rate": MARKET_PRESETS["US"]["risk_free_rate"],
    "market_return": MARKET_PRESETS["US"]["market_return"],
    "terminal_growth_rate": MARKET_PRESETS["US"]["terminal_growth_rate"],
    "default_discount_rate": MARKET_PRESETS["US"]["default_discount_rate"],
    "default_analyst_growth_rate": MARKET_PRESETS["US"]["default_analyst_growth_rate"],
    # shared
    "high_growth_period": 8,              # years at base growth
    "transition_period": 5,               # years fading linearly to terminal growth
    "sensitivity_range": 0.04,            # std-dev of growth; other std-devs scale from this
    "num_monte_carlo_sims": 10000,
    "terminal_method": "exit_multiple",   # "exit_multiple" (P/FCF) or "perpetual_growth"
    "exit_multiple": 20,                  # terminal equity value / terminal-year FCF
    "confidence_level": 0.95,
    "growth_source": "user",              # "user" = default_analyst_growth_rate, "historical" = revenue CAGR
    "adjust_for_sbc": False,              # deduct stock-based compensation from FCF (matters for US tech)
    "min_spread": 0.02,                   # enforce discount_rate - terminal_growth >= 2%
    "random_seed": None,                  # set an int for reproducible simulations
    "gemini_model": "gemini-2.5-flash",
}
_DEFAULTS = dict(config)  # snapshot so callers' mutations of `config` never corrupt defaults


def _preset(cfg):
    return MARKET_PRESETS.get(cfg.get("market", "US"), MARKET_PRESETS["US"])


def _money(x, sym):
    for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(x) >= div:
            return f"{sym}{x / div:,.2f}{unit}"
    return f"{sym}{x:,.0f}"


# --------------------------------------------------------------------------- #
# Data fetching
# --------------------------------------------------------------------------- #
def normalize_ticker(raw, cfg):
    """US: uppercase, and share-class tickers use Yahoo's dash form (BRK.B -> BRK-B).
    India: append the .NS suffix when no exchange suffix is given."""
    t = (raw or "").strip().upper().replace(" ", "")
    market = cfg.get("market", "US")
    suffix = _preset(cfg)["suffix"]
    if t and suffix and "." not in t and not t.startswith("^"):
        t += suffix
    if market == "US" and re.fullmatch(r"[A-Z]{1,5}\.[ABC]", t):
        t = t.replace(".", "-")
    return t


def _ok(x):
    return x is not None and isinstance(x, (int, float, np.floating, np.integer)) and np.isfinite(x)


def _pick(df, names, mode):
    """First matching row. mode='ttm' sums the latest 4 quarters (all 4 must be present);
    mode='annual' takes the latest fiscal year. Returns None if unavailable."""
    if df is None or getattr(df, "empty", True):
        return None
    df = df.sort_index(axis=1, ascending=False)  # newest column first
    for name in names:
        if name in df.index:
            row = df.loc[name]
            if mode == "ttm":
                row = row.iloc[:4]
                if len(row) == 4 and row.notna().all():
                    return float(row.sum())
            else:
                val = row.iloc[0]
                if pd.notna(val):
                    return float(val)
    return None


def _free_cash_flow(cf, mode):
    """Returns (fcf, capex_found). Yahoo reports capex as a negative number, so use -abs()."""
    fcf = _pick(cf, ["Free Cash Flow"], mode)
    if fcf is not None:
        return fcf, True
    ocf = _pick(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"], mode)
    if ocf is None:
        return None, False
    capex = _pick(cf, ["Capital Expenditure"], mode)
    return ocf - abs(capex if capex is not None else 0.0), capex is not None


def _estimate_beta(ticker, info_beta, market):
    """Weekly 3y regression vs the market benchmark (S&P 500 for US, Nifty 50 for India);
    falls back to Yahoo's beta, then None."""
    p = MARKET_PRESETS[market]
    bench = p["benchmark"]
    domestic = ("." not in ticker) if market == "US" else ticker.endswith((".NS", ".BO"))
    if domestic:
        try:
            px = yf.download([ticker, bench], period="3y", interval="1wk",
                             progress=False, auto_adjust=True)["Close"].dropna()
            rets = px.pct_change().dropna()
            if len(rets) >= 60 and ticker in rets and bench in rets:
                cov = np.cov(rets[ticker], rets[bench])
                return float(np.clip(cov[0, 1] / cov[1, 1], 0.3, 2.5)), f"regression vs {p['benchmark_name']}"
        except Exception:
            pass
    if _ok(info_beta) and info_beta > 0:
        return float(np.clip(info_beta, 0.3, 2.5)), "Yahoo Finance"
    return None, "unavailable"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_market_data(ticker, market="US", max_retries=3):
    """One pass over yfinance; returns plain python values. Raises (not cached) on repeated failure."""
    last_err = None
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}

            hist = stock.history(period="5d")  # 5d so holidays/weekends don't give an empty frame
            if hist is not None and not hist.empty:
                price = float(hist["Close"].dropna().iloc[-1])
            else:
                price = info.get("currentPrice") or info.get("regularMarketPrice")

            fin_q, cf_q = stock.quarterly_financials, stock.quarterly_cashflow
            fin_a, cf_a = stock.financials, stock.cashflow

            rev_names = ["Total Revenue", "Operating Revenue"]
            rev, (fcf, capex_found), basis = _pick(fin_q, rev_names, "ttm"), _free_cash_flow(cf_q, "ttm"), "TTM"
            if rev is None or fcf is None:  # use one consistent period for both
                rev, (fcf, capex_found), basis = _pick(fin_a, rev_names, "annual"), _free_cash_flow(cf_a, "annual"), "Latest fiscal year"

            # Stock-based compensation, on the same period basis as FCF
            sbc = (_pick(cf_q, ["Stock Based Compensation"], "ttm") if basis == "TTM"
                   else _pick(cf_a, ["Stock Based Compensation"], "annual"))

            rev_hist = []
            if fin_a is not None and not fin_a.empty:
                for name in rev_names:
                    if name in fin_a.index:
                        s = fin_a.sort_index(axis=1, ascending=False).loc[name].dropna()
                        rev_hist = [float(v) for v in s.values]
                        break

            # Share count: market cap / price covers ALL share classes (dual-class names such as
            # GOOGL or BRK-B), whereas Yahoo's sharesOutstanding can cover a single class.
            market_cap = info.get("marketCap")
            shares_info = info.get("sharesOutstanding")
            implied = market_cap / price if _ok(market_cap) and _ok(price) and price > 0 else None
            shares = implied if implied else shares_info
            share_mismatch = bool(implied and _ok(shares_info) and shares_info > 0
                                  and abs(implied / shares_info - 1) > 0.05)

            beta, beta_source = _estimate_beta(ticker, info.get("beta"), market)

            return {
                "price": float(price) if _ok(price) else None,
                "revenue": rev, "fcf": fcf, "capex_found": capex_found, "basis": basis,
                "sbc": sbc,
                "rev_history": rev_hist,
                "shares": float(shares) if _ok(shares) else None,
                "share_mismatch": share_mismatch,
                "market_cap": float(market_cap) if _ok(market_cap) else None,
                "beta": beta, "beta_source": beta_source,
                "name": info.get("longName") or info.get("shortName") or ticker,
                "sector": info.get("sector") or "",
                "currency": info.get("currency"),
                "fin_currency": info.get("financialCurrency"),
            }
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff (Yahoo rate-limits aggressively)
    raise RuntimeError(f"{type(last_err).__name__}: {last_err}")


# --------------------------------------------------------------------------- #
# Valuation maths
# --------------------------------------------------------------------------- #
def cost_of_equity(beta, cfg):
    if beta is None:
        return cfg["default_discount_rate"]
    ke = cfg["risk_free_rate"] + beta * (cfg["market_return"] - cfg["risk_free_rate"])
    return float(np.clip(ke, 0.06, 0.25))


def revenue_cagr(history):
    """history: annual revenues, newest first. CAGR over up to 3 intervals, clipped."""
    vals = [v for v in history if v and v > 0]
    if len(vals) < 2:
        return None
    n = min(len(vals), 4) - 1
    return float(np.clip((vals[0] / vals[n]) ** (1 / n) - 1, -0.10, 0.30))


def dcf_per_share(revenue, margin, g0, r, tg, mult, shares, cfg):
    """Vectorised multi-stage FCFE DCF. margin, g0, r, tg, mult are arrays of shape (n,).
    Growth = g0 for `high_growth_period` years, then fades linearly to tg over `transition_period`."""
    H = int(cfg["high_growth_period"])
    T = max(int(cfg["transition_period"]), 1)
    N = H + T
    years = np.arange(1, N + 1)

    fade = np.clip(years - H, 0, None) / T                       # 0 in high-growth, ramps to 1
    growth = g0[:, None] + (tg - g0)[:, None] * fade[None, :]
    rev = revenue * np.cumprod(1 + growth, axis=1)
    fcf = rev * margin[:, None]

    pv = (fcf / (1 + r[:, None]) ** years[None, :]).sum(axis=1)
    if cfg["terminal_method"] == "perpetual_growth":
        tv = fcf[:, -1] * (1 + tg) / (r - tg)                    # spread is enforced > 0 upstream
    else:  # exit_multiple: P/FCF applied to terminal-year FCF
        tv = fcf[:, -1] * mult
    return (pv + tv / (1 + r) ** N) / shares


def monte_carlo_simulation(revenue, fcf_margin, base_growth, discount_rate,
                           terminal_growth_rate, shares, cfg):
    """Returns ALL simulated per-share values (no filtering, so skew isn't hidden)."""
    rng = np.random.default_rng(cfg.get("random_seed"))
    n = int(cfg["num_monte_carlo_sims"])
    s = cfg["sensitivity_range"]
    scale = s / 0.04  # at the default 0.04: growth sd 4pt, discount 1pt, terminal 0.5pt, margin 15%

    g0 = np.clip(rng.normal(base_growth, s, n), -0.20, 0.40)
    r = np.clip(rng.normal(discount_rate, 0.01 * scale, n), 0.06, 0.30)
    tg = np.clip(rng.normal(terminal_growth_rate, 0.005 * scale, n), 0.0, 0.08)
    tg = np.minimum(tg, r - cfg["min_spread"])                   # guarantees r - tg >= min_spread
    margin = fcf_margin * np.clip(1 + rng.normal(0, 0.15 * scale, n), 0.3, 1.7)
    mult = np.clip(rng.normal(cfg["exit_multiple"], 0.15 * scale * cfg["exit_multiple"], n), 3, 80)

    return dcf_per_share(revenue, margin, g0, r, tg, mult, shares, cfg)


def compare_prices(stock_price, implied_price):
    if implied_price <= 0:
        return "Not Meaningful"
    if stock_price < implied_price * 0.9:
        return "Undervalued"
    if stock_price > implied_price * 1.1:
        return "Overvalued"
    return "Fairly Priced"


# --------------------------------------------------------------------------- #
# Plot (unique file per run - no overwrites between concurrent users)
# --------------------------------------------------------------------------- #
def _cleanup_old_plots(max_age_s=3600):
    now = time.time()
    for p in glob.glob(os.path.join(tempfile.gettempdir(), "valuation_*.png")):
        try:
            if now - os.path.getmtime(p) > max_age_s:
                os.remove(p)
        except OSError:
            pass


def _make_plot(prices, stock_price, median_price, lo, hi, ticker, conf, sym):
    _cleanup_old_plots()
    p1, p99 = np.percentile(prices, [1, 99])
    shown = prices[(prices >= p1) & (prices <= p99)]  # trim extreme tails for readability
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(shown, bins=50, alpha=0.8, color="#1f77b4", edgecolor="black")
    ax.axvline(stock_price, color="red", linestyle="dashed", label=f"Market price ({sym}{stock_price:,.0f})")
    ax.axvline(median_price, color="green", linestyle="dashed", label=f"Median value ({sym}{median_price:,.0f})")
    ax.axvline(lo, color="gray", linestyle="dotted", label=f"{conf:.0%} range ({sym}{lo:,.0f} - {sym}{hi:,.0f})")
    ax.axvline(hi, color="gray", linestyle="dotted")
    ax.set_title(f"Monte Carlo Valuation for {ticker} ({len(prices):,} simulations)")
    ax.set_xlabel(f"Intrinsic value per share ({sym})")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    safe = re.sub(r"[^A-Za-z0-9]+", "_", ticker)
    path = os.path.join(tempfile.gettempdir(), f"valuation_{safe}_{uuid.uuid4().hex[:8]}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Gemini qualitative brief (search-grounded, aware of the DCF result)
# --------------------------------------------------------------------------- #
def _get_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return os.environ.get("GOOGLE_API_KEY")


def _grounding_sources(resp, limit=5):
    out, seen = [], set()
    try:
        for ch in (resp.candidates[0].grounding_metadata.grounding_chunks or []):
            web = getattr(ch, "web", None)
            if web and web.uri and web.uri not in seen:
                seen.add(web.uri)
                out.append(f"[{web.title or 'source'}]({web.uri})")
            if len(out) >= limit:
                break
    except Exception:
        pass
    return out


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _gemini_brief(ticker, company, context, today, model, market, _api_key):
    """Raises on failure so errors are never cached. `_api_key` is excluded from the cache key."""
    from google import genai
    from google.genai import types

    mk = MARKET_PRESETS[market]
    prompt = f"""You are a senior equity research analyst. Produce a decision-oriented investment brief on {company} ({ticker}) as of {today}. Use current data from search (latest results, filings, news). If a figure is unavailable or uncertain, say so rather than estimating. {mk['research_note']}

Our quantitative model (FCFE DCF with Monte Carlo) reports: {context}
Treat this as one input. In the Verdict, state whether you agree or disagree with it and why.

HARD LIMIT: 500 words maximum. No preamble, no disclaimers, no restating the question.

Use exactly these bold markdown headings, in this order, without printing the word budgets:
**1. Snapshot** (40 words): one-line business description, market cap, price, forward P/E (or EV/Sales if unprofitable), one-sentence investment thesis.
**2. Technology & Innovation** (80): durable moats (IP, scale, switching costs, network effects), R&D intensity vs peers, the 1-2 growth vectors that matter most to earnings over 3-5 years.
**3. Geopolitics & Supply Chain** (70): revenue/manufacturing exposure by region, tariff or export-control sensitivity, single-source dependencies. Quantify exposure (% of revenue/COGS) where possible.
**4. Macro Sensitivity** (70): impact of rates, inflation, FX and the cycle on margins, demand and debt servicing. State sensitivity as high/medium/low and why.
**5. Legal & Regulatory** (70): active or probable antitrust, IP, privacy or compliance matters, each with potential financial impact and timeline.
**6. Management & Governance** (70): capital-allocation record, guidance accuracy, insider ownership and recent insider activity, board independence, dual-class or related-party red flags.
**7. Verdict** (100): Rating Buy/Hold/Avoid with confidence (Low/Med/High); bull/base/bear case, one line each with approximate price target and probability; top 3 catalysts (with dates if known); top 3 risks that would invalidate the thesis; what to monitor next quarter.

Style: direct, quantified, specific to {company}. Prefer numbers over adjectives. Use bullets. Put the single most important insight in bold. Cite data sources and dates inline in brackets, e.g. {mk['cite_example']}."""

    # Search grounding is only exposed on the v1beta API, so pin it explicitly.
    client = genai.Client(
        api_key=_api_key,
        http_options=types.HttpOptions(api_version="v1beta"),
    )
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.3,
        ),
    )
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    sources = _grounding_sources(resp)
    if sources:
        text += "\n\n**Sources:** " + " · ".join(sources)
    return text


def fetch_genai_stock_analysis(ticker, company_name=None, dcf_context="", model=None, market="US"):
    """Returns (analysis_text | None, error_message | None)."""
    api_key = _get_api_key()
    if not api_key:
        return None, "GOOGLE_API_KEY not found in Streamlit secrets or environment"
    today = datetime.now(ZoneInfo(MARKET_PRESETS[market]["timezone"])).strftime("%d %b %Y")
    try:
        text = _gemini_brief(ticker, company_name or ticker, dcf_context, today,
                             model or _DEFAULTS["gemini_model"], market, api_key)
        return text, None
    except Exception as e:
        return None, f"GenAI analysis failed: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate_stock(ticker, config_in=None):
    cfg = {**_DEFAULTS, **(config_in or {})}
    if cfg["terminal_method"] not in ("perpetual_growth", "exit_multiple"):
        return None, "Invalid terminal method", None
    market = cfg.get("market", "US")
    if market not in MARKET_PRESETS:
        return None, f"Unknown market: {market}", None
    sym = MARKET_PRESETS[market]["currency_symbol"]

    ticker = normalize_ticker(ticker, cfg)
    if not ticker:
        return None, "Please enter a stock ticker", None

    try:
        d = fetch_market_data(ticker, market)
    except Exception as e:
        return None, f"Failed to fetch stock data: {e}", None

    missing = [n for n, v in [("price", d["price"]), ("revenue", d["revenue"]),
                              ("free cash flow", d["fcf"]), ("share count", d["shares"])]
               if not _ok(v)]
    if missing:
        return None, "Failed to fetch stock data: missing " + ", ".join(missing), None
    if d["revenue"] <= 0:
        return None, "Revenue is zero or negative", None
    if d["shares"] <= 0:
        return None, "Share count is invalid", None

    price, revenue, shares = d["price"], d["revenue"], d["shares"]
    fcf_reported, sbc = d["fcf"], d.get("sbc")

    warnings = []
    # ---- FCF, optionally after stock-based compensation --------------------
    fcf, sbc_adjusted = fcf_reported, False
    if cfg.get("adjust_for_sbc") and sbc and sbc > 0:
        fcf, sbc_adjusted = fcf_reported - sbc, True
        warnings.append(f"FCF is shown after deducting stock-based compensation of {_money(sbc, sym)} "
                        f"(reported FCF: {_money(fcf_reported, sym)}).")
    elif sbc and fcf_reported > 0 and sbc / fcf_reported > 0.15:
        warnings.append(f"Stock-based compensation is {sbc / fcf_reported:.0%} of reported FCF, so FCF may "
                        "overstate owner earnings. Consider enabling the SBC adjustment.")
    fcf_margin = fcf / revenue

    # ---- assumptions -------------------------------------------------------
    ke = cost_of_equity(d["beta"], cfg)
    hist_cagr = revenue_cagr(d["rev_history"])
    if cfg["growth_source"] == "historical" and hist_cagr is not None:
        base_growth = hist_cagr
    else:
        base_growth = cfg["default_analyst_growth_rate"]

    # ---- data-quality warnings --------------------------------------------
    if market == "US" and ticker.endswith((".NS", ".BO")):
        warnings.append("This looks like an Indian listing but the market is set to United States "
                        "(US risk-free rate and S&P 500 beta). Switch the market to India.")
    if market == "IN" and "." in ticker and not ticker.endswith((".NS", ".BO")):
        warnings.append("This does not look like an NSE/BSE ticker but the market is set to India.")
    if d["basis"] != "TTM":
        warnings.append("Four quarters of data were not available; used the latest fiscal year for revenue and FCF.")
    if not d["capex_found"]:
        warnings.append("Capital expenditure not found; FCF may be overstated.")
    if fcf_margin <= 0:
        warnings.append("Free cash flow is negative; a cash-flow DCF is not meaningful for this company.")
    if d["sector"] in ("Financial Services", "Real Estate"):
        warnings.append(f"Sector is {d['sector']}: FCF-based DCF is unreliable for banks, insurers, brokers "
                        "and REITs. Use P/B-ROE, residual income or FFO instead.")
    if d["currency"] and d["fin_currency"] and d["currency"] != d["fin_currency"]:
        warnings.append(f"Price currency ({d['currency']}) differs from financial statement currency "
                        f"({d['fin_currency']}), which is common for ADRs. Per-share values are unreliable.")
    if d.get("share_mismatch"):
        warnings.append("Yahoo's share count differs from market cap / price by more than 5% (often dual-class "
                        "shares). Using market cap / price.")
    if d["beta"] is None:
        warnings.append("Beta unavailable; using the default discount rate.")
    if hist_cagr is not None and abs(hist_cagr - base_growth) > 0.05:
        warnings.append(f"Growth assumption ({base_growth:.1%}) differs from historical revenue CAGR "
                        f"({hist_cagr:.1%}) by more than 5 points.")

    # ---- simulation --------------------------------------------------------
    paths = monte_carlo_simulation(revenue, fcf_margin, base_growth, ke,
                                   cfg["terminal_growth_rate"], shares, cfg)
    paths = paths[np.isfinite(paths)]
    if paths.size == 0:
        return None, "No valid simulations", None

    # deterministic base case (all inputs at their means)
    one = lambda x: np.array([float(x)])
    tg_base = min(cfg["terminal_growth_rate"], ke - cfg["min_spread"])
    base_case = float(dcf_per_share(revenue, one(fcf_margin), one(base_growth), one(ke),
                                    one(tg_base), one(cfg["exit_multiple"]), shares, cfg)[0])

    conf = cfg["confidence_level"]
    mean_price = float(np.mean(paths))
    median_price = float(np.median(paths))
    lo = float(np.percentile(paths, (1 - conf) / 2 * 100))
    hi = float(np.percentile(paths, (1 + conf) / 2 * 100))
    frac_nonpos = float(np.mean(paths <= 0))
    if frac_nonpos > 0.2 and fcf_margin > 0:
        warnings.append(f"{frac_nonpos:.0%} of simulations produced a non-positive value; results are unstable.")

    status_label = compare_prices(price, median_price)
    plot_path = _make_plot(paths, price, median_price, lo, hi, ticker, conf, sym)

    # ---- AI brief (given the quantitative result so the two can be reconciled) ----
    context = (f"price {sym}{price:,.2f}; median intrinsic value {sym}{median_price:,.2f} "
               f"({conf:.0%} range {sym}{lo:,.0f}-{sym}{hi:,.0f}); {status_label}; "
               f"probability value > price {np.mean(paths > price):.0%}; cost of equity {ke:.1%}; "
               f"base growth {base_growth:.1%}; FCF margin {fcf_margin:.1%}"
               + ("; FCF is after stock-based compensation" if sbc_adjusted else ""))
  #  genai_text, genai_error = fetch_genai_stock_analysis(
   #     ticker, d["name"], context, cfg["gemini_model"], market)

    output = {
        # keys the original front end reads
        "stock_price": price,
        "mean_simulated_price": mean_price,
        "valuation_status": status_label,
        "revenue": revenue,
        "free_cash_flow": fcf,
        "fcf_margin": fcf_margin,
       # "genai_analysis": genai_text,
        # additional keys
        "ticker": ticker,
        "market": market,
        "currency_symbol": sym,
        "company_name": d["name"],
        "sector": d["sector"],
        "median_simulated_price": median_price,
        "base_case_price": base_case,
        "prob_undervalued": float(np.mean(paths > price)),
        "margin_of_safety": (median_price - price) / median_price if median_price > 0 else None,
        "ci_lower": lo,
        "ci_upper": hi,
        "confidence_level": conf,
        "discount_rate": ke,
        "beta": d["beta"],
        "beta_source": d["beta_source"],
        "growth_used": base_growth,
        "hist_revenue_cagr": hist_cagr,
        "data_basis": d["basis"],
        "fcf_reported": fcf_reported,
        "sbc": sbc,
        "sbc_adjusted": sbc_adjusted,
        "fraction_nonpositive": frac_nonpos,
        "warnings": warnings,
        "genai_error": genai_error,
    }
    return output, "Success", plot_path


__all__ = ["evaluate_stock", "config", "MARKET_PRESETS"]
