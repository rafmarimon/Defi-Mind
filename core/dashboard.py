#!/usr/bin/env python3
"""
Enhanced DEFIMIND Dashboard

Incorporates industry best practices:
- Lighter, more readable theme
- Caching for performance
- Async data fetching (optional)
- Context-aware AI agent using Streamlit for chat interface
- Modular design: separate data fetching from UI
- Proactive alerts demonstration
"""

import asyncio
import time
import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import sys
sys.path.append('.')  # Ensure local imports work
try:
    from core.pyth_searcher import PythSearcher, Opportunity, run_pyth_searcher_demo
    PYTH_AVAILABLE = True
except ImportError:
    PYTH_AVAILABLE = False

# ---------------------------------------------
# OPTIONAL: If using an actual back-end (FastAPI)
# or a local DB, you'd import them here. For now,
# we keep it all in a single file for simplicity.
# ---------------------------------------------

# If you have an LLM or LangChain:
try:
    from core.langchain_agent import LangChainAgent  # Example; replace with actual
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Try to import our new components
try:
    from core.yield_tracker import YieldTracker
    from core.agent_communication import AgentCommunicator
    from core.strategies.yield_optimizer import YieldOptimizer
    from core.autonomous_agent import AutonomousAgent
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    print("Warning: Some DEFIMIND components are not available. Running in demo mode.")

# ------------- PAGE SETUP & LIGHTER THEME -------------
st.set_page_config(
    page_title="Enhanced DEFIMIND Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* GLOBALS */
body, .stApp {
    background-color: #fafafa !important;
    color: #333 !important;
}
[data-testid="stSidebar"] {
    background-color: #f5f5f5 !important;
    color: #333 !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #1f1f1f !important;
}
.stButton>button {
    background-color: #3a86ff !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    opacity: 0.9;
}
.stMetric, .stAlert, .stInfo, .stWarning, .stError {
    color: #333 !important;
}
.streamlit-expanderHeader {
    font-weight: 600 !important;
    color: #333 !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------------
# CACHING & DATA FETCH
# -------------------------

@st.cache_data(show_spinner=False)
def fetch_mock_market_data():
    """
    Mock function to simulate market data retrieval.
    In a real app, you might concurrently fetch from:
      - Alchemy (on-chain data)
      - DeFi Llama (TVL, protocol analytics)
      - CoinGecko (price data)
    Then combine or store in a local DB.
    """
    time.sleep(1)  # Simulate a slow API
    return {
        "market_sentiment": "Neutral",
        "total_tvl_usd": 18500000000,
        "average_utilization": 0.61,
        "protocols_analyzed": 3
    }

@st.cache_data(show_spinner=False)
def fetch_mock_gas_data(hours=24):
    """
    Mock function for historical gas prices.
    Replace with your actual logic to fetch from a DB or API.
    """
    now = datetime.now()
    times = [now - timedelta(hours=i) for i in range(hours)]
    prices = [20 + np.random.rand()*10 for _ in range(hours)]
    df = pd.DataFrame({"datetime": times, "gas_price": prices})
    return df.sort_values("datetime")

@st.cache_data(show_spinner=False)
def fetch_mock_protocol_performance():
    """Mock for protocol performance data (TVL)."""
    now = datetime.now()
    days = [now - timedelta(days=i) for i in range(14)]
    protocols = ["Aave", "Compound", "Uniswap"]
    data = []
    for d in days:
        for p in protocols:
            base = {"Aave": 8500000000, "Compound": 7000000000, "Uniswap": 5600000000}.get(p, 1e9)
            # simple random variation
            data.append({
                "protocol": p,
                "datetime": d,
                "tvl_usd": base * (1 + np.random.randn()*0.01)
            })
    df = pd.DataFrame(data)
    return df.sort_values("datetime")

@st.cache_data(show_spinner=False)
def fetch_mock_strategy_data():
    """Mock data for strategy allocations."""
    return [
        {"protocol": "Aave", "chain": "ethereum", "allocation_pct": 40, "status": "active"},
        {"protocol": "Uniswap", "chain": "ethereum", "allocation_pct": 30, "status": "active"},
        {"protocol": "Cash", "chain": "none", "allocation_pct": 30, "status": "active"},
    ]


# -------------------------
# OPTIONAL ASYNC EXAMPLE
# -------------------------
# If you want to demonstrate concurrency:
# (Note: Streamlit runs on an event loop that might conflict
# with user-defined async calls, so be mindful or run outside.)
async def async_fetch_example():
    # This is just a placeholder to show how you'd do concurrency
    await asyncio.sleep(1)
    return "Async data loaded"


# ------------------------------------------
#  AI AGENT LOGIC (MORE HUMAN-LIKE RESPONSES)
# ------------------------------------------
def get_ai_agent_response(user_msg, context=None):
    """
    Return a more human-like response from an AI agent.
    - context: optional dict with relevant data (market, gas, user profile)
    In a real system, you'd call:
       - LangChain / GPT / Claude with a carefully designed prompt
    """
    # For demonstration, we do a mock response that references the context:
    market_sentiment = context.get("market_sentiment", "Neutral")
    tvl = context.get("total_tvl_usd", 1.0)/1e9
    if LANGCHAIN_AVAILABLE:
        # e.g., use your LangChain agent
        # response = langchain_agent.process_message(user_msg, context)
        return f"(LangChain) The market sentiment is {market_sentiment}, total TVL: ${tvl:.2f}B.\n\nYour question: {user_msg}"
    else:
        # Return a mock "friendly" answer
        return (f"Hey there! I see you're curious about the market.\n"
                f"Right now, I'd say the overall sentiment is '{market_sentiment}' "
                f"with about ${tvl:.2f}B locked across protocols.\n\n"
                f"You asked: '{user_msg}'. I'd love to help more! "
                f"(This is a mock AI reply; integrate GPT/Claude for real data.)")


# -------------------------
# AGENT CHAT STATE
# -------------------------
def init_chat_state():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


# -------------------------
# PAGE CONTENT FUNCTIONS
# -------------------------
def page_overview():
    st.title("Overview")

    # Suppose we fetch market data
    data = fetch_mock_market_data()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Market Sentiment", data["market_sentiment"])
    col2.metric("Total TVL (B)", f"${data['total_tvl_usd']/1e9:.2f}B")
    col3.metric("Utilization", f"{data['average_utilization']*100:.1f}%")
    col4.metric("Protocols Analyzed", data["protocols_analyzed"])

    st.write("---")
    st.subheader("AI-Generated Market Analysis")
    with st.expander("View AI's Insights"):
        # We just generate a static-like text
        st.info("""**Key Observations**  
- Markets show moderate activity with stable TVL.  
- Gas prices remain feasible for normal transactions.  

**Lending vs. DEX**  
- Lending protocols appear slightly more profitable than DEX pools.  

**Risk Assessment**  
- Current market volatility is moderate.  

**Recommendations**  
- Balance your allocations between lending and stable assets.  
- Monitor gas for big moves or rebalances.
        """)


def page_protocol_analytics():
    st.title("Protocol Analytics")
    st.write("Analysis of DeFi protocols (example: TVL over time).")

    df = fetch_mock_protocol_performance()
    if df.empty:
        st.info("No protocol performance data.")
    else:
        fig = px.line(df, x="datetime", y="tvl_usd", color="protocol",
                      title="Total Value Locked by Protocol",
                      labels={"datetime": "Date", "tvl_usd": "TVL (USD)"})
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)


def page_trading_strategy():
    st.title("Trading Strategy")
    st.write("See current allocations and potential performance updates.")

    data = fetch_mock_strategy_data()
    df = pd.DataFrame(data)
    st.dataframe(df)

    fig = px.pie(df, values="allocation_pct", names="protocol",
                 title="Allocations (%)", hole=0.4)
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Example proactive alert: if any allocation > 50, highlight it
    large_allocs = df[df["allocation_pct"] > 50]
    if not large_allocs.empty:
        st.warning(f"High allocation detected: {large_allocs.to_dict('records')}")


def page_blockchain_insights():
    st.title("Blockchain Insights")
    st.write("Gas prices or on-chain metrics.")

    df = fetch_mock_gas_data(24)
    fig = px.line(df, x="datetime", y="gas_price", title="Gas Price (Gwei) - Last 24h",
                  labels={"datetime": "Time", "gas_price": "Gwei"})
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Example "Proactive Alert" if gas < 25 or > 80
    latest_gas = df.iloc[-1]["gas_price"]
    if latest_gas < 25:
        st.success(f"Gas is quite low: ~{latest_gas:.1f} Gwei. Good time to transact!")
    elif latest_gas > 80:
        st.error(f"Gas is extremely high: ~{latest_gas:.1f} Gwei! Perhaps wait.")


def page_market_history():
    st.title("Market History")
    st.write("Placeholder for more detailed historical data, e.g. 7-day or 30-day stats.")


def page_agent_chat():
    st.title("AI Agent Chat")
    st.write("Interact with a more human-like AI agent about market conditions or strategy ideas.")

    # Ensure chat state is initialized
    init_chat_state()

    # Show chat history
    for (role, text) in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Agent:** {text}")

    user_input = st.text_input("Ask the AI Agent:")
    if st.button("Send"):
        st.session_state.chat_history.append(("user", user_input))

        # Provide context from market data or other sources
        context_data = fetch_mock_market_data()
        agent_reply = get_ai_agent_response(user_input, context=context_data)
        st.session_state.chat_history.append(("agent", agent_reply))

        st.rerun()


# Add a new function to render the Pyth searcher section
def render_pyth_searcher_section():
    """Render the Pyth SVM searcher section"""
    st.header("Pyth SVM Searcher")
    
    # Note about Pyth integration
    st.markdown("""
    <div style="background-color: rgba(58, 134, 255, 0.15); border-radius: 5px; padding: 10px; 
    margin-bottom: 20px; font-size: 0.9rem; color: white !important;">
        🧪 <strong>Note:</strong> Pyth Express Relay integration for SVM limit order opportunities
    </div>
    """, unsafe_allow_html=True)
    
    if not PYTH_AVAILABLE:
        st.warning("Pyth searcher module is not available. Make sure pyth_searcher.py is in your project directory.")
        return
    
    # Create tabs for different views
    pyth_tabs = st.tabs(["Overview", "Recent Opportunities", "Submitted Bids", "Performance"])
    
    # Create a session state for the searcher if it doesn't exist
    if 'pyth_searcher' not in st.session_state:
        st.session_state.pyth_searcher = None
        st.session_state.pyth_running = False
        st.session_state.pyth_stats = {
            "opportunities_received": 0,
            "opportunities_evaluated": 0,
            "bids_submitted": 0,
            "bids_accepted": 0,
            "total_profit": 0.0,
            "start_time": datetime.now().isoformat(),
            "uptime_human": "0:00:00",
            "success_rate": 0,
        }
        st.session_state.pyth_opportunities = []
        st.session_state.pyth_bids = []
        st.session_state.pyth_successful_bids = []
    
    with pyth_tabs[0]:  # Overview tab
        st.markdown('<div style="background-color: rgba(30, 33, 40, 0.8); padding: 20px; border-radius: 10px; margin-bottom: 20px;">', unsafe_allow_html=True)
        st.subheader("Pyth SVM Searcher Overview")
        
        # Controls for the searcher
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not st.session_state.pyth_running:
                if st.button("Start Searcher (Simulation)"):
                    st.session_state.pyth_running = True
                    st.success("Searcher starting in simulation mode...")
                    
                    # In a real implementation, this would start the searcher
                    # For demonstration, we'll just update the stats periodically
                    
                    # Add some initial values
                    st.session_state.pyth_stats["start_time"] = datetime.now().isoformat()
            else:
                if st.button("Stop Searcher"):
                    st.session_state.pyth_running = False
                    st.info("Searcher stopped")
        
        with col2:
            chain_select = st.selectbox(
                "Target Chain",
                ["solana", "ethereum", "arbitrum", "optimism"],
                index=0
            )
        
        with col3:
            mode = st.selectbox(
                "Operation Mode",
                ["Simulation", "Production"],
                index=0,
                disabled=st.session_state.pyth_running
            )
            
        # Display current status
        st.markdown('<div style="background-color: rgba(58, 134, 255, 0.1); padding: 15px; border-radius: 10px; margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown("### Searcher Status")
        
        status_color = "green" if st.session_state.pyth_running else "red"
        status_text = "Running (Simulation)" if st.session_state.pyth_running else "Stopped"
        
        status_html = f"""
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {status_color}; margin-right: 10px;"></div>
            <div><strong>Status:</strong> {status_text}</div>
        </div>
        """
        st.markdown(status_html, unsafe_allow_html=True)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Opportunities",
                f"{st.session_state.pyth_stats['opportunities_received']}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Bids Submitted",
                f"{st.session_state.pyth_stats['bids_submitted']}",
                delta=None
            )
            
        with col3:
            st.metric(
                "Success Rate",
                f"{st.session_state.pyth_stats['success_rate']*100:.1f}%",
                delta=None
            )
            
        with col4:
            st.metric(
                "Est. Profit",
                f"${st.session_state.pyth_stats['total_profit']:.2f}",
                delta=None
            )
        
        st.markdown("### Configuration")
        st.code("""
# Pyth Express Relay Configuration
PYTH_EXPRESS_RELAY_URL = "https://pyth-express-relay-mainnet.asymmetric.re"
TARGET_CHAINS = ["solana"]
MIN_PROFIT_THRESHOLD = 0.5  # USD
WALLET_ADDRESS = "CENSORED"
        """, language="python")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with pyth_tabs[1]:  # Recent Opportunities tab
        st.markdown('<div style="background-color: rgba(30, 33, 40, 0.8); padding: 20px; border-radius: 10px; margin-bottom: 20px;">', unsafe_allow_html=True)
        st.subheader("Recent Opportunities")
        
        # Generate random opportunities for demo
        if st.session_state.pyth_running and len(st.session_state.pyth_opportunities) < 10:
            # Add a new opportunity every few seconds
            if len(st.session_state.pyth_opportunities) == 0 or random.random() < 0.3:
                new_opp = {
                    "order_address": f"So1ana{random.randint(10000, 99999)}Random{random.randint(100, 999)}Address",
                    "program": "limo",
                    "chain_id": chain_select,
                    "version": "v1",
                    "received_at": datetime.now().isoformat(),
                    "parsed_at": datetime.now().isoformat(),
                    "is_parsed": True,
                    "estimated_profit": round(random.uniform(0.2, 8.0), 2),
                    "status": random.choice(["evaluating", "submitted", "accepted", "rejected"])
                }
                st.session_state.pyth_opportunities.append(new_opp)
                st.session_state.pyth_stats["opportunities_received"] += 1
                
                # If profit above threshold, add to bids
                if new_opp["estimated_profit"] > 0.5 and new_opp["status"] in ["submitted", "accepted"]:
                    bid = {
                        "opportunity_address": new_opp["order_address"],
                        "chain_id": new_opp["chain_id"],
                        "estimated_profit": new_opp["estimated_profit"],
                        "timestamp": datetime.now().isoformat(),
                        "status": new_opp["status"]
                    }
                    st.session_state.pyth_bids.append(bid)
                    st.session_state.pyth_stats["bids_submitted"] += 1
                    
                    if new_opp["status"] == "accepted":
                        st.session_state.pyth_successful_bids.append(bid)
                        st.session_state.pyth_stats["bids_accepted"] += 1
                        st.session_state.pyth_stats["total_profit"] += new_opp["estimated_profit"]
                
        # Update success rate
        if st.session_state.pyth_stats["bids_submitted"] > 0:
            st.session_state.pyth_stats["success_rate"] = st.session_state.pyth_stats["bids_accepted"] / st.session_state.pyth_stats["bids_submitted"]
        
        # Display table of opportunities
        if st.session_state.pyth_opportunities:
            # Convert to dataframe for display
            opportunities_df = pd.DataFrame(st.session_state.pyth_opportunities)
            
            # Format dataframe
            display_df = opportunities_df[["order_address", "program", "chain_id", "estimated_profit", "status"]].rename(
                columns={
                    "order_address": "Order Address",
                    "program": "Program",
                    "chain_id": "Chain",
                    "estimated_profit": "Est. Profit (USD)",
                    "status": "Status"
                }
            )
            
            # Apply formatting
            display_df["Est. Profit (USD)"] = display_df["Est. Profit (USD)"].apply(lambda x: f"${x:.2f}")
            display_df["Status"] = display_df["Status"].apply(lambda x: f"⏳ {x}" if x == "evaluating" else 
                                                            f"🔄 {x}" if x == "submitted" else
                                                            f"✅ {x}" if x == "accepted" else
                                                            f"❌ {x}")
            
            st.table(display_df)
        else:
            st.info("No opportunities received yet. Start the searcher to begin receiving opportunities.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with pyth_tabs[2]:  # Submitted Bids tab
        st.markdown('<div style="background-color: rgba(30, 33, 40, 0.8); padding: 20px; border-radius: 10px; margin-bottom: 20px;">', unsafe_allow_html=True)
        st.subheader("Submitted Bids")
        
        if st.session_state.pyth_bids:
            # Convert to dataframe for display
            bids_df = pd.DataFrame(st.session_state.pyth_bids)
            
            # Format dataframe
            if not bids_df.empty:
                display_df = bids_df[["opportunity_address", "chain_id", "estimated_profit", "status"]].rename(
                    columns={
                        "opportunity_address": "Order Address",
                        "chain_id": "Chain",
                        "estimated_profit": "Est. Profit (USD)",
                        "status": "Status"
                    }
                )
                
                # Apply formatting
                display_df["Est. Profit (USD)"] = display_df["Est. Profit (USD)"].apply(lambda x: f"${x:.2f}")
                display_df["Status"] = display_df["Status"].apply(lambda x: f"🔄 {x}" if x == "submitted" else
                                                                f"✅ {x}" if x == "accepted" else
                                                                f"❌ {x}")
                
                st.table(display_df)
            else:
                st.info("No bids submitted yet.")
        else:
            st.info("No bids submitted yet.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with pyth_tabs[3]:  # Performance tab
        st.markdown('<div style="background-color: rgba(30, 33, 40, 0.8); padding: 20px; border-radius: 10px; margin-bottom: 20px;">', unsafe_allow_html=True)
        st.subheader("Performance Metrics")
        
        # Calculate uptime
        start_time = datetime.fromisoformat(st.session_state.pyth_stats["start_time"])
        uptime = datetime.now() - start_time
        st.session_state.pyth_stats["uptime_human"] = str(timedelta(seconds=int(uptime.total_seconds())))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Uptime", st.session_state.pyth_stats["uptime_human"])
            st.metric("Opportunities Received", st.session_state.pyth_stats["opportunities_received"])
            st.metric("Opportunities Evaluated", st.session_state.pyth_stats["opportunities_evaluated"] or st.session_state.pyth_stats["opportunities_received"])
        
        with col2:
            st.metric("Bids Submitted", st.session_state.pyth_stats["bids_submitted"])
            st.metric("Bids Accepted", st.session_state.pyth_stats["bids_accepted"])
            st.metric("Total Estimated Profit", f"${st.session_state.pyth_stats['total_profit']:.2f}")
        
        # Create profit chart
        if st.session_state.pyth_successful_bids:
            st.markdown("### Profit Over Time")
            
            # Generate time series for profit
            profit_data = []
            cumulative_profit = 0
            
            for bid in sorted(st.session_state.pyth_successful_bids, key=lambda x: x["timestamp"]):
                cumulative_profit += bid["estimated_profit"]
                profit_data.append({
                    "timestamp": datetime.fromisoformat(bid["timestamp"]),
                    "profit": bid["estimated_profit"],
                    "cumulative_profit": cumulative_profit
                })
            
            profit_df = pd.DataFrame(profit_data)
            
            # Create chart
            fig = px.line(
                profit_df,
                x="timestamp",
                y="cumulative_profit",
                title="Cumulative Profit from Accepted Bids",
                labels={
                    "timestamp": "Time",
                    "cumulative_profit": "Cumulative Profit (USD)"
                }
            )
            
            # Add scatter points for individual profits
            fig.add_scatter(
                x=profit_df["timestamp"],
                y=profit_df["profit"],
                mode="markers",
                name="Individual Profits",
                marker=dict(size=8)
            )
            
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(30, 33, 40, 0.8)",
                paper_bgcolor="rgba(30, 33, 40, 0.8)",
                font=dict(color="white"),
                title_font=dict(color="white"),
                legend_font=dict(color="white")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    

def render_yield_section():
    """Render the DeFi yield comparison section"""
    st.header("📈 DeFi Yield Opportunities")
    
    try:
        from core.yield_tracker import YieldTracker
        
        # Create yield tracker instance
        yield_tracker = YieldTracker()
        
        # Create tabs for different views
        yield_tabs = st.tabs(["Best Opportunities", "Token Comparison", "Historical Trends"])
        
        with yield_tabs[0]:
            st.subheader("Best Yield Opportunities")
            
            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                min_liquidity = st.slider(
                    "Minimum Liquidity (USD)", 
                    min_value=10000, 
                    max_value=10000000, 
                    value=100000,
                    step=10000,
                    format="$%d"
                )
            
            with col2:
                risk_options = {
                    "All": None,
                    "Low Risk Only": "low",
                    "Medium Risk or Lower": "medium",
                    "Any Risk Level": "high"
                }
                risk_filter = st.selectbox("Risk Filter", options=list(risk_options.keys()))
                
            # Fetch best yields
            best_yields = yield_tracker.get_best_yields_by_token(min_liquidity=min_liquidity)
            
            # Display as a styled table
            if not best_yields.empty:
                # Format APY as percentage
                best_yields['apy_display'] = best_yields['apy'].apply(lambda x: f"{x:.2f}%")
                
                # Format liquidity with commas
                best_yields['liquidity_display'] = best_yields['liquidity_usd'].apply(lambda x: f"${x:,.0f}")
                
                # Add risk column if available
                if 'risk_score' in best_yields.columns:
                    best_yields['risk_display'] = best_yields['risk_score'].apply(
                        lambda x: "Low" if x <= 1.3 else "Medium" if x <= 1.6 else "High"
                    )
                    
                    # Apply risk filter if selected
                    selected_risk = risk_options[risk_filter]
                    if selected_risk == "low":
                        best_yields = best_yields[best_yields['risk_score'] <= 1.3]
                    elif selected_risk == "medium":
                        best_yields = best_yields[best_yields['risk_score'] <= 1.6]
                
                # Display table
                if 'risk_display' in best_yields.columns:
                    display_df = best_yields[['token', 'platform', 'apy_display', 'liquidity_display', 'risk_display']]
                    column_config = {
                        "token": "Token",
                        "platform": "Platform",
                        "apy_display": "APY",
                        "liquidity_display": "Liquidity",
                        "risk_display": "Risk Level"
                    }
                else:
                    display_df = best_yields[['token', 'platform', 'apy_display', 'liquidity_display']]
                    column_config = {
                        "token": "Token",
                        "platform": "Platform",
                        "apy_display": "APY",
                        "liquidity_display": "Liquidity"
                    }
                
                st.dataframe(
                    display_df,
                    column_config=column_config,
                    hide_index=True
                )
            else:
                st.info("No yield data available matching the criteria")
        
        with yield_tabs[1]:
            st.subheader("Token Yield Comparison")
            
            # Token selection
            tokens = ["USDC", "ETH", "BTC", "DAI", "USDT"]
            selected_tokens = st.multiselect("Select Tokens", tokens, default=["USDC", "ETH"])
            
            if selected_tokens:
                # Get yields for selected tokens
                all_yields = yield_tracker.fetch_current_yields()
                
                # Prepare data for comparison
                comparison_data = []
                
                for platform, pools in all_yields.items():
                    for pool in pools:
                        if pool['token'] in selected_tokens:
                            comparison_data.append({
                                'token': pool['token'],
                                'platform': platform,
                                'apy': pool['apy'],
                                'liquidity': pool['liquidity_usd']
                            })
                
                if comparison_data:
                    # Convert to DataFrame
                    df = pd.DataFrame(comparison_data)
                    
                    # Create bar chart
                    fig = px.bar(
                        df, 
                        x='token', 
                        y='apy', 
                        color='platform',
                        barmode='group',
                        title="Yield Comparison by Token",
                        labels={'apy': 'APY (%)', 'token': 'Token'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    st.subheader("Detailed Comparison")
                    df['apy_display'] = df['apy'].apply(lambda x: f"{x:.2f}%")
                    df['liquidity_display'] = df['liquidity'].apply(lambda x: f"${x:,.0f}")
                    
                    st.dataframe(
                        df[['token', 'platform', 'apy_display', 'liquidity_display']],
                        column_config={
                            "token": "Token",
                            "platform": "Platform",
                            "apy_display": "APY",
                            "liquidity_display": "Liquidity"
                        },
                        hide_index=True
                    )
                else:
                    st.info("No data available for selected tokens")
            else:
                st.info("Please select at least one token for comparison")
        
        with yield_tabs[2]:
            st.subheader("Historical Yield Trends")
            
            # Token selection
            selected_token = st.selectbox("Select Token", tokens, index=0)
            
            # Date range
            days = st.slider("Days of History", min_value=7, max_value=90, value=30)
            
            # Get historical data
            hist_data = yield_tracker.get_historical_yield_trends(days=days, tokens=[selected_token])
            
            if not hist_data.empty:
                # Plot the data
                fig = px.line(
                    hist_data, 
                    x='date', 
                    y='apy', 
                    color='platform',
                    title=f"{selected_token} Yield Trends ({days} Days)",
                    labels={'apy': 'APY (%)', 'date': 'Date', 'platform': 'Platform'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show highest and lowest yields
                st.subheader("Yield Range")
                
                latest_data = hist_data[hist_data['date'] == hist_data['date'].max()]
                
                if not latest_data.empty:
                    max_idx = latest_data['apy'].idxmax()
                    min_idx = latest_data['apy'].idxmin()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Highest Current Yield", 
                            f"{latest_data.loc[max_idx, 'apy']:.2f}%",
                            f"on {latest_data.loc[max_idx, 'platform'].title()}"
                        )
                    
                    with col2:
                        st.metric(
                            "Lowest Current Yield", 
                            f"{latest_data.loc[min_idx, 'apy']:.2f}%",
                            f"on {latest_data.loc[min_idx, 'platform'].title()}"
                        )
            else:
                st.info(f"No historical yield data for {selected_token}")
    
    except ImportError:
        st.warning("Yield tracking module not available. Make sure yield_tracker.py is in your project directory.")
    except Exception as e:
        st.error(f"Error rendering yield section: {str(e)}")


def render_agent_communication_section():
    """Render the agent communication section"""
    st.header("🤖 Agent Communication")
    
    try:
        from core.agent_communication import AgentCommunicator
        from core.autonomous_agent import AutonomousAgent
        
        # Create communicator
        agent_comm = AgentCommunicator()
        
        # Create tabs
        comm_tabs = st.tabs(["Market Updates", "Agent Activity", "Ask DEFIMIND"])
        
        with comm_tabs[0]:
            st.subheader("Latest Market Analysis")
            
            # Get or generate market update
            if 'market_update' not in st.session_state or 'market_update_time' not in st.session_state:
                st.session_state.market_update = None
                st.session_state.market_update_time = None
            
            # Button to generate new update
            if st.button("Generate New Market Update"):
                with st.spinner("Analyzing market conditions..."):
                    update = agent_comm.generate_market_update()
                    st.session_state.market_update = update
                    st.session_state.market_update_time = datetime.now()
            
            # Display update if available
            if st.session_state.market_update:
                st.info(st.session_state.market_update)
                st.caption(f"Generated at: {st.session_state.market_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                # Initial placeholder
                st.info("Click 'Generate New Market Update' to get the latest DeFi market analysis.")
        
        with comm_tabs[1]:
            st.subheader("Recent Agent Activity")
            
            # Try to get the autonomous agent for real activity
            try:
                agent = AutonomousAgent()
                status = agent.get_status()
                
                # Show agent status
                st.write("**Agent Status:**")
                
                status_cols = st.columns(3)
                with status_cols[0]:
                    st.metric("Mode", "Simulation" if status.get('simulation_mode', True) else "Live")
                
                with status_cols[1]:
                    st.metric("Risk Tolerance", status.get('risk_tolerance', 'medium').title())
                
                with status_cols[2]:
                    actions_count = status.get('actions_taken_count', 0)
                    st.metric("Actions Taken", f"{actions_count}")
                
                # Show last action time if available
                if status.get('last_action_time'):
                    st.caption(f"Last action taken at: {status['last_action_time']}")
                
                # Generate activity report if agent has actions
                if actions_count > 0:
                    with st.spinner("Generating activity report..."):
                        activity_report = agent_comm.generate_activity_report()
                        st.write(activity_report)
                else:
                    # Display demo activity
                    st.info("The agent has not taken any actions yet. Here's a sample of what activity reports look like:")
                    
                    sample_actions = [
                        {
                            "type": "allocation",
                            "platform": "aave",
                            "token": "ETH",
                            "amount": 0.5,
                            "expected_apy": 1.8,
                            "timestamp": datetime.now().isoformat(),
                            "status": "simulated"
                        },
                        {
                            "type": "rebalance",
                            "from_platform": "compound",
                            "to_platform": "aave",
                            "token": "USDC",
                            "amount": 500,
                            "value_usd": 500,
                            "current_apy": 2.9,
                            "new_apy": 3.2,
                            "timestamp": datetime.now().isoformat(),
                            "status": "simulated",
                            "reason": "better yield (3.2% vs 2.9%)"
                        }
                    ]
                    
                    report = agent_comm.generate_activity_report(sample_actions)
                    st.write(report)
                    st.caption("This is a sample report. The agent will generate real reports as it takes actions.")
            
            except ImportError:
                st.warning("Autonomous agent module not available.")
                
                # Fallback to sample activity
                st.info("Sample Agent Activity Report:")
                sample_report = """
                ## DEFIMIND Activity Report

                In the past 24 hours, I've made the following portfolio adjustments:

                ### Actions Taken:
                1. Allocated 0.5 ETH to Aave (~$1,640) at 1.8% APY
                2. Rebalanced 500 USDC from Compound to Aave to take advantage of higher yield (3.2% vs 2.9%)

                ### Current Portfolio:
                - 2.5 ETH on Aave ($8,201.13) earning 1.8% APY
                - 2,249.1 USDC on Compound ($2,249.10) earning 3.0% APY
                - Total portfolio value: $10,450.23
                - Current weighted average yield: 2.8%

                ### Reasoning:
                The rebalancing of USDC was performed because Aave is currently offering a 0.3% higher yield than Compound. This difference exceeds our rebalance threshold of 0.2% and justifies the transaction costs.
                """
                st.write(sample_report)
                st.caption("This is a sample report for demonstration purposes.")
        
        with comm_tabs[2]:
            st.subheader("Ask DEFIMIND")
            
            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.write(f"**You:** {message['content']}")
                else:
                    st.write(f"**DEFIMIND:** {message['content']}")
            
            # Input for new question
            user_question = st.text_input("Your question:", key="agent_question")
            
            if user_question and st.button("Ask"):
                # Add user question to history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                # Get response
                with st.spinner("Thinking..."):
                    response = agent_comm.answer_user_question(user_question)
                    
                    # Add response to history
                    st.session_state.chat_history.append({"role": "agent", "content": response})
                
                # Force a rerun to show the updated chat
                st.rerun()
    
    except ImportError:
        st.warning("Agent communication module not available. Make sure agent_communication.py is in your project directory.")
    except Exception as e:
        st.error(f"Error rendering agent communication section: {str(e)}")


# Initialize session state for components
def init_defimind_components():
    if 'yield_tracker' not in st.session_state:
        if COMPONENTS_AVAILABLE:
            try:
                st.session_state.yield_tracker = YieldTracker()
            except:
                st.session_state.yield_tracker = None
        else:
            st.session_state.yield_tracker = None
            
    if 'agent_communicator' not in st.session_state:
        if COMPONENTS_AVAILABLE:
            try:
                st.session_state.agent_communicator = AgentCommunicator()
            except:
                st.session_state.agent_communicator = None
        else:
            st.session_state.agent_communicator = None
            
    if 'yield_optimizer' not in st.session_state:
        if COMPONENTS_AVAILABLE:
            try:
                st.session_state.yield_optimizer = YieldOptimizer()
            except:
                st.session_state.yield_optimizer = None
        else:
            st.session_state.yield_optimizer = None
            
    if 'autonomous_agent' not in st.session_state:
        if COMPONENTS_AVAILABLE:
            try:
                st.session_state.autonomous_agent = AutonomousAgent()
            except:
                st.session_state.autonomous_agent = None
        else:
            st.session_state.autonomous_agent = None


# -------------------------
# MAIN (SIDEBAR & ROUTING)
# -------------------------
def main():
    # (Optional) Could run concurrency or background tasks here
    # e.g. data = asyncio.run(async_fetch_example())
    
    # Initialize DEFIMIND components
    init_defimind_components()

    st.sidebar.title("Navigation")
    
    # Define navigation options with descriptions and emojis
    nav_options = {
        "overview": {
            "emoji": "📈",
            "title": "Overview",
            "desc": "Dashboard summary and key metrics"
        },
        "protocol_analytics": {
            "emoji": "🔬",
            "title": "Protocol Analytics",
            "desc": "Analysis of DeFi protocols"
        },
        "trading_strategy": {
            "emoji": "♟️",
            "title": "Trading Strategy",
            "desc": "Current allocations and performance"
        },
        "yield_comparison": {
            "emoji": "💰",
            "title": "Yield Comparison",
            "desc": "Compare yields across DeFi platforms"
        },
        "blockchain_insights": {
            "emoji": "📦",
            "title": "Blockchain Insights",
            "desc": "On-chain data and trends"
        },
        "market_history": {
            "emoji": "⏱️",
            "title": "Market History",
            "desc": "Historical market data"
        },
        "pyth_searcher": {
            "emoji": "🔍",
            "title": "Pyth Searcher",
            "desc": "SVM limit order opportunities"
        },
        "agent_communication": {
            "emoji": "🤖",
            "title": "Agent Communication",
            "desc": "Get updates and chat with the agent"
        },
    }

    choice = st.sidebar.radio("Go to:", list(nav_options.keys()))
    st.sidebar.write("---")
    if st.sidebar.button("Update Market Data"):
        with st.spinner("Refreshing data..."):
            # This triggers the cached functions to re-run next time they're called
            fetch_mock_market_data.clear()
            fetch_mock_gas_data.clear()
            fetch_mock_protocol_performance.clear()
            fetch_mock_strategy_data.clear()
            time.sleep(1)
        st.sidebar.success("Data has been refreshed (mock).")

    # Display selected content based on navigation
    if choice == 'overview':
        page_overview()
    elif choice == 'protocol_analytics':
        page_protocol_analytics()
    elif choice == 'trading_strategy':
        page_trading_strategy()
    elif choice == 'yield_comparison':
        render_yield_section()
    elif choice == 'blockchain_insights':
        page_blockchain_insights()
    elif choice == 'market_history':
        page_market_history()
    elif choice == 'pyth_searcher':
        render_pyth_searcher_section()
    elif choice == 'agent_communication':
        render_agent_communication_section()

    st.write("---")
    st.markdown(f"""
    <div style="text-align: center; font-size: 0.9rem; color: #555; margin-top: 20px;">
        DEFIMIND Dashboard | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
