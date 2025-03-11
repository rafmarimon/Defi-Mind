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
    

# -------------------------
# MAIN (SIDEBAR & ROUTING)
# -------------------------
def main():
    # (Optional) Could run concurrency or background tasks here
    # e.g. data = asyncio.run(async_fetch_example())

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
        "agent_chat": {
            "emoji": "🤖",
            "title": "Agent Chat",
            "desc": "Chat with the trading agent"
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
    elif choice == 'blockchain_insights':
        page_blockchain_insights()
    elif choice == 'market_history':
        page_market_history()
    elif choice == 'pyth_searcher':
        render_pyth_searcher_section()
    elif choice == 'agent_chat':
        page_agent_chat()

    st.write("---")
    st.markdown(f"""
    <div style="text-align: center; font-size: 0.9rem; color: #555; margin-top: 20px;">
        DEFIMIND Dashboard | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
