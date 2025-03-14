#!/usr/bin/env python3
"""
DEFIMIND Dashboard

A simple Streamlit dashboard for the DEFIMIND DeFi analytics platform.
"""

import os
import streamlit as st
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="DEFIMIND Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2196F3;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar
    st.sidebar.title("DEFIMIND")
    st.sidebar.markdown("---")
    
    # API Keys status
    api_keys = {
        "OpenAI API": os.environ.get("OPENAI_API_KEY", "") != "",
        "Etherscan API": os.environ.get("ETHERSCAN_API_KEY", "") != "",
        "Infura API": os.environ.get("INFURA_API_KEY", "") != "",
        "Alchemy API": os.environ.get("ALCHEMY_API_KEY", "") != "",
    }
    
    with st.sidebar.expander("API Keys Status"):
        for name, status in api_keys.items():
            st.write(f"{name}: {'✅' if status else '❌'}")
    
    st.sidebar.markdown("---")
    st.sidebar.info("DEFIMIND v0.1.0 - Running on Digital Ocean")
    
    # Main content
    st.markdown('<h1 class="main-header">DEFIMIND Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Real-time DeFi Insights</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total Value Locked", value="$45.2B", delta="-2.3%")
    
    with col2:
        st.metric(label="Avg. Yield", value="4.8%", delta="+0.3%")
    
    with col3:
        st.metric(label="Active Protocols", value="132", delta="0")
    
    with col4:
        st.metric(label="Blockchain Networks", value="14", delta="+1")
    
    # Sample data table
    st.subheader("Top DeFi Protocols")
    data = {
        "Protocol": ["Aave v3", "Compound v3", "Curve", "Lido", "Uniswap v3"],
        "Type": ["Lending", "Lending", "DEX", "Liquid Staking", "DEX"],
        "TVL ($B)": [6.5, 3.2, 1.8, 8.7, 7.2],
        "APY (%)": [3.8, 3.2, 4.5, 4.7, 5.1],
    }
    df = pd.DataFrame(data)
    st.dataframe(df)
    
    # System info
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("Running on Digital Ocean App Platform")

if __name__ == "__main__":
    main() 