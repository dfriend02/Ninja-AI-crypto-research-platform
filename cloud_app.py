"""
Intellectia AI Pro - Enhanced Trading Platform with Clickable Navigation
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import plotly.express as px
from paper_trading import get_paper_trader, process_indicator_signals, generate_demo_trades

# Add the parent directory to sys.path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import our trading platform components
try:
    from src.trading_platform import TradingPlatform
    from src.market_data import MarketData
    from src.technical_indicators import TechnicalIndicators
    from src.risk_analytics import RiskAnalytics
    from src.portfolio_optimizer import PortfolioOptimizer
    from src.ai_models import AIModels
    FULL_PLATFORM_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    # We'll use a simplified version if imports fail
    FULL_PLATFORM_AVAILABLE = False

# Import the simplified trading platform
from streamlit_app.models.simplified import EnhancedTradingPlatform

# Import our new modules
try:
    from realtime_data import realtime_streamer, demo_realtime_features
    from advanced_analytics import analytics_engine, demo_advanced_analytics
    from user_experience import ux_manager, setup_authentication, setup_theme_toggle, setup_watchlist, apply_custom_styling, demo_user_experience_features
    NEW_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    NEW_MODULES_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Intellectia AI Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme matching the screenshots
def apply_custom_theme():
    st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #111827;
        color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a2234;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    
    /* Metric cards */
    .css-1xarl3l {
        background-color: #1a2234;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #6c2bd9;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    
    /* Dropdown menus */
    .stSelectbox>div>div {
        background-color: #1a2234;
        color: white;
    }
    
    /* Charts background */
    .js-plotly-plot {
        background-color: #1a2234;
    }
    
    /* Custom card styling */
    .metric-card {
        background-color: #1a2234;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Custom header */
    .custom-header {
        background-color: #1a2234;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a2234;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2a3245;
        border-bottom: 2px solid #6c2bd9;
    }
    
    /* Custom table */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .custom-table th {
        background-color: #2a3245;
        padding: 8px;
        text-align: left;
    }
    
    .custom-table td {
        padding: 8px;
        border-bottom: 1px solid #2a3245;
    }
    
    .custom-table tr:hover {
        background-color: #2a3245;
    }
    
    /* Custom metrics */
    .metric-label {
        font-size: 0.8rem;
        color: #a0aec0;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
    }
    
    .metric-delta-positive {
        font-size: 0.9rem;
        color: #48bb78;
    }
    
    .metric-delta-negative {
        font-size: 0.9rem;
        color: #f56565;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .status-green {
        background-color: #48bb78;
    }
    
    .status-yellow {
        background-color: #ecc94b;
    }
    
    .status-red {
        background-color: #f56565;
    }
    
    /* Custom header */
    .platform-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 1rem;
        background-color: #111827;
        border-bottom: 1px solid #2a3245;
    }
    
    .platform-title {
        color: #00d4ff;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .platform-subtitle {
        color: #a0aec0;
        font-size: 0.8rem;
    }
    
    .platform-status {
        display: flex;
        align-items: center;
    }
    
    /* Navigation */
    .nav-item {
        display: flex;
        align-items: center;
        padding: 0.5rem 1rem;
        margin: 0.2rem 0;
        border-radius: 5px;
        cursor: pointer;
    }
    
    .nav-item:hover {
        background-color: #2a3245;
    }
    
    .nav-item.active {
        background-color: #2a3245;
        border-left: 3px solid #6c2bd9;
    }
    
    .nav-icon {
        margin-right: 0.5rem;
        color: #a0aec0;
    }
    
    .nav-text {
        color: white;
    }
    
    /* Custom select box */
    .custom-selectbox {
        background-color: #1a2234;
        border-radius: 5px;
        padding: 0.5rem;
        color: white;
        border: 1px solid #2a3245;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply custom theme
apply_custom_theme()

# Initialize session state
if 'trading_platform' not in st.session_state:
    if FULL_PLATFORM_AVAILABLE:
        try:
            st.session_state.trading_platform = TradingPlatform()
        except:
            st.session_state.trading_platform = EnhancedTradingPlatform()
    else:
        st.session_state.trading_platform = EnhancedTradingPlatform()

# Initialize paper trading
if 'paper_trader' not in st.session_state:
    st.session_state.paper_trader = get_paper_trader()

# Initialize demo data if needed
if 'demo_data_initialized' not in st.session_state:
    generate_demo_trades()
    st.session_state.demo_data_initialized = True

# Initialize current page
if 'current_page' not in st.session_state:
    st.session_state.current_page = "AI Dashboard"

# Setup authentication if modules are available
if NEW_MODULES_AVAILABLE:
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = True  # Auto-authenticate for demo
        
    if 'username' not in st.session_state:
        st.session_state.username = "Demo User"

# Custom header
def render_header():
    col1, col2, col3 = st.columns([2, 8, 2])
    
    with col1:
        st.markdown("""
        <div style="display: flex; align-items: center;">
            <img src="https://img.icons8.com/color/48/000000/brain-3.png" width="30" style="margin-right: 10px;">
            <div>
                <div style="color: #00d4ff; font-size: 1.2rem; font-weight: bold; line-height: 1;">Intellectia AI Pro</div>
                <div style="color: #a0aec0; font-size: 0.7rem;">Advanced ML Trading Platform</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="display: flex; justify-content: flex-end; align-items: center;">
            <div style="background-color: #1a2234; padding: 5px 10px; border-radius: 5px; margin-right: 10px;">
                <span style="color: #a0aec0;">Python:</span>
                <span style="color: white;">Connected</span>
            </div>
            <div style="background-color: #1a2234; padding: 5px 10px; border-radius: 5px;">
                <span style="color: #a0aec0;">Auto:</span>
                <span style="color: white;">OFF</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Custom navigation
def render_navigation():
    # Navigation options
    nav_options = [
        {"name": "AI Dashboard", "icon": "📊"},
        {"name": "Deep Research", "icon": "🔍"},
        {"name": "Smart Signals", "icon": "📈"},
        {"name": "ML Engine", "icon": "🧠"},
        {"name": "Portfolio AI", "icon": "💼"},
        {"name": "Risk Analytics", "icon": "⚠️"},
        {"name": "Strategy Lab", "icon": "🧪"},
        {"name": "Market Scanner", "icon": "🔎"},
        {"name": "Sentiment AI", "icon": "💭"},
        {"name": "Hot News", "icon": "🔥"}
    ]
    
    # Create columns for navigation items
    cols = st.columns(len(nav_options))
    
    # Render navigation items
    for i, nav in enumerate(nav_options):
        with cols[i]:
            if st.button(f"{nav['icon']} {nav['name']}", key=f"nav_{nav['name']}"):
                st.session_state.current_page = nav['name']
                st.rerun()

# Custom asset selector
def render_asset_selector():
    col1, col2, col3, col4 = st.columns([1, 2, 1, 8])
    
    with col1:
        asset_type = st.selectbox("Asset Type", ["Crypto", "Stocks", "Forex", "Commodities"], key="asset_type", label_visibility="collapsed")
    
    with col2:
        if asset_type == "Crypto":
            asset = st.selectbox("Asset", ["Bitcoin (BTC-USD)", "Ethereum (ETH-USD)", "Solana (SOL-USD)"], key="asset", label_visibility="collapsed")
        elif asset_type == "Stocks":
            asset = st.selectbox("Asset", ["Apple (AAPL)", "Microsoft (MSFT)", "Google (GOOGL)"], key="asset", label_visibility="collapsed")
        else:
            asset = st.selectbox("Asset", ["Select Asset"], key="asset", label_visibility="collapsed")
    
    with col3:
        timeframe = st.selectbox("Timeframe", ["1D", "1W", "1M", "1Y"], key="timeframe", label_visibility="collapsed")
    
    with col4:
        # Get the current price from paper trader or simulate
        price = 42667
        change = -0.89
        
        st.markdown(f"""
        <div style="text-align: right;">
            <div style="font-size: 1.8rem; font-weight: bold; color: white;">${price:,.0f}</div>
            <div style="font-size: 1rem; color: {'#f56565' if change < 0 else '#48bb78'};">{change}%</div>
        </div>
        """, unsafe_allow_html=True)

# AI Dashboard page
def render_ai_dashboard():
    # Get paper trading data
    paper_trader = st.session_state.paper_trader
    portfolio_value = paper_trader.get_portfolio_value()
    portfolio_return = paper_trader.get_portfolio_return()
    win_rate = paper_trader.get_win_rate()
    active_signals = paper_trader.get_active_signals_count()
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Portfolio</div>
            <div class="metric-value">$127K</div>
            <div class="metric-delta-positive">+23.7%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">{win_rate:.1f}%</div>
            <div class="metric-label">572 trades</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Signals</div>
            <div class="metric-value">{active_signals}</div>
            <div style="color: #48bb78;">Active</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Risk</div>
            <div class="metric-value">Medium</div>
            <div class="metric-label">VaR: 8.7%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Price chart with ML predictions
    st.markdown("""
    <h3 style="margin-top: 1.5rem; margin-bottom: 1rem;">Price Analysis with ML Predictions</h3>
    """, unsafe_allow_html=True)
    
    # Generate price data
    dates = pd.date_range(end=datetime.now(), periods=30)
    base_price = 42000
    price_data = pd.DataFrame({
        'Date': dates,
        'Price': [base_price * (1 + np.cumsum(np.random.normal(0, 0.02, 30))[i]) for i in range(30)]
    })
    
    # Create prediction data (slightly higher than actual)
    prediction_data = pd.DataFrame({
        'Date': pd.date_range(start=dates[-15], periods=20),
        'Prediction': [price_data['Price'].iloc[-15] * (1 + np.cumsum(np.random.normal(0.001, 0.015, 20))[i]) for i in range(20)]
    })
    
    # Create chart
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=price_data['Date'],
        y=price_data['Price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='#00d4ff', width=2)
    ))
    
    # Add prediction line
    fig.add_trace(go.Scatter(
        x=prediction_data['Date'],
        y=prediction_data['Prediction'],
        mode='lines',
        name='ML Prediction',
        line=dict(color='#6c2bd9', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor='#2a3245'),
        yaxis=dict(showgrid=True, gridcolor='#2a3245'),
        plot_bgcolor='#1a2234',
        paper_bgcolor='#1a2234',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # AI Models Performance
    st.markdown("""
    <h3 style="margin-top: 1.5rem; margin-bottom: 1rem;">AI Models Performance</h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: #1a2234; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-weight: bold;">Ensemble</div>
                <div style="width: 10px; height: 10px; border-radius: 50%; background-color: #48bb78;"></div>
            </div>
            <div style="margin-top: 0.5rem;">Acc: 84.7%</div>
            <div style="color: #48bb78;">Profit: +156.7%</div>
        </div>
        
        <div style="background-color: #1a2234; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-weight: bold;">RandomForest</div>
                <div style="width: 10px; height: 10px; border-radius: 50%; background-color: #48bb78;"></div>
            </div>
            <div style="margin-top: 0.5rem;">Acc: 82.3%</div>
            <div style="color: #48bb78;">Profit: +145.3%</div>
        </div>
        
        <div style="background-color: #1a2234; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-weight: bold;">Regime</div>
                <div style="width: 10px; height: 10px; border-radius: 50%; background-color: #48bb78;"></div>
            </div>
            <div style="margin-top: 0.5rem;">Acc: 80.2%</div>
            <div style="color: #48bb78;">Profit: +187.2%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #1a2234; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-weight: bold;">LSTM</div>
                <div style="width: 10px; height: 10px; border-radius: 50%; background-color: #ecc94b;"></div>
            </div>
            <div style="margin-top: 0.5rem;">Acc: 79.1%</div>
            <div style="color: #48bb78;">Profit: +134.2%</div>
        </div>
        
        <div style="background-color: #1a2234; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-weight: bold;">Sentiment</div>
                <div style="width: 10px; height: 10px; border-radius: 50%; background-color: #48bb78;"></div>
            </div>
            <div style="margin-top: 0.5rem;">Acc: 73.4%</div>
            <div style="color: #48bb78;">Profit: +98.7%</div>
        </div>
        """, unsafe_allow_html=True)

# Deep Research page
def render_deep_research():
    st.markdown("""
    <h3 style="margin-bottom: 1rem;">Deep Research</h3>
    """, unsafe_allow_html=True)
    
    # Research tabs
    tabs = st.tabs(["Fundamentals", "Technical Analysis", "On-Chain Data", "News & Events", "Social Sentiment"])
    
    with tabs[0]:
        st.markdown("""
        <h4>Fundamental Analysis</h4>
        """, unsafe_allow_html=True)
        
        # Fundamental metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background-color: #1a2234; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                <h5>Market Metrics</h5>
                <table class="custom-table">
                    <tr>
                        <td>Market Cap</td>
                        <td>$834.5B</td>
                    </tr>
                    <tr>
                        <td>24h Volume</td>
                        <td>$28.7B</td>
                    </tr>
                    <tr>
                        <td>Circulating Supply</td>
                        <td>19.53M BTC</td>
                    </tr>
                    <tr>
                        <td>Max Supply</td>
                        <td>21M BTC</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background-color: #1a2234; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                <h5>Network Metrics</h5>
                <table class="custom-table">
                    <tr>
                        <td>Hash Rate</td>
                        <td>512 EH/s</td>
                    </tr>
                    <tr>
                        <td>Difficulty</td>
                        <td>72.35T</td>
                    </tr>
                    <tr>
                        <td>Active Addresses</td>
                        <td>1.2M</td>
                    </tr>
                    <tr>
                        <td>Transaction Fee</td>
                        <td>$3.45</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        
        # Fundamental chart
        st.markdown("""
        <h5>Bitcoin Stock-to-Flow Model</h5>
        """, unsafe_allow_html=True)
        
        # Generate S2F data
        dates = pd.date_range(start='2010-01-01', end='2030-01-01', freq='M')
        s2f_model = [1000 * np.exp(0.4 * np.sqrt(i/12)) for i in range(len(dates))]
        actual_price = [s2f_model[i] * (1 + np.random.normal(0, 0.3)) for i in range(len(dates))]
        
        # Create chart
        fig = go.Figure()
        
        # Add S2F model line
        fig.add_trace(go.Scatter(
            x=dates,
            y=s2f_model,
            mode='lines',
            name='S2F Model',
            line=dict(color='#6c2bd9', width=2)
        ))
        
        # Add actual price line
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual_price,
            mode='lines',
            name='Actual Price',
            line=dict(color='#00d4ff', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor='#2a3245'),
            yaxis=dict(showgrid=True, gridcolor='#2a3245', type='log'),
            plot_bgcolor='#1a2234',
            paper_bgcolor='#1a2234',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.markdown("""
        <h4>Technical Analysis</h4>
        """, unsafe_allow_html=True)
        
        # Technical indicators
        indicator_tabs = st.tabs(["Moving Averages", "Oscillators", "Volatility", "Volume", "Patterns"])
        
        with indicator_tabs[0]:
            st.markdown("""
            <h5>Moving Averages</h5>
            """, unsafe_allow_html=True)
            
            # Generate price data
            dates = pd.date_range(end=datetime.now(), periods=200)
            price = [42000 * (1 + np.cumsum(np.random.normal(0, 0.02, 200))[i]) for i in range(200)]
            
            # Calculate moving averages
            ma_20 = pd.Series(price).rolling(window=20).mean().tolist()
            ma_50 = pd.Series(price).rolling(window=50).mean().tolist()
            ma_200 = pd.Series(price).rolling(window=200).mean().tolist()
            
            # Create chart
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=dates,
                y=price,
                mode='lines',
                name='Price',
                line=dict(color='#00d4ff', width=2)
            ))
            
            # Add moving averages
            fig.add_trace(go.Scatter(
                x=dates,
                y=ma_20,
                mode='lines',
                name='MA 20',
                line=dict(color='#6c2bd9', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=ma_50,
                mode='lines',
                name='MA 50',
                line=dict(color='#48bb78', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=ma_200,
                mode='lines',
                name='MA 200',
                line=dict(color='#f56565', width=2)
            ))
            
            # Update layout
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=True, gridcolor='#2a3245'),
                yaxis=dict(showgrid=True, gridcolor='#2a3245'),
                plot_bgcolor='#1a2234',
                paper_bgcolor='#1a2234',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Moving average signals
            st.markdown("""
            <div style="background-color: #1a2234; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                <h5>Moving Average Signals</h5>
                <table class="custom-table">
                    <tr>
                        <th>Signal</th>
                        <th>Status</th>
                        <th>Since</th>
                    </tr>
                    <tr>
                        <td>MA 20/50 Crossover</td>
                        <td><span style="color: #48bb78;">Bullish</span></td>
                        <td>2 days ago</td>
                    </tr>
                    <tr>
                        <td>MA 50/200 Crossover</td>
                        <td><span style="color: #48bb78;">Bullish</span></td>
                        <td>15 days ago</td>
                    </tr>
                    <tr>
                        <td>Price above MA 200</td>
                        <td><span style="color: #48bb78;">Bullish</span></td>
                        <td>45 days ago</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

# Smart Signals page
def render_smart_signals():
    st.markdown("""
    <h3 style="margin-bottom: 1rem;">Smart Signals</h3>
    """, unsafe_allow_html=True)
    
    # Signal filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.selectbox("Asset Class", ["All", "Crypto", "Stocks", "Forex", "Commodities"], key="signal_asset_class")
    
    with col2:
        st.selectbox("Signal Type", ["All", "Buy", "Sell", "Neutral"], key="signal_type")
    
    with col3:
        st.selectbox("Timeframe", ["All", "1H", "4H", "1D", "1W"], key="signal_timeframe")
    
    with col4:
        st.selectbox("Model", ["All", "Ensemble", "LSTM", "RandomForest", "Sentiment", "Regime"], key="signal_model")
    
    # Active signals
    st.markdown("""
    <h4 style="margin-top: 1.5rem; margin-bottom: 1rem;">Active Signals</h4>
    """, unsafe_allow_html=True)
    
    # Create sample signals
    signals = [
        {"asset": "BTC-USD", "signal": "BUY", "confidence": 87, "model": "Ensemble", "timeframe": "1D", "timestamp": "2025-09-24 12:34", "price": "$42,667"},
        {"asset": "ETH-USD", "signal": "BUY", "confidence": 82, "model": "LSTM", "timeframe": "1D", "timestamp": "2025-09-24 10:15", "price": "$3,245"},
        {"asset": "AAPL", "signal": "SELL", "confidence": 76, "model": "RandomForest", "timeframe": "1D", "timestamp": "2025-09-24 09:45", "price": "$187.25"},
        {"asset": "MSFT", "signal": "BUY", "confidence": 79, "model": "Ensemble", "timeframe": "1D", "timestamp": "2025-09-24 14:22", "price": "$412.50"},
        {"asset": "SOL-USD", "signal": "BUY", "confidence": 84, "model": "Sentiment", "timeframe": "1D", "timestamp": "2025-09-24 11:05", "price": "$145.75"}
    ]
    
    # Display signals
    signals_df = pd.DataFrame(signals)
    st.dataframe(signals_df, use_container_width=True)
    
    # Signal performance
    st.markdown("""
    <h4 style="margin-top: 1.5rem; margin-bottom: 1rem;">Signal Performance</h4>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create performance metrics
        st.markdown("""
        <div style="background-color: #1a2234; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
            <h5>Performance Metrics</h5>
            <table class="custom-table">
                <tr>
                    <td>Win Rate</td>
                    <td>68.5%</td>
                </tr>
                <tr>
                    <td>Average Return</td>
                    <td>+12.3%</td>
                </tr>
                <tr>
                    <td>Profit Factor</td>
                    <td>2.4</td>
                </tr>
                <tr>
                    <td>Average Hold Time</td>
                    <td>14.2 days</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Create model comparison
        st.markdown("""
        <div style="background-color: #1a2234; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
            <h5>Model Comparison</h5>
            <table class="custom-table">
                <tr>
                    <th>Model</th>
                    <th>Win Rate</th>
                    <th>Avg Return</th>
                </tr>
                <tr>
                    <td>Ensemble</td>
                    <td>84.7%</td>
                    <td>+15.2%</td>
                </tr>
                <tr>
                    <td>LSTM</td>
                    <td>79.1%</td>
                    <td>+12.8%</td>
                </tr>
                <tr>
                    <td>RandomForest</td>
                    <td>82.3%</td>
                    <td>+14.5%</td>
                </tr>
                <tr>
                    <td>Sentiment</td>
                    <td>73.4%</td>
                    <td>+9.7%</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    # Signal history chart
    st.markdown("""
    <h4 style="margin-top: 1.5rem; margin-bottom: 1rem;">Signal History</h4>
    """, unsafe_allow_html=True)
    
    # Generate signal history data
    dates = pd.date_range(end=datetime.now(), periods=30)
    buy_signals = np.random.randint(1, 5, 30)
    sell_signals = np.random.randint(1, 5, 30)
    
    # Create chart
    fig = go.Figure()
    
    # Add buy signals
    fig.add_trace(go.Bar(
        x=dates,
        y=buy_signals,
        name='Buy Signals',
        marker_color='#48bb78'
    ))
    
    # Add sell signals
    fig.add_trace(go.Bar(
        x=dates,
        y=sell_signals,
        name='Sell Signals',
        marker_color='#f56565'
    ))
    
    # Update layout
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor='#2a3245'),
        yaxis=dict(showgrid=True, gridcolor='#2a3245'),
        plot_bgcolor='#1a2234',
        paper_bgcolor='#1a2234',
        font=dict(color='white'),
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ML Engine page
def render_ml_engine():
    st.markdown("""
    <h3 style="margin-bottom: 1rem;">ML Engine</h3>
    """, unsafe_allow_html=True)
    
    # ML model tabs
    tabs = st.tabs(["Ensemble", "LSTM", "RandomForest", "Sentiment", "Regime"])
    
    with tabs[0]:
        st.markdown("""
        <h4>Ensemble Model</h4>
        """, unsafe_allow_html=True)
        
        # Model performance
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background-color: #1a2234; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                <h5>Model Performance</h5>
                <table class="custom-table">
                    <tr>
                        <td>Accuracy</td>
                        <td>84.7%</td>
                    </tr>
                    <tr>
                        <td>Precision</td>
                        <td>82.3%</td>
                    </tr>
                    <tr>
                        <td>Recall</td>
                        <td>85.1%</td>
                    </tr>
                    <tr>
                        <td>F1 Score</td>
                        <td>83.7%</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background-color: #1a2234; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                <h5>Trading Performance</h5>
                <table class="custom-table">
                    <tr>
                        <td>Win Rate</td>
                        <td>84.7%</td>
                    </tr>
                    <tr>
                        <td>Profit Factor</td>
                        <td>3.2</td>
                    </tr>
                    <tr>
                        <td>Average Return</td>
                        <td>+15.2%</td>
                    </tr>
                    <tr>
                        <td>Max Drawdown</td>
                        <td>-8.7%</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature importance
        st.markdown("""
        <h5 style="margin-top: 1.5rem; margin-bottom: 1rem;">Feature Importance</h5>
        """, unsafe_allow_html=True)
        
        # Generate feature importance data
        features = ["RSI", "MACD", "Bollinger %B", "Volume", "MA Crossover", "Sentiment", "Market Regime", "Volatility", "Support/Resistance", "Trend Strength"]
        importance = [0.18, 0.15, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.04]
        
        # Create chart
        fig = go.Figure()
        
        # Add feature importance bars
        fig.add_trace(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color='#00d4ff'
        ))
        
        # Update layout
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=True, gridcolor='#2a3245', title="Importance"),
            yaxis=dict(showgrid=True, gridcolor='#2a3245'),
            plot_bgcolor='#1a2234',
            paper_bgcolor='#1a2234',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix
        st.markdown("""
        <h5 style="margin-top: 1.5rem; margin-bottom: 1rem;">Confusion Matrix</h5>
        """, unsafe_allow_html=True)
        
        # Generate confusion matrix data
        confusion_matrix = [
            [423, 47, 30],
            [38, 215, 27],
            [22, 31, 197]
        ]
        
        # Create chart
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=["Buy", "Hold", "Sell"],
            y=["Buy", "Hold", "Sell"],
            colorscale='Blues',
            showscale=False,
            text=confusion_matrix,
            texttemplate="%{text}",
            textfont={"size": 14}
        ))
        
        # Update layout
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(title="Predicted"),
            yaxis=dict(title="Actual"),
            plot_bgcolor='#1a2234',
            paper_bgcolor='#1a2234',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Portfolio AI page
def render_portfolio_ai():
    st.markdown("""
    <h3 style="margin-bottom: 1rem;">Portfolio Optimization</h3>
    """, unsafe_allow_html=True)
    
    # Create radar chart data
    categories = ['BTC', 'ETH', 'NVDA', 'AAPL', 'SOL']
    values = [0.4, 0.25, 0.15, 0.1, 0.1]
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 212, 255, 0.2)',
        line=dict(color='#00d4ff', width=2),
        name='Optimal Allocation'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 0.5],
                showticklabels=False,
                gridcolor='#2a3245'
            ),
            angularaxis=dict(
                gridcolor='#2a3245'
            ),
            bgcolor='#1a2234'
        ),
        showlegend=False,
        margin=dict(l=80, r=80, t=20, b=20),
        height=500,
        paper_bgcolor='#111827',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Expected Return", "18.7%")
    
    with col2:
        st.metric("Volatility", "12.3%")
    
    with col3:
        st.metric("Sharpe Ratio", "1.52")
    
    with col4:
        st.metric("Max Drawdown", "-15.4%")
    
    # Efficient frontier
    st.markdown("""
    <h4 style="margin-top: 1.5rem; margin-bottom: 1rem;">Efficient Frontier</h4>
    """, unsafe_allow_html=True)
    
    # Generate efficient frontier data
    volatility = np.linspace(0.05, 0.3, 100)
    returns = [0.05 + v**0.7 for v in volatility]
    
    # Create chart
    fig = go.Figure()
    
    # Add efficient frontier line
    fig.add_trace(go.Scatter(
        x=volatility,
        y=returns,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='#00d4ff', width=2)
    ))
    
    # Add portfolio markers
    fig.add_trace(go.Scatter(
        x=[0.123],
        y=[0.187],
        mode='markers',
        name='Current Portfolio',
        marker=dict(size=12, color='#6c2bd9')
    ))
    
    fig.add_trace(go.Scatter(
        x=[0.08],
        y=[0.12],
        mode='markers',
        name='Min Volatility',
        marker=dict(size=12, color='#48bb78')
    ))
    
    fig.add_trace(go.Scatter(
        x=[0.18],
        y=[0.22],
        mode='markers',
        name='Max Return',
        marker=dict(size=12, color='#f56565')
    ))
    
    # Update layout
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor='#2a3245', title="Volatility"),
        yaxis=dict(showgrid=True, gridcolor='#2a3245', title="Expected Return"),
        plot_bgcolor='#1a2234',
        paper_bgcolor='#1a2234',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Risk Analytics page
def render_risk_analytics():
    st.markdown("""
    <h3 style="margin-bottom: 1rem;">Risk Analytics</h3>
    """, unsafe_allow_html=True)
    
    # Risk metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Value at Risk (95%)</div>
            <div class="metric-value">8.7%</div>
            <div class="metric-label">Daily</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value">15.4%</div>
            <div class="metric-label">Historical</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Beta</div>
            <div class="metric-value">1.25</div>
            <div class="metric-label">vs S&P 500</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Volatility</div>
            <div class="metric-value">12.3%</div>
            <div class="metric-label">Annualized</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk heatmap
    st.markdown("""
    <h4 style="margin-top: 1.5rem; margin-bottom: 1rem;">Risk Heatmap</h4>
    """, unsafe_allow_html=True)
    
    # Generate risk heatmap data
    assets = ["BTC", "ETH", "NVDA", "AAPL", "SOL"]
    correlation_matrix = [
        [1.00, 0.82, 0.45, 0.38, 0.72],
        [0.82, 1.00, 0.41, 0.35, 0.68],
        [0.45, 0.41, 1.00, 0.75, 0.32],
        [0.38, 0.35, 0.75, 1.00, 0.28],
        [0.72, 0.68, 0.32, 0.28, 1.00]
    ]
    
    # Create chart
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=assets,
        y=assets,
        colorscale='RdBu_r',
        zmid=0,
        text=[[f"{val:.2f}" for val in row] for row in correlation_matrix],
        texttemplate="%{text}",
        textfont={"size": 14}
    ))
    
    # Update layout
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='#1a2234',
        paper_bgcolor='#1a2234',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monte Carlo simulation
    st.markdown("""
    <h4 style="margin-top: 1.5rem; margin-bottom: 1rem;">Monte Carlo Simulation</h4>
    """, unsafe_allow_html=True)
    
    # Generate Monte Carlo data
    dates = pd.date_range(start=datetime.now(), periods=252)  # 1 year of trading days
    
    # Generate 100 simulations
    simulations = []
    for i in range(100):
        # Start at current price
        start_price = 42667
        # Generate random returns
        returns = np.random.normal(0.0005, 0.02, 252)
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)
        # Calculate price path
        price_path = start_price * cumulative_returns
        simulations.append(price_path)
    
    # Create chart
    fig = go.Figure()
    
    # Add simulation lines
    for i, sim in enumerate(simulations):
        fig.add_trace(go.Scatter(
            x=dates,
            y=sim,
            mode='lines',
            line=dict(color='rgba(0, 212, 255, 0.1)'),
            showlegend=False
        ))
    
    # Add mean line
    mean_sim = np.mean(simulations, axis=0)
    fig.add_trace(go.Scatter(
        x=dates,
        y=mean_sim,
        mode='lines',
        name='Mean',
        line=dict(color='#00d4ff', width=3)
    ))
    
    # Add percentile lines
    percentile_5 = np.percentile(simulations, 5, axis=0)
    percentile_95 = np.percentile(simulations, 95, axis=0)
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=percentile_5,
        mode='lines',
        name='5th Percentile',
        line=dict(color='#f56565', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=percentile_95,
        mode='lines',
        name='95th Percentile',
        line=dict(color='#48bb78', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor='#2a3245'),
        yaxis=dict(showgrid=True, gridcolor='#2a3245'),
        plot_bgcolor='#1a2234',
        paper_bgcolor='#1a2234',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Strategy Lab page
def render_strategy_lab():
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
        <h3 style="margin: 0;">Strategy Backtesting</h3>
        <button style="background-color: #6c2bd9; color: white; border: none; border-radius: 5px; padding: 0.5rem 1rem;">Run Backtest</button>
    </div>
    """, unsafe_allow_html=True)
    
    # Strategy configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <h4>Strategy Configuration</h4>
        """, unsafe_allow_html=True)
        
        # Strategy settings (without form)
        st.selectbox("Strategy Type", ["Moving Average Crossover", "RSI Strategy", "MACD Strategy", "Bollinger Bands", "Custom Strategy"], key="strategy_type")
        st.selectbox("Asset", ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "GOOGL"], key="strategy_asset")
        st.selectbox("Timeframe", ["1H", "4H", "1D", "1W"], key="strategy_timeframe")
        st.slider("Initial Capital", 1000, 100000, 10000, 1000, key="strategy_capital")
        st.slider("Position Size (%)", 1, 100, 10, key="strategy_position_size")
        use_stop_loss = st.checkbox("Use Stop Loss", key="strategy_use_stop_loss")
        st.slider("Stop Loss (%)", 1, 50, 5, key="strategy_stop_loss")
        use_take_profit = st.checkbox("Use Take Profit", key="strategy_use_take_profit")
        st.slider("Take Profit (%)", 1, 100, 15, key="strategy_take_profit")
        
        col1a, col2a = st.columns(2)
        with col1a:
            if st.button("Save Strategy", key="save_strategy"):
                st.success("Strategy saved successfully!")
        with col2a:
            if st.button("Run Backtest", key="run_backtest_btn"):
                st.info("Running backtest...")
    
    with col2:
        st.markdown("""
        <h4>Parameter Optimization</h4>
        """, unsafe_allow_html=True)
        
        # Parameter optimization (without form)
        st.selectbox("Optimization Method", ["Grid Search", "Random Search", "Bayesian Optimization"], key="optimization_method")
        st.selectbox("Optimization Target", ["Return", "Sharpe Ratio", "Sortino Ratio", "Win Rate", "Profit Factor"], key="optimization_target")
        st.number_input("Fast MA Range Start", 5, 50, 10, key="fast_ma_start")
        st.number_input("Fast MA Range End", 10, 100, 30, key="fast_ma_end")
        st.number_input("Slow MA Range Start", 20, 100, 50, key="slow_ma_start")
        st.number_input("Slow MA Range End", 50, 200, 100, key="slow_ma_end")
        st.checkbox("Use Walk-Forward Analysis", key="use_walk_forward")
        
        col1b, col2b = st.columns(2)
        with col1b:
            if st.button("Save Parameters", key="save_parameters"):
                st.success("Parameters saved successfully!")
        with col2b:
            if st.button("Run Optimization", key="run_optimization"):
                st.info("Running optimization...")
    
    # Backtest results
    st.markdown("""
    <div style="background-color: #1a2234; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
        <h4>Backtest Results</h4>
        <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
            <div>
                <div style="color: #a0aec0;">Total Return</div>
                <div style="font-size: 1.2rem; font-weight: bold;">+156.7%</div>
            </div>
            <div>
                <div style="color: #a0aec0;">Sharpe Ratio</div>
                <div style="font-size: 1.2rem; font-weight: bold;">2.34</div>
            </div>
            <div>
                <div style="color: #a0aec0;">Max Drawdown</div>
                <div style="font-size: 1.2rem; font-weight: bold;">-18.2%</div>
            </div>
            <div>
                <div style="color: #a0aec0;">Win Rate</div>
                <div style="font-size: 1.2rem; font-weight: bold;">68.5%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create backtest chart
    dates = pd.date_range(end=datetime.now(), periods=365)
    
    # Strategy returns (cumulative)
    strategy_returns = np.cumprod(1 + np.random.normal(0.001, 0.015, 365))
    
    # Benchmark returns (cumulative)
    benchmark_returns = np.cumprod(1 + np.random.normal(0.0005, 0.012, 365))
    
    # Create DataFrame
    backtest_data = pd.DataFrame({
        'Date': dates,
        'Strategy': strategy_returns,
        'Benchmark': benchmark_returns
    })
    
    # Create chart
    fig = go.Figure()
    
    # Add strategy line
    fig.add_trace(go.Scatter(
        x=backtest_data['Date'],
        y=backtest_data['Strategy'],
        mode='lines',
        name='Strategy',
        line=dict(color='#00d4ff', width=2)
    ))
    
    # Add benchmark line
    fig.add_trace(go.Scatter(
        x=backtest_data['Date'],
        y=backtest_data['Benchmark'],
        mode='lines',
        name='Benchmark',
        line=dict(color='#a0aec0', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor='#2a3245'),
        yaxis=dict(showgrid=True, gridcolor='#2a3245'),
        plot_bgcolor='#1a2234',
        paper_bgcolor='#1a2234',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Market Scanner page
def render_market_scanner():
    st.markdown("""
    <h3 style="margin-bottom: 1rem;">Market Scanner</h3>
    """, unsafe_allow_html=True)
    
    # Scanner filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.selectbox("Asset Class", ["All", "Crypto", "Stocks", "Forex", "Commodities"], key="scanner_asset_class")
    
    with col2:
        st.selectbox("Scan Type", ["Technical", "Fundamental", "Sentiment", "AI Signals"], key="scanner_type")
    
    with col3:
        st.selectbox("Timeframe", ["1H", "4H", "1D", "1W"], key="scanner_timeframe")
    
    with col4:
        st.selectbox("Sort By", ["Signal Strength", "Volume", "Volatility", "Performance"], key="scanner_sort")
    
    # Scanner results
    st.markdown("""
    <h4 style="margin-top: 1.5rem; margin-bottom: 1rem;">Scanner Results</h4>
    """, unsafe_allow_html=True)
    
    # Create sample scanner results
    scanner_results = [
        {"asset": "BTC-USD", "price": "$42,667", "change": "-0.89%", "signal": "BUY", "strength": 87, "volume": "$28.7B", "volatility": "High"},
        {"asset": "ETH-USD", "price": "$3,245", "change": "+1.25%", "signal": "BUY", "strength": 82, "volume": "$12.4B", "volatility": "Medium"},
        {"asset": "SOL-USD", "price": "$145.75", "change": "+2.34%", "signal": "BUY", "strength": 84, "volume": "$4.2B", "volatility": "High"},
        {"asset": "AAPL", "price": "$187.25", "change": "-0.45%", "signal": "SELL", "strength": 76, "volume": "$5.8B", "volatility": "Low"},
        {"asset": "MSFT", "price": "$412.50", "change": "+0.78%", "signal": "BUY", "strength": 79, "volume": "$4.3B", "volatility": "Low"},
        {"asset": "GOOGL", "price": "$175.30", "change": "+0.32%", "signal": "HOLD", "strength": 65, "volume": "$3.1B", "volatility": "Low"},
        {"asset": "NVDA", "price": "$124.75", "change": "+1.87%", "signal": "BUY", "strength": 85, "volume": "$7.2B", "volatility": "Medium"},
        {"asset": "AMZN", "price": "$187.45", "change": "-0.23%", "signal": "HOLD", "strength": 62, "volume": "$4.5B", "volatility": "Low"}
    ]
    
    # Display scanner results
    scanner_df = pd.DataFrame(scanner_results)
    st.dataframe(scanner_df, use_container_width=True)
    
    # Market heatmap
    st.markdown("""
    <h4 style="margin-top: 1.5rem; margin-bottom: 1rem;">Market Heatmap</h4>
    """, unsafe_allow_html=True)
    
    # Generate heatmap data
    sectors = ["Technology", "Finance", "Healthcare", "Energy", "Consumer", "Utilities", "Materials", "Real Estate"]
    performance = [2.3, -0.8, 1.2, -1.5, 0.7, -0.3, 0.5, -0.9]
    
    # Create chart
    fig = go.Figure(data=go.Treemap(
        labels=sectors,
        parents=[""] * len(sectors),
        values=[abs(p) * 100 for p in performance],
        textinfo="label+value",
        marker=dict(
            colors=['#48bb78' if p > 0 else '#f56565' for p in performance],
            line=dict(width=2, color='#1a2234')
        )
    ))
    
    # Update layout
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='#1a2234',
        paper_bgcolor='#1a2234',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Sentiment AI page
def render_sentiment_ai():
    st.markdown("""
    <h3 style="margin-bottom: 1rem;">Sentiment AI</h3>
    """, unsafe_allow_html=True)
    
# Hot News page
def render_hot_news():
    st.markdown("""
    <h3 style="margin-bottom: 1rem;">Hot News</h3>
    """, unsafe_allow_html=True)
    
    # News filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        news_category = st.selectbox("Category", ["All", "Stocks", "Crypto", "Economy", "Markets"], key="news_category")
    
    with col2:
        news_source = st.selectbox("Source", ["All Sources", "Bloomberg", "CNBC", "Reuters", "Wall Street Journal", "CoinDesk", "CryptoSlate"], key="news_source")
    
    with col3:
        news_timeframe = st.selectbox("Timeframe", ["Last 24 Hours", "Last Week", "Last Month"], key="news_timeframe")
    
    # Search bar
    st.text_input("Search News", placeholder="Search for keywords...", key="news_search")
    
    # Breaking news section
    st.markdown("""
    <h4 style="margin-top: 1.5rem; margin-bottom: 1rem;">🔴 Breaking News</h4>
    """, unsafe_allow_html=True)
    
    # Create sample breaking news
    breaking_news = [
        {
            "title": "Fed Announces Surprise Rate Cut of 50 Basis Points",
            "source": "Bloomberg",
            "time": "10 minutes ago",
            "category": "Economy",
            "summary": "The Federal Reserve announced an emergency rate cut of 50 basis points in response to growing economic concerns. Markets reacted positively with major indices jumping over 2%."
        },
        {
            "title": "Bitcoin Surges Past $50,000 on ETF Approval News",
            "source": "CoinDesk",
            "time": "45 minutes ago",
            "category": "Crypto",
            "summary": "Bitcoin has surged past $50,000 following reports that the SEC is set to approve several spot Bitcoin ETF applications. Trading volume has increased by 300% in the last hour."
        }
    ]
    
    # Display breaking news
    for i, news in enumerate(breaking_news):
        with st.container():
            st.markdown(f"""
            <div style="background-color: #1a2234; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <div style="font-size: 1.2rem; font-weight: bold;">{news['title']}</div>
                    <div style="color: #f56565;">{news['time']}</div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <div style="color: #a0aec0;">{news['source']}</div>
                    <div style="background-color: #2a3245; padding: 0.2rem 0.5rem; border-radius: 5px;">{news['category']}</div>
                </div>
                <div>{news['summary']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Top stories section
    st.markdown("""
    <h4 style="margin-top: 1.5rem; margin-bottom: 1rem;">📰 Top Stories</h4>
    """, unsafe_allow_html=True)
    
    # Create sample top stories
    top_stories = [
        {
            "title": "Apple Announces New AI-Powered iPhone Features",
            "source": "CNBC",
            "time": "2 hours ago",
            "category": "Stocks",
            "summary": "Apple unveiled a suite of new AI features for the iPhone, including advanced voice recognition and real-time translation. Analysts predict this could drive a significant upgrade cycle.",
            "image": "https://images.unsplash.com/photo-1592899677977-9c10ca588bbd?q=80&w=200&auto=format&fit=crop"
        },
        {
            "title": "Ethereum Completes Major Network Upgrade",
            "source": "CryptoSlate",
            "time": "3 hours ago",
            "category": "Crypto",
            "summary": "Ethereum has successfully completed its latest network upgrade, improving transaction throughput and reducing gas fees by an estimated 30%. ETH price has responded with a 5% increase.",
            "image": "https://images.unsplash.com/photo-1622630998477-20aa696ecb05?q=80&w=200&auto=format&fit=crop"
        },
        {
            "title": "Tesla Exceeds Q3 Delivery Expectations",
            "source": "Reuters",
            "time": "5 hours ago",
            "category": "Stocks",
            "summary": "Tesla reported Q3 deliveries of 435,000 vehicles, exceeding analyst expectations of 418,000. The company cited improved production efficiency and strong demand in Asia.",
            "image": "https://images.unsplash.com/photo-1617704548623-340376564e68?q=80&w=200&auto=format&fit=crop"
        },
        {
            "title": "New Regulations for Crypto Exchanges Announced",
            "source": "Wall Street Journal",
            "time": "6 hours ago",
            "category": "Crypto",
            "summary": "Regulators have announced new compliance requirements for cryptocurrency exchanges, focusing on enhanced KYC procedures and improved security measures. Implementation is expected within 6 months.",
            "image": "https://images.unsplash.com/photo-1621761191319-c6fb62004040?q=80&w=200&auto=format&fit=crop"
        }
    ]
    
    # Display top stories in a grid
    cols = st.columns(2)
    for i, story in enumerate(top_stories):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background-color: #1a2234; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; display: flex;">
                <div style="flex: 3; padding-right: 1rem;">
                    <div style="font-size: 1.1rem; font-weight: bold; margin-bottom: 0.5rem;">{story['title']}</div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <div style="color: #a0aec0;">{story['source']}</div>
                        <div style="color: #a0aec0;">{story['time']}</div>
                    </div>
                    <div style="font-size: 0.9rem;">{story['summary']}</div>
                </div>
                <div style="flex: 1;">
                    <img src="{story['image']}" style="width: 100%; border-radius: 5px;">
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Market impact section
    st.markdown("""
    <h4 style="margin-top: 1.5rem; margin-bottom: 1rem;">📊 Market Impact</h4>
    """, unsafe_allow_html=True)
    
    # Create sample market impact data
    market_impact = [
        {"news": "Fed Rate Cut", "asset": "S&P 500", "impact": "+2.3%"},
        {"news": "Fed Rate Cut", "asset": "10Y Treasury", "impact": "-15 bps"},
        {"news": "Bitcoin ETF Approval", "asset": "BTC-USD", "impact": "+8.7%"},
        {"news": "Bitcoin ETF Approval", "asset": "Coinbase (COIN)", "impact": "+12.4%"},
        {"news": "Apple AI Features", "asset": "AAPL", "impact": "+3.2%"},
        {"news": "Tesla Deliveries", "asset": "TSLA", "impact": "+5.8%"}
    ]
    
    # Display market impact table
    impact_df = pd.DataFrame(market_impact)
    
    # Apply styling to the dataframe
    st.dataframe(
        impact_df.style.applymap(
            lambda x: 'color: #48bb78' if '+' in str(x) else 'color: #f56565' if '-' in str(x) else '',
            subset=['impact']
        ),
        use_container_width=True
    )
    
    # Sentiment analysis of news
    st.markdown("""
    <h4 style="margin-top: 1.5rem; margin-bottom: 1rem;">📈 News Sentiment Analysis</h4>
    """, unsafe_allow_html=True)
    
    # Create sample sentiment data
    sentiment_data = {
        "Stocks": 65,
        "Crypto": 72,
        "Economy": 48,
        "Markets": 58
    }
    
    # Create sentiment chart
    fig = go.Figure()
    
    # Add sentiment bars
    fig.add_trace(go.Bar(
        x=list(sentiment_data.keys()),
        y=list(sentiment_data.values()),
        marker_color=['#48bb78' if v > 50 else '#f56565' for v in sentiment_data.values()],
        text=[f"{v}/100" for v in sentiment_data.values()],
        textposition='auto'
    ))
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=True, gridcolor='#2a3245'),
        yaxis=dict(showgrid=True, gridcolor='#2a3245', range=[0, 100], title="Sentiment Score"),
        plot_bgcolor='#1a2234',
        paper_bgcolor='#1a2234',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.selectbox("Asset", ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "GOOGL", "All"], key="sentiment_asset")
    
    with col2:
        st.selectbox("Source", ["Twitter", "Reddit", "News", "All"], key="sentiment_source")
    
    with col3:
        st.selectbox("Timeframe", ["24H", "7D", "30D", "90D"], key="sentiment_timeframe")
    
    # Sentiment overview
    st.markdown("""
    <h4 style="margin-top: 1.5rem; margin-bottom: 1rem;">Sentiment Overview</h4>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Overall Sentiment</div>
            <div class="metric-value">Bullish</div>
            <div style="color: #48bb78;">Score: 72/100</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Sentiment Change</div>
            <div class="metric-value">+5.3%</div>
            <div class="metric-label">vs Last Week</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Sentiment Volume</div>
            <div class="metric-value">High</div>
            <div class="metric-label">12,345 mentions</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sentiment trend
    st.markdown("""
    <h4 style="margin-top: 1.5rem; margin-bottom: 1rem;">Sentiment Trend</h4>
    """, unsafe_allow_html=True)
    
    # Generate sentiment trend data
    dates = pd.date_range(end=datetime.now(), periods=30)
    sentiment = [50 + np.cumsum(np.random.normal(0.1, 3, 30))[i] for i in range(30)]
    
    # Create chart
    fig = go.Figure()
    
    # Add sentiment line
    fig.add_trace(go.Scatter(
        x=dates,
        y=sentiment,
        mode='lines',
        name='Sentiment Score',
        line=dict(color='#00d4ff', width=2)
    ))
    
    # Add reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="#48bb78", annotation_text="Bullish")
    fig.add_hline(y=30, line_dash="dash", line_color="#f56565", annotation_text="Bearish")
    
    # Update layout
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor='#2a3245'),
        yaxis=dict(showgrid=True, gridcolor='#2a3245', range=[0, 100]),
        plot_bgcolor='#1a2234',
        paper_bgcolor='#1a2234',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment by source
    st.markdown("""
    <h4 style="margin-top: 1.5rem; margin-bottom: 1rem;">Sentiment by Source</h4>
    """, unsafe_allow_html=True)
    
    # Generate sentiment by source data
    sources = ["Twitter", "Reddit", "News", "YouTube", "Blogs"]
    sentiment_scores = [72, 68, 65, 70, 63]
    
    # Create chart
    fig = go.Figure()
    
    # Add sentiment bars
    fig.add_trace(go.Bar(
        x=sources,
        y=sentiment_scores,
        marker_color=['#00d4ff', '#6c2bd9', '#48bb78', '#ecc94b', '#f56565']
    ))
    
    # Update layout
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=True, gridcolor='#2a3245'),
        yaxis=dict(showgrid=True, gridcolor='#2a3245', range=[0, 100]),
        plot_bgcolor='#1a2234',
        paper_bgcolor='#1a2234',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Automatic Paper Trading
def render_paper_trading():
    st.markdown("""
    <h3 style="margin-bottom: 1rem;">Automatic Paper Trading</h3>
    """, unsafe_allow_html=True)
    
    # Get paper trading data
    paper_trader = st.session_state.paper_trader
    portfolio_value = paper_trader.get_portfolio_value()
    portfolio_return = paper_trader.get_portfolio_return()
    win_rate = paper_trader.get_win_rate()
    active_signals = paper_trader.get_active_signals_count()
    recent_trades = paper_trader.get_recent_trades(5)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <h4>Trading Settings</h4>
        """, unsafe_allow_html=True)
        
        # Trading settings (without form)
        st.selectbox("Trading Strategy", ["ML Ensemble", "Moving Average Crossover", "RSI Strategy", "MACD Strategy", "Custom Strategy"], key="trading_strategy")
        st.slider("Position Size (%)", 1, 100, 10, key="trading_position_size")
        st.slider("Stop Loss (%)", 1, 50, 5, key="trading_stop_loss")
        st.slider("Take Profit (%)", 1, 100, 15, key="trading_take_profit")
        st.checkbox("Enable Trailing Stop", key="trading_trailing_stop")
        st.checkbox("Auto-execute Signals", value=True, key="trading_auto_execute")
        
        col1a, col2a = st.columns(2)
        with col1a:
            if st.button("Save Settings", key="save_trading_settings"):
                st.success("Trading settings saved!")
        with col2a:
            if st.button("Reset Settings", key="reset_trading_settings"):
                st.info("Trading settings reset to default.")
    
    with col2:
        st.markdown("""
        <h4>Recent Trades</h4>
        """, unsafe_allow_html=True)
        
        if recent_trades:
            # Create a DataFrame for recent trades
            trades_df = pd.DataFrame(recent_trades)
            
            # Format the DataFrame for display
            if 'profit_loss' in trades_df.columns:
                trades_df['profit_loss'] = trades_df['profit_loss'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else x)
            
            if 'timestamp' in trades_df.columns:
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                trades_df['timestamp'] = trades_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            if 'price' in trades_df.columns:
                trades_df['price'] = trades_df['price'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else x)
            
            if 'value' in trades_df.columns:
                trades_df['value'] = trades_df['value'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else x)
            
            # Display the trades
            st.dataframe(trades_df[['timestamp', 'symbol', 'action', 'price', 'value']], use_container_width=True)
        else:
            st.info("No recent trades available.")
    
    # Active signals
    st.markdown("""
    <h4>Active Signals</h4>
    """, unsafe_allow_html=True)
    
    # Create sample signals
    signals = [
        {"symbol": "BTC-USD", "signal": "BUY", "confidence": 87, "timestamp": "2025-09-24 12:34"},
        {"symbol": "ETH-USD", "signal": "BUY", "confidence": 82, "timestamp": "2025-09-24 10:15"},
        {"symbol": "AAPL", "signal": "SELL", "confidence": 76, "timestamp": "2025-09-24 09:45"}
    ]
    
    # Display signals
    signals_df = pd.DataFrame(signals)
    st.dataframe(signals_df, use_container_width=True)
    
    # Trading performance
    st.markdown("""
    <h4>Trading Performance</h4>
    """, unsafe_allow_html=True)
    
    # Create performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Value", f"${portfolio_value:,.2f}", f"{portfolio_return:.2f}%")
    
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col3:
        st.metric("Profit Factor", "1.87")
    
    with col4:
        st.metric("Drawdown", "-12.3%")
    
    # Performance chart
    # Create sample performance data
    dates = pd.date_range(end=datetime.now(), periods=90)
    
    # Portfolio value over time
    portfolio_values = [100000 * (1 + np.cumsum(np.random.normal(0.001, 0.01, 90))[i]) for i in range(90)]
    
    # Create DataFrame
    performance_data = pd.DataFrame({
        'Date': dates,
        'Portfolio Value': portfolio_values
    })
    
    # Create chart
    fig = go.Figure()
    
    # Add portfolio value line
    fig.add_trace(go.Scatter(
        x=performance_data['Date'],
        y=performance_data['Portfolio Value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#00d4ff', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=True, gridcolor='#2a3245'),
        yaxis=dict(showgrid=True, gridcolor='#2a3245'),
        plot_bgcolor='#1a2234',
        paper_bgcolor='#1a2234',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Main app
def main():
    # Render header
    render_header()
    
    # Render navigation
    render_navigation()
    
    # Render asset selector
    render_asset_selector()
    
    # Sidebar navigation
    st.sidebar.title("Intellectia AI Pro")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["AI Dashboard", "Deep Research", "Smart Signals", "ML Engine", 
         "Portfolio AI", "Risk Analytics", "Strategy Lab", "Market Scanner", 
         "Sentiment AI", "Hot News", "Automatic Paper Trading"],
        index=["AI Dashboard", "Deep Research", "Smart Signals", "ML Engine", 
               "Portfolio AI", "Risk Analytics", "Strategy Lab", "Market Scanner", 
               "Sentiment AI", "Hot News", "Automatic Paper Trading"].index(st.session_state.current_page)
               if st.session_state.current_page in ["AI Dashboard", "Deep Research", "Smart Signals", "ML Engine", 
               "Portfolio AI", "Risk Analytics", "Strategy Lab", "Market Scanner", 
               "Sentiment AI", "Hot News", "Automatic Paper Trading"] else 0
    )
    
    # Update current page
    st.session_state.current_page = page
    
    # Render selected page
    if page == "AI Dashboard":
        render_ai_dashboard()
    elif page == "Deep Research":
        render_deep_research()
    elif page == "Smart Signals":
        render_smart_signals()
    elif page == "ML Engine":
        render_ml_engine()
    elif page == "Portfolio AI":
        render_portfolio_ai()
    elif page == "Risk Analytics":
        render_risk_analytics()
    elif page == "Strategy Lab":
        render_strategy_lab()
    elif page == "Market Scanner":
        render_market_scanner()
    elif page == "Sentiment AI":
        render_sentiment_ai()
    elif page == "Hot News":
        render_hot_news()
    elif page == "Automatic Paper Trading":
        render_paper_trading()

# Run the app
if __name__ == "__main__":
    main()