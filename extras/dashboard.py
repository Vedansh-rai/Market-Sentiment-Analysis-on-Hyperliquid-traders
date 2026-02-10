"""
Market Sentiment Analysis Dashboard
Interactive exploration of Fear/Greed impact on Hyperliquid traders
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Market Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .fear-card {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
    }
    .greed-card {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and prepare the datasets"""
    sentiment_df = pd.read_csv('data/bitcoin_sentiment.csv')
    trader_df = pd.read_csv('data/hyperliquid_trader_data.csv')
    
    # Clean sentiment data
    sentiment_df['sentiment'] = sentiment_df['classification'].apply(
        lambda x: 'Fear' if 'Fear' in x else 'Greed'
    )
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    
    # Parse trader timestamps
    trader_df['datetime'] = pd.to_datetime(trader_df['Timestamp IST'], format='%d-%m-%Y %H:%M')
    trader_df['date'] = trader_df['datetime'].dt.date
    trader_df['date'] = pd.to_datetime(trader_df['date'])
    
    # Merge datasets
    merged_df = trader_df.merge(
        sentiment_df[['date', 'sentiment', 'value', 'classification']], 
        on='date', 
        how='inner'
    )
    
    # Rename and clean
    merged_df = merged_df.rename(columns={
        'Account': 'account',
        'Coin': 'coin',
        'Execution Price': 'price',
        'Size Tokens': 'size_tokens',
        'Size USD': 'size_usd',
        'Side': 'side',
        'Closed PnL': 'pnl',
        'Fee': 'fee'
    })
    
    merged_df['pnl'] = pd.to_numeric(merged_df['pnl'], errors='coerce').fillna(0)
    merged_df['size_usd'] = pd.to_numeric(merged_df['size_usd'], errors='coerce').fillna(0)
    merged_df['is_win'] = (merged_df['pnl'] > 0).astype(int)
    merged_df['is_long'] = (merged_df['side'] == 'BUY').astype(int)
    
    return sentiment_df, trader_df, merged_df

# Load data
try:
    sentiment_df, trader_df, merged_df = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

# Header
st.markdown('<p class="main-header">📊 Market Sentiment Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown("**Analyze how Fear/Greed sentiment impacts Hyperliquid trader behavior**")
st.divider()

if data_loaded:
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(merged_df['date'].min(), merged_df['date'].max()),
        min_value=merged_df['date'].min(),
        max_value=merged_df['date'].max()
    )
    
    sentiment_filter = st.sidebar.multiselect(
        "Sentiment",
        options=['Fear', 'Greed'],
        default=['Fear', 'Greed']
    )
    
    coins = st.sidebar.multiselect(
        "Coins",
        options=merged_df['coin'].unique().tolist()[:20],
        default=merged_df['coin'].unique().tolist()[:5]
    )
    
    # Filter data
    if len(date_range) == 2:
        mask = (
            (merged_df['date'] >= pd.Timestamp(date_range[0])) &
            (merged_df['date'] <= pd.Timestamp(date_range[1])) &
            (merged_df['sentiment'].isin(sentiment_filter))
        )
        if coins:
            mask &= merged_df['coin'].isin(coins)
        filtered_df = merged_df[mask]
    else:
        filtered_df = merged_df
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Trades", f"{len(filtered_df):,}")
    
    with col2:
        st.metric("👥 Unique Traders", f"{filtered_df['account'].nunique():,}")
    
    with col3:
        total_pnl = filtered_df['pnl'].sum()
        st.metric("💰 Total PnL", f"${total_pnl:,.2f}")
    
    with col4:
        win_rate = filtered_df['is_win'].mean() * 100 if len(filtered_df) > 0 else 0
        st.metric("📈 Win Rate", f"{win_rate:.1f}%")
    
    st.divider()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "🔄 Behavior", "👥 Segments", "🤖 Predictions"])
    
    with tab1:
        st.header("Fear vs Greed Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # PnL by Sentiment
            fig, ax = plt.subplots(figsize=(8, 5))
            pnl_by_sentiment = filtered_df.groupby('sentiment')['pnl'].sum()
            colors = ['#e74c3c' if s == 'Fear' else '#2ecc71' for s in pnl_by_sentiment.index]
            bars = ax.bar(pnl_by_sentiment.index, pnl_by_sentiment.values, color=colors, edgecolor='black')
            ax.set_title('Total PnL by Sentiment', fontsize=14, fontweight='bold')
            ax.set_ylabel('Total PnL ($)')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            st.pyplot(fig)
        
        with col2:
            # Win Rate by Sentiment
            fig, ax = plt.subplots(figsize=(8, 5))
            win_rate_by_sentiment = filtered_df.groupby('sentiment')['is_win'].mean() * 100
            colors = ['#e74c3c' if s == 'Fear' else '#2ecc71' for s in win_rate_by_sentiment.index]
            bars = ax.bar(win_rate_by_sentiment.index, win_rate_by_sentiment.values, color=colors, edgecolor='black')
            ax.set_title('Win Rate by Sentiment', fontsize=14, fontweight='bold')
            ax.set_ylabel('Win Rate (%)')
            ax.set_ylim(0, 100)
            st.pyplot(fig)
        
        # Summary table
        st.subheader("Performance Summary")
        summary = filtered_df.groupby('sentiment').agg({
            'pnl': ['sum', 'mean', 'count'],
            'is_win': 'mean',
            'size_usd': 'mean'
        }).round(2)
        summary.columns = ['Total PnL', 'Avg PnL', 'Trade Count', 'Win Rate', 'Avg Size']
        st.dataframe(summary, use_container_width=True)
    
    with tab2:
        st.header("Trading Behavior by Sentiment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Trade Frequency
            fig, ax = plt.subplots(figsize=(6, 4))
            freq = filtered_df.groupby('sentiment').size()
            colors = ['#e74c3c', '#2ecc71']
            ax.bar(freq.index, freq.values, color=colors, edgecolor='black')
            ax.set_title('Trade Count', fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            # Long/Short Ratio
            fig, ax = plt.subplots(figsize=(6, 4))
            long_ratio = filtered_df.groupby('sentiment')['is_long'].mean() * 100
            ax.bar(long_ratio.index, long_ratio.values, color=colors, edgecolor='black')
            ax.set_title('Long Trade %', fontweight='bold')
            ax.axhline(y=50, color='gray', linestyle='--')
            st.pyplot(fig)
        
        with col3:
            # Avg Trade Size
            fig, ax = plt.subplots(figsize=(6, 4))
            avg_size = filtered_df.groupby('sentiment')['size_usd'].mean()
            ax.bar(avg_size.index, avg_size.values, color=colors, edgecolor='black')
            ax.set_title('Avg Trade Size ($)', fontweight='bold')
            st.pyplot(fig)
    
    with tab3:
        st.header("Trader Segmentation")
        
        # Create trader metrics
        trader_metrics = filtered_df.groupby('account').agg({
            'pnl': ['sum', 'count', 'mean'],
            'is_win': 'mean',
            'size_usd': 'mean',
            'is_long': 'mean'
        }).reset_index()
        trader_metrics.columns = ['account', 'total_pnl', 'trade_count', 'avg_pnl', 'win_rate', 'avg_size', 'long_ratio']
        
        # Size segments
        trader_metrics['size_segment'] = pd.qcut(
            trader_metrics['avg_size'], 
            q=3, 
            labels=['Small', 'Medium', 'Large'],
            duplicates='drop'
        )
        
        # Display top traders
        st.subheader("Top Performing Traders")
        top_traders = trader_metrics.nlargest(10, 'total_pnl')[['account', 'total_pnl', 'trade_count', 'win_rate', 'avg_size']]
        top_traders['total_pnl'] = top_traders['total_pnl'].apply(lambda x: f"${x:,.2f}")
        top_traders['win_rate'] = top_traders['win_rate'].apply(lambda x: f"{x*100:.1f}%")
        top_traders['avg_size'] = top_traders['avg_size'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(top_traders, use_container_width=True)
        
        # Segment distribution
        st.subheader("Segment Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            trader_metrics['size_segment'].value_counts().plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title('Traders by Size Segment', fontweight='bold')
            ax.set_ylabel('Count')
            plt.xticks(rotation=0)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(trader_metrics['avg_size'], trader_metrics['win_rate'], alpha=0.5, c='steelblue')
            ax.set_xlabel('Avg Trade Size ($)')
            ax.set_ylabel('Win Rate')
            ax.set_title('Trade Size vs Win Rate', fontweight='bold')
            st.pyplot(fig)
    
    with tab4:
        st.header("Strategy Insights")
        
        st.info("""
        ### 💡 Strategy 1: Sentiment-Based Sizing
        - **During Fear**: Reduce position sizes by 20-30% for large traders
        - **During Greed**: Maintain normal sizing with gradual scaling
        
        ### 💡 Strategy 2: Frequency Optimization
        - **Infrequent traders in Fear**: Increase selectivity, wait for extreme readings
        - **Frequent traders in Greed**: Reduce frequency by 15-20%
        """)
        
        # Show sentiment distribution
        st.subheader("Current Sentiment Distribution")
        sentiment_counts = filtered_df['sentiment'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔴 Fear Days Trades", f"{sentiment_counts.get('Fear', 0):,}")
        with col2:
            st.metric("🟢 Greed Days Trades", f"{sentiment_counts.get('Greed', 0):,}")
        
        # Historical sentiment timeline
        st.subheader("Sentiment Timeline")
        daily_sentiment = filtered_df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        st.line_chart(daily_sentiment)

else:
    st.warning("Please ensure the data files are in the `data/` directory.")
    st.code("""
    Required files:
    - data/bitcoin_sentiment.csv
    - data/hyperliquid_trader_data.csv
    """)

# Footer
st.divider()
st.markdown("*Built for Hyperliquid Market Sentiment Analysis*")
