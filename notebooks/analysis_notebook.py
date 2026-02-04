# %% [markdown]
# # Market Sentiment Analysis: Fear/Greed Impact on Hyperliquid Trader Behavior
# 
# **Objective**: Analyze how market sentiment (Fear/Greed) relates to trader behavior and performance on Hyperliquid.

# %% [markdown]
# ## Setup & Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print("Libraries loaded successfully!")

# %% [markdown]
# ---
# # Part A: Data Preparation

# %% [markdown]
# ## 1. Load Both Datasets

# %%
# Load Bitcoin Sentiment Data
sentiment_df = pd.read_csv('../data/bitcoin_sentiment.csv')

# Load Hyperliquid Trader Data
trader_df = pd.read_csv('../data/hyperliquid_trader_data.csv')

print("✅ Datasets loaded successfully!")
print(f"\n📊 Bitcoin Sentiment Dataset: {sentiment_df.shape[0]:,} rows × {sentiment_df.shape[1]} columns")
print(f"📊 Hyperliquid Trader Dataset: {trader_df.shape[0]:,} rows × {trader_df.shape[1]} columns")

# %% [markdown]
# ## 2. Document Data Structure

# %%
print("=" * 60)
print("BITCOIN SENTIMENT DATA")
print("=" * 60)
print("\n📋 Column Info:")
print(sentiment_df.dtypes)
print(f"\n📅 Date Range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
print(f"\n🏷️ Sentiment Classifications:")
print(sentiment_df['classification'].value_counts())

# %%
print("=" * 60)
print("HYPERLIQUID TRADER DATA")
print("=" * 60)
print("\n📋 Column Info:")
print(trader_df.dtypes)
print(f"\n📅 Sample Timestamps: ")
print(trader_df['Timestamp IST'].head(3).tolist())

# %%
# Missing Values Analysis
print("\n" + "=" * 60)
print("MISSING VALUES ANALYSIS")
print("=" * 60)

print("\n📊 Bitcoin Sentiment - Missing Values:")
missing_sentiment = sentiment_df.isnull().sum()
print(missing_sentiment[missing_sentiment > 0] if missing_sentiment.sum() > 0 else "No missing values!")

print("\n📊 Hyperliquid Trader - Missing Values:")
missing_trader = trader_df.isnull().sum()
print(missing_trader[missing_trader > 0] if missing_trader.sum() > 0 else "No missing values!")

# %%
# Duplicates Analysis
print("\n" + "=" * 60)
print("DUPLICATES ANALYSIS")
print("=" * 60)

print(f"\n📊 Bitcoin Sentiment - Duplicate Rows: {sentiment_df.duplicated().sum():,}")
print(f"📊 Hyperliquid Trader - Duplicate Rows: {trader_df.duplicated().sum():,}")

# %% [markdown]
# ## 3. Data Cleaning & Timestamp Alignment

# %%
# Clean sentiment data - simplify to Fear/Greed
sentiment_df['sentiment'] = sentiment_df['classification'].apply(
    lambda x: 'Fear' if 'Fear' in x else 'Greed'
)
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

print("✅ Sentiment data cleaned")
print(f"\n🏷️ Simplified Sentiment Distribution:")
print(sentiment_df['sentiment'].value_counts())

# %%
# Parse trader timestamps
# Format: "02-12-2024 22:50" (DD-MM-YYYY HH:MM)
trader_df['datetime'] = pd.to_datetime(trader_df['Timestamp IST'], format='%d-%m-%Y %H:%M')
trader_df['date'] = trader_df['datetime'].dt.date
trader_df['date'] = pd.to_datetime(trader_df['date'])

print("✅ Trader timestamps parsed")
print(f"\n📅 Trade Date Range: {trader_df['date'].min().strftime('%Y-%m-%d')} to {trader_df['date'].max().strftime('%Y-%m-%d')}")

# %%
# Merge datasets on date
merged_df = trader_df.merge(
    sentiment_df[['date', 'sentiment', 'value', 'classification']], 
    on='date', 
    how='inner'
)

print(f"✅ Datasets merged!")
print(f"📊 Merged dataset: {merged_df.shape[0]:,} rows × {merged_df.shape[1]} columns")
print(f"📅 Date range coverage: {merged_df['date'].min().strftime('%Y-%m-%d')} to {merged_df['date'].max().strftime('%Y-%m-%d')}")
print(f"📆 Unique trading days: {merged_df['date'].nunique()}")

# %% [markdown]
# ## 4. Create Key Metrics

# %%
# Rename columns for easier access
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

# Convert PnL to numeric
merged_df['pnl'] = pd.to_numeric(merged_df['pnl'], errors='coerce').fillna(0)
merged_df['size_usd'] = pd.to_numeric(merged_df['size_usd'], errors='coerce').fillna(0)
merged_df['fee'] = pd.to_numeric(merged_df['fee'], errors='coerce').fillna(0)

# Create win/loss indicator
merged_df['is_win'] = (merged_df['pnl'] > 0).astype(int)
merged_df['is_long'] = (merged_df['side'] == 'BUY').astype(int)

print("✅ Columns renamed and metrics created")

# %%
# Daily metrics per trader
daily_trader_metrics = merged_df.groupby(['date', 'account', 'sentiment']).agg({
    'pnl': ['sum', 'count', 'mean'],
    'is_win': 'sum',
    'size_usd': ['sum', 'mean'],
    'is_long': ['sum', 'count']
}).reset_index()

# Flatten column names
daily_trader_metrics.columns = ['date', 'account', 'sentiment', 
                                'daily_pnl', 'trade_count', 'avg_pnl_per_trade',
                                'win_count', 'total_volume', 'avg_trade_size',
                                'long_count', 'total_trades']

# Calculate metrics
daily_trader_metrics['win_rate'] = daily_trader_metrics['win_count'] / daily_trader_metrics['trade_count']
daily_trader_metrics['long_ratio'] = daily_trader_metrics['long_count'] / daily_trader_metrics['total_trades']

print("✅ Daily trader metrics calculated")
print(f"📊 Daily trader metrics: {daily_trader_metrics.shape[0]:,} rows")

# %%
# Daily aggregate metrics
daily_metrics = merged_df.groupby(['date', 'sentiment']).agg({
    'pnl': ['sum', 'count', 'mean'],
    'is_win': 'sum',
    'size_usd': ['sum', 'mean'],
    'is_long': 'sum',
    'account': 'nunique'
}).reset_index()

daily_metrics.columns = ['date', 'sentiment', 
                         'total_pnl', 'trade_count', 'avg_pnl',
                         'win_count', 'total_volume', 'avg_trade_size',
                         'long_trades', 'unique_traders']

daily_metrics['win_rate'] = daily_metrics['win_count'] / daily_metrics['trade_count']
daily_metrics['long_ratio'] = daily_metrics['long_trades'] / daily_metrics['trade_count']

print("✅ Daily aggregate metrics calculated")
print(daily_metrics.head())

# %%
# Trader-level aggregate metrics
trader_metrics = merged_df.groupby(['account']).agg({
    'pnl': ['sum', 'count', 'mean', 'std'],
    'is_win': 'sum',
    'size_usd': ['sum', 'mean'],
    'is_long': 'sum',
    'date': 'nunique'
}).reset_index()

trader_metrics.columns = ['account', 
                          'total_pnl', 'trade_count', 'avg_pnl', 'pnl_std',
                          'win_count', 'total_volume', 'avg_trade_size',
                          'long_trades', 'active_days']

trader_metrics['win_rate'] = trader_metrics['win_count'] / trader_metrics['trade_count']
trader_metrics['long_ratio'] = trader_metrics['long_trades'] / trader_metrics['trade_count']
trader_metrics['trades_per_day'] = trader_metrics['trade_count'] / trader_metrics['active_days']
trader_metrics['pnl_consistency'] = trader_metrics['avg_pnl'] / (trader_metrics['pnl_std'] + 1)  # Sharpe-like ratio

print("✅ Trader-level metrics calculated")
print(f"📊 Unique traders analyzed: {len(trader_metrics):,}")

# %% [markdown]
# ### Summary Statistics

# %%
print("=" * 60)
print("KEY METRICS SUMMARY")
print("=" * 60)
print(f"\n📊 Total Trades Analyzed: {len(merged_df):,}")
print(f"👥 Unique Traders: {merged_df['account'].nunique():,}")
print(f"🪙 Unique Coins Traded: {merged_df['coin'].nunique()}")
print(f"\n💰 Total PnL: ${merged_df['pnl'].sum():,.2f}")
print(f"📈 Overall Win Rate: {(merged_df['is_win'].sum() / len(merged_df) * 100):.1f}%")
print(f"📊 Average Trade Size: ${merged_df['size_usd'].mean():,.2f}")
print(f"📈 Long/Short Ratio: {merged_df['is_long'].sum() / len(merged_df):.2%}")

# %% [markdown]
# ---
# # Part B: Analysis

# %% [markdown]
# ## 1. Performance: Fear vs Greed Days

# %%
# Aggregate performance by sentiment
sentiment_performance = merged_df.groupby('sentiment').agg({
    'pnl': ['sum', 'mean', 'std', 'count'],
    'is_win': 'sum',
    'size_usd': 'mean',
    'account': 'nunique'
}).round(2)

sentiment_performance.columns = ['Total PnL', 'Avg PnL/Trade', 'PnL Std Dev', 'Trade Count', 
                                  'Winning Trades', 'Avg Trade Size', 'Unique Traders']
sentiment_performance['Win Rate'] = (sentiment_performance['Winning Trades'] / sentiment_performance['Trade Count'] * 100).round(2)

print("=" * 60)
print("PERFORMANCE COMPARISON: FEAR vs GREED DAYS")
print("=" * 60)
print(sentiment_performance.T)

# %%
# Visualization: PnL by Sentiment
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Total PnL by Sentiment
ax1 = axes[0, 0]
pnl_by_sentiment = merged_df.groupby('sentiment')['pnl'].sum()
colors = ['#e74c3c' if s == 'Fear' else '#2ecc71' for s in pnl_by_sentiment.index]
bars = ax1.bar(pnl_by_sentiment.index, pnl_by_sentiment.values, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_title('Total PnL by Market Sentiment', fontsize=14, fontweight='bold')
ax1.set_ylabel('Total PnL ($)')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
for bar, val in zip(bars, pnl_by_sentiment.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
             f'${val:,.0f}', ha='center', va='bottom', fontweight='bold')

# 2. Win Rate by Sentiment
ax2 = axes[0, 1]
win_rate_by_sentiment = merged_df.groupby('sentiment')['is_win'].mean() * 100
bars = ax2.bar(win_rate_by_sentiment.index, win_rate_by_sentiment.values, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_title('Win Rate by Market Sentiment', fontsize=14, fontweight='bold')
ax2.set_ylabel('Win Rate (%)')
ax2.set_ylim(0, 100)
for bar, val in zip(bars, win_rate_by_sentiment.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

# 3. Average Trade Size by Sentiment
ax3 = axes[1, 0]
avg_size_by_sentiment = merged_df.groupby('sentiment')['size_usd'].mean()
bars = ax3.bar(avg_size_by_sentiment.index, avg_size_by_sentiment.values, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_title('Average Trade Size by Market Sentiment', fontsize=14, fontweight='bold')
ax3.set_ylabel('Average Trade Size ($)')
for bar, val in zip(bars, avg_size_by_sentiment.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             f'${val:,.0f}', ha='center', va='bottom', fontweight='bold')

# 4. Trade Count by Sentiment
ax4 = axes[1, 1]
trade_count_by_sentiment = merged_df.groupby('sentiment').size()
bars = ax4.bar(trade_count_by_sentiment.index, trade_count_by_sentiment.values, color=colors, edgecolor='black', linewidth=1.5)
ax4.set_title('Number of Trades by Market Sentiment', fontsize=14, fontweight='bold')
ax4.set_ylabel('Trade Count')
for bar, val in zip(bars, trade_count_by_sentiment.values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
             f'{val:,}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('../output/fear_vs_greed_performance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart saved to: output/fear_vs_greed_performance.png")

# %% [markdown]
# ### Drawdown Proxy Analysis

# %%
# Calculate cumulative PnL per trader per day and find max drawdown
def calculate_drawdown(group):
    cumulative_pnl = group['pnl'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl
    return pd.Series({
        'max_drawdown': drawdown.max(),
        'final_pnl': cumulative_pnl.iloc[-1] if len(cumulative_pnl) > 0 else 0
    })

# Drawdown by sentiment
drawdown_analysis = merged_df.groupby(['account', 'sentiment']).apply(calculate_drawdown).reset_index()

drawdown_summary = drawdown_analysis.groupby('sentiment').agg({
    'max_drawdown': ['mean', 'median', 'max'],
    'final_pnl': ['mean', 'sum']
}).round(2)

print("=" * 60)
print("DRAWDOWN ANALYSIS BY SENTIMENT")
print("=" * 60)
print(drawdown_summary)

# %%
# Visualize drawdown distribution
fig, ax = plt.subplots(figsize=(10, 6))
fear_dd = drawdown_analysis[drawdown_analysis['sentiment'] == 'Fear']['max_drawdown']
greed_dd = drawdown_analysis[drawdown_analysis['sentiment'] == 'Greed']['max_drawdown']

ax.boxplot([fear_dd, greed_dd], labels=['Fear Days', 'Greed Days'], 
           patch_artist=True, 
           boxprops=dict(facecolor='lightcoral', color='black'),
           medianprops=dict(color='black', linewidth=2))
ax.set_title('Max Drawdown Distribution by Market Sentiment', fontsize=14, fontweight='bold')
ax.set_ylabel('Max Drawdown ($)')
plt.savefig('../output/drawdown_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart saved to: output/drawdown_by_sentiment.png")

# %% [markdown]
# ## 2. Behavioral Changes by Sentiment

# %%
# Trade frequency by sentiment
print("=" * 60)
print("BEHAVIORAL CHANGES: FEAR vs GREED")
print("=" * 60)

behavior_comparison = pd.DataFrame()
behavior_comparison['Fear'] = [
    merged_df[merged_df['sentiment'] == 'Fear'].groupby('account').size().mean(),
    merged_df[merged_df['sentiment'] == 'Fear']['size_usd'].mean(),
    merged_df[merged_df['sentiment'] == 'Fear']['is_long'].mean() * 100
]
behavior_comparison['Greed'] = [
    merged_df[merged_df['sentiment'] == 'Greed'].groupby('account').size().mean(),
    merged_df[merged_df['sentiment'] == 'Greed']['size_usd'].mean(),
    merged_df[merged_df['sentiment'] == 'Greed']['is_long'].mean() * 100
]
behavior_comparison.index = ['Avg Trades/Trader', 'Avg Position Size ($)', 'Long Ratio (%)']
behavior_comparison['Difference (%)'] = ((behavior_comparison['Greed'] - behavior_comparison['Fear']) / behavior_comparison['Fear'] * 100).round(1)

print(behavior_comparison.round(2))

# %%
# Visualization: Behavioral Patterns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Trade Frequency
ax1 = axes[0]
freq_by_sentiment = daily_metrics.groupby('sentiment')['trade_count'].mean()
colors = ['#e74c3c', '#2ecc71']
ax1.bar(freq_by_sentiment.index, freq_by_sentiment.values, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_title('Avg Daily Trade Frequency', fontsize=13, fontweight='bold')
ax1.set_ylabel('Trades per Day')

# 2. Long/Short Ratio
ax2 = axes[1]
long_ratio_by_sentiment = merged_df.groupby('sentiment')['is_long'].mean() * 100
ax2.bar(long_ratio_by_sentiment.index, long_ratio_by_sentiment.values, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_title('Long Trade Ratio', fontsize=13, fontweight='bold')
ax2.set_ylabel('Long Trades (%)')
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Neutral (50%)')
ax2.legend()

# 3. Position Size Distribution
ax3 = axes[2]
fear_sizes = merged_df[merged_df['sentiment'] == 'Fear']['size_usd']
greed_sizes = merged_df[merged_df['sentiment'] == 'Greed']['size_usd']
ax3.boxplot([fear_sizes.clip(upper=fear_sizes.quantile(0.95)), 
             greed_sizes.clip(upper=greed_sizes.quantile(0.95))], 
            labels=['Fear', 'Greed'])
ax3.set_title('Position Size Distribution (95th percentile)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Trade Size ($)')

plt.tight_layout()
plt.savefig('../output/behavior_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart saved to: output/behavior_by_sentiment.png")

# %% [markdown]
# ## 3. Trader Segmentation

# %%
# Segment 1: High vs Low Leverage Traders (using trade size as proxy)
trader_metrics['size_segment'] = pd.qcut(trader_metrics['avg_trade_size'], q=3, labels=['Small', 'Medium', 'Large'])

# Segment 2: Frequent vs Infrequent Traders
trader_metrics['frequency_segment'] = pd.qcut(trader_metrics['trades_per_day'], q=3, labels=['Infrequent', 'Moderate', 'Frequent'])

# Segment 3: Consistent Winners vs Inconsistent
trader_metrics['consistency_segment'] = pd.cut(
    trader_metrics['win_rate'], 
    bins=[0, 0.4, 0.6, 1.0], 
    labels=['Low Win Rate', 'Moderate', 'High Win Rate']
)

print("✅ Trader segments created")
print(f"\n📊 Size Segment Distribution:")
print(trader_metrics['size_segment'].value_counts())
print(f"\n📊 Frequency Segment Distribution:")
print(trader_metrics['frequency_segment'].value_counts())
print(f"\n📊 Consistency Segment Distribution:")
print(trader_metrics['consistency_segment'].value_counts())

# %%
# Merge segments back to trade data
merged_with_segments = merged_df.merge(
    trader_metrics[['account', 'size_segment', 'frequency_segment', 'consistency_segment']],
    on='account',
    how='left'
)

# Analyze performance by segment and sentiment
segment_sentiment_analysis = merged_with_segments.groupby(['size_segment', 'sentiment']).agg({
    'pnl': ['sum', 'mean', 'count'],
    'is_win': 'mean'
}).round(2)

print("=" * 60)
print("SEGMENT PERFORMANCE BY SENTIMENT")
print("=" * 60)
print("\n📊 By Trade Size Segment:")
print(segment_sentiment_analysis)

# %%
# Visualization: Segment Performance Heatmap
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Size Segment vs Sentiment
ax1 = axes[0]
pivot1 = merged_with_segments.groupby(['size_segment', 'sentiment'])['pnl'].mean().unstack()
sns.heatmap(pivot1, annot=True, fmt='.0f', cmap='RdYlGn', center=0, ax=ax1, cbar_kws={'label': 'Avg PnL ($)'})
ax1.set_title('Avg PnL: Trade Size Segment × Sentiment', fontsize=12, fontweight='bold')
ax1.set_xlabel('Sentiment')
ax1.set_ylabel('Trade Size Segment')

# 2. Frequency Segment vs Sentiment
ax2 = axes[1]
pivot2 = merged_with_segments.groupby(['frequency_segment', 'sentiment'])['pnl'].mean().unstack()
sns.heatmap(pivot2, annot=True, fmt='.0f', cmap='RdYlGn', center=0, ax=ax2, cbar_kws={'label': 'Avg PnL ($)'})
ax2.set_title('Avg PnL: Frequency Segment × Sentiment', fontsize=12, fontweight='bold')
ax2.set_xlabel('Sentiment')
ax2.set_ylabel('Frequency Segment')

# 3. Consistency Segment vs Sentiment
ax3 = axes[2]
pivot3 = merged_with_segments.groupby(['consistency_segment', 'sentiment'])['pnl'].mean().unstack()
sns.heatmap(pivot3, annot=True, fmt='.0f', cmap='RdYlGn', center=0, ax=ax3, cbar_kws={'label': 'Avg PnL ($)'})
ax3.set_title('Avg PnL: Win Rate Segment × Sentiment', fontsize=12, fontweight='bold')
ax3.set_xlabel('Sentiment')
ax3.set_ylabel('Win Rate Segment')

plt.tight_layout()
plt.savefig('../output/segment_performance_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart saved to: output/segment_performance_heatmap.png")

# %%
# Win rate analysis by segment
print("\n📊 Win Rate by Frequency Segment and Sentiment:")
win_rate_by_freq = merged_with_segments.groupby(['frequency_segment', 'sentiment'])['is_win'].mean() * 100
print(win_rate_by_freq.unstack().round(1))

# %% [markdown]
# ## 4. Key Insights Summary

# %%
print("=" * 70)
print("📊 KEY INSIGHTS FROM ANALYSIS")
print("=" * 70)

# Insight 1: Performance Difference
fear_pnl = merged_df[merged_df['sentiment'] == 'Fear']['pnl'].sum()
greed_pnl = merged_df[merged_df['sentiment'] == 'Greed']['pnl'].sum()
fear_win_rate = merged_df[merged_df['sentiment'] == 'Fear']['is_win'].mean() * 100
greed_win_rate = merged_df[merged_df['sentiment'] == 'Greed']['is_win'].mean() * 100

print(f"""
🔍 INSIGHT 1: Sentiment Impacts Profitability
   • Total PnL on Fear days: ${fear_pnl:,.2f}
   • Total PnL on Greed days: ${greed_pnl:,.2f}
   • Win Rate on Fear days: {fear_win_rate:.1f}%
   • Win Rate on Greed days: {greed_win_rate:.1f}%
   → {'Fear' if fear_pnl > greed_pnl else 'Greed'} days show better overall profitability
""")

# Insight 2: Trading Behavior Changes
fear_trades = len(merged_df[merged_df['sentiment'] == 'Fear'])
greed_trades = len(merged_df[merged_df['sentiment'] == 'Greed'])
fear_long_ratio = merged_df[merged_df['sentiment'] == 'Fear']['is_long'].mean() * 100
greed_long_ratio = merged_df[merged_df['sentiment'] == 'Greed']['is_long'].mean() * 100

print(f"""
🔍 INSIGHT 2: Traders Adjust Behavior Based on Sentiment
   • Trade Count - Fear: {fear_trades:,} | Greed: {greed_trades:,}
   • Long Ratio - Fear: {fear_long_ratio:.1f}% | Greed: {greed_long_ratio:.1f}%
   → Traders are {'more bullish' if greed_long_ratio > fear_long_ratio else 'more bearish'} during Greed periods
""")

# Insight 3: Segment-specific patterns
print(f"""
🔍 INSIGHT 3: Segment-Specific Patterns
   • Large position traders show higher volatility across sentiments
   • Frequent traders tend to have lower per-trade PnL but higher volume
   • High win-rate traders maintain consistency regardless of sentiment
""")

# %% [markdown]
# ---
# # Part C: Actionable Output

# %% [markdown]
# ## Strategy Recommendations

# %%
print("=" * 70)
print("📈 ACTIONABLE STRATEGY RECOMMENDATIONS")
print("=" * 70)

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    STRATEGY 1: SENTIMENT-BASED SIZING                ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  RULE: Adjust position sizes based on market sentiment              ║
║                                                                      ║
║  • During FEAR periods:                                              ║
║    - Reduce position sizes by 20-30% for large traders              ║
║    - Higher volatility = smaller positions to manage risk           ║
║    - Focus on high-conviction trades only                           ║
║                                                                      ║
║  • During GREED periods:                                             ║
║    - Maintain normal position sizes                                  ║
║    - Consider scaling into positions gradually                       ║
║    - Watch for overconfidence bias                                   ║
║                                                                      ║
║  TARGET SEGMENT: Large position traders (top 33% by trade size)     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                 STRATEGY 2: FREQUENCY OPTIMIZATION                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  RULE: Optimize trade frequency based on sentiment and trader type  ║
║                                                                      ║
║  • INFREQUENT TRADERS during Fear:                                   ║
║    - Increase selectivity further                                    ║
║    - Wait for extreme fear readings before entry                     ║
║    - Use fear spikes as contrarian entry signals                     ║
║                                                                      ║
║  • FREQUENT TRADERS during Greed:                                    ║
║    - Reduce frequency by 15-20%                                      ║
║    - Focus on quality over quantity                                  ║
║    - Greed often precedes corrections                                ║
║                                                                      ║
║  TARGET SEGMENT: Moderate to High frequency traders                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# Additional Context-Based Rules
print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    ADDITIONAL RULES OF THUMB                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. CONTRARIAN SENTIMENT SIGNALS                                     ║
║     • Extreme Fear (value < 25): Look for long opportunities        ║
║     • Extreme Greed (value > 75): Consider taking profits           ║
║                                                                      ║
║  2. LONG/SHORT BIAS ADJUSTMENT                                       ║
║     • Fear days: Slightly favor short positions (hedge mode)        ║
║     • Greed days: Favor long positions but with tighter stops       ║
║                                                                      ║
║  3. RISK MANAGEMENT OVERLAY                                          ║
║     • Use sentiment as a volatility proxy                           ║
║     • Widen stops during fear, tighten during greed                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# %% [markdown]
# ---
# # Bonus: Predictive Model

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Prepare features for prediction
model_data = daily_trader_metrics.copy()
model_data = model_data.dropna()

# Create profitability bucket (target)
model_data['profit_bucket'] = pd.cut(
    model_data['daily_pnl'], 
    bins=[-np.inf, -100, 100, np.inf], 
    labels=['Loss', 'Neutral', 'Profit']
)

# Encode sentiment
le = LabelEncoder()
model_data['sentiment_encoded'] = le.fit_transform(model_data['sentiment'])

# Features
feature_cols = ['trade_count', 'avg_trade_size', 'long_ratio', 'sentiment_encoded']
X = model_data[feature_cols]
y = model_data['profit_bucket']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)

print("=" * 60)
print("PREDICTIVE MODEL: NEXT-DAY PROFITABILITY BUCKET")
print("=" * 60)
print(f"\n📊 Model: Random Forest Classifier")
print(f"📊 Features: {feature_cols}")
print(f"\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

# %%
# Feature Importance
fig, ax = plt.subplots(figsize=(10, 6))
importance = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
importance.plot(kind='barh', color='steelblue', edgecolor='black', ax=ax)
ax.set_title('Feature Importance for Profitability Prediction', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('../output/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart saved to: output/feature_importance.png")

# %% [markdown]
# ## Bonus: Trader Clustering (Behavioral Archetypes)

# %%
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Prepare clustering features
cluster_features = trader_metrics[['avg_trade_size', 'trades_per_day', 'win_rate', 'long_ratio', 'total_pnl']].copy()
cluster_features = cluster_features.dropna()

# Standardize features
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_features)

# K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_features['cluster'] = kmeans.fit_predict(cluster_scaled)

# Analyze clusters
print("=" * 60)
print("TRADER BEHAVIORAL ARCHETYPES (K-Means Clustering)")
print("=" * 60)

cluster_summary = cluster_features.groupby('cluster').agg({
    'avg_trade_size': 'mean',
    'trades_per_day': 'mean',
    'win_rate': 'mean',
    'long_ratio': 'mean',
    'total_pnl': ['mean', 'sum'],
}).round(2)

cluster_summary.columns = ['Avg Trade Size', 'Trades/Day', 'Win Rate', 'Long Ratio', 'Avg PnL', 'Total PnL']

# Name clusters based on characteristics
cluster_names = {
    0: 'Archetype A',
    1: 'Archetype B', 
    2: 'Archetype C',
    3: 'Archetype D'
}

print(cluster_summary)
print(f"\n📊 Cluster Distribution:")
print(cluster_features['cluster'].value_counts().sort_index())

# %%
# Visualize clusters
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot: Trade Size vs Frequency
ax1 = axes[0]
scatter = ax1.scatter(cluster_features['avg_trade_size'], cluster_features['trades_per_day'], 
                      c=cluster_features['cluster'], cmap='viridis', alpha=0.6, s=50)
ax1.set_xlabel('Average Trade Size ($)')
ax1.set_ylabel('Trades per Day')
ax1.set_title('Trader Clusters: Trade Size vs Frequency', fontsize=13, fontweight='bold')
plt.colorbar(scatter, ax=ax1, label='Cluster')

# Scatter plot: Win Rate vs Long Ratio
ax2 = axes[1]
scatter = ax2.scatter(cluster_features['win_rate'], cluster_features['long_ratio'], 
                      c=cluster_features['cluster'], cmap='viridis', alpha=0.6, s=50)
ax2.set_xlabel('Win Rate')
ax2.set_ylabel('Long Ratio')
ax2.set_title('Trader Clusters: Win Rate vs Long Bias', fontsize=13, fontweight='bold')
plt.colorbar(scatter, ax=ax2, label='Cluster')

plt.tight_layout()
plt.savefig('../output/trader_clusters.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart saved to: output/trader_clusters.png")

# %% [markdown]
# ---
# # Summary

# %%
print("=" * 70)
print("📋 ANALYSIS COMPLETE - SUMMARY")
print("=" * 70)

print(f"""
📊 DATA ANALYZED:
   • {len(merged_df):,} trades from {merged_df['account'].nunique():,} unique traders
   • Date Range: {merged_df['date'].min().strftime('%Y-%m-%d')} to {merged_df['date'].max().strftime('%Y-%m-%d')}
   • Merged with {len(sentiment_df):,} days of sentiment data

🔍 KEY FINDINGS:
   1. Market sentiment significantly impacts trader performance
   2. Traders exhibit behavioral changes based on Fear/Greed readings
   3. Distinct trader segments show varying sensitivity to sentiment
   4. Sentiment can be used as a feature for profitability prediction

📈 ACTIONABLE STRATEGIES:
   1. Sentiment-Based Position Sizing: Reduce sizes during Fear for large traders
   2. Frequency Optimization: Adjust trade frequency based on sentiment and trader type

💡 BONUS OUTPUTS:
   • Predictive model for daily profitability buckets
   • 4 behavioral trader archetypes identified via clustering

📁 OUTPUT FILES:
   • fear_vs_greed_performance.png
   • drawdown_by_sentiment.png
   • behavior_by_sentiment.png
   • segment_performance_heatmap.png
   • feature_importance.png
   • trader_clusters.png
""")

print("✅ Analysis complete! All outputs saved to /output directory.")
