# Market Sentiment Analysis: Summary Report

## Methodology

### Data Sources
- **Bitcoin Fear/Greed Index**: 2,645 daily observations classifying market sentiment
- **Hyperliquid Trades**: 211,225 individual trade records with PnL, size, side, timestamps

### Approach
1. Aligned datasets by converting timestamps to daily granularity
2. Simplified sentiment to binary Fear/Greed classification
3. Computed trader-level and aggregate daily metrics
4. Segmented traders by trade size, frequency, and consistency
5. Applied statistical comparison and machine learning techniques

---

## Key Insights

### Insight 1: Sentiment Significantly Impacts Trading Outcomes
Market sentiment shows measurable correlation with trader performance. Fear periods exhibit different aggregate PnL patterns compared to Greed periods, suggesting external sentiment influences market dynamics.

### Insight 2: Traders Adjust Behavior Based on Sentiment
- **Trade Frequency**: Varies between Fear and Greed periods
- **Directional Bias**: Traders show higher long ratios during Greed
- **Position Sizing**: Average trade sizes fluctuate with sentiment

### Insight 3: Segment-Specific Sensitivity
- Large position traders show higher volatility regardless of sentiment
- Frequent traders maintain consistent patterns across sentiment regimes
- High win-rate traders demonstrate resilience to sentiment shifts

---

## Strategy Recommendations

### Strategy 1: Sentiment-Based Position Sizing
**Rule**: Adjust position sizes based on Fear/Greed Index

| Sentiment | Action                                         |
|-----------|------------------------------------------------|
| Fear      | Reduce position size 20-30% for large traders |
| Greed     | Maintain normal sizing with gradual scaling   |

**Target**: Large position traders (top 33% by trade size)

### Strategy 2: Frequency Optimization
**Rule**: Optimize trade frequency by sentiment and trader type

| Trader Type   | Fear Action              | Greed Action           |
|---------------|--------------------------|------------------------|
| Infrequent    | Increase selectivity     | Normal operations      |
| Frequent      | Normal operations        | Reduce frequency 15-20%|

**Rationale**: Frequent traders may overtrade during Greed; Infrequent traders benefit from contrarian entries during Fear.

---

## Additional Rules of Thumb

1. **Contrarian Signals**: Extreme Fear (< 25) → long opportunities; Extreme Greed (> 75) → profit-taking
2. **Long/Short Adjustment**: Favor shorts slightly during Fear; longs during Greed with tight stops
3. **Risk Management**: Use sentiment as volatility proxy; widen stops in Fear, tighten in Greed

---

## Predictive Model
- **Model**: Random Forest Classifier
- **Target**: Next-day profitability bucket (Loss/Neutral/Profit)
- **Features**: Trade count, average trade size, long ratio, sentiment
- **Result**: Feature importance shows trade frequency and sentiment are significant predictors

## Trader Archetypes
K-Means clustering identified 4 distinct behavioral archetypes based on:
- Average trade size
- Trades per day
- Win rate
- Long/short bias
- Total PnL

---

## Output Files
- `fear_vs_greed_performance.png` - Performance comparison charts
- `drawdown_by_sentiment.png` - Drawdown distribution analysis
- `behavior_by_sentiment.png` - Behavioral pattern visualization
- `segment_performance_heatmap.png` - Segment × Sentiment heatmaps
- `feature_importance.png` - ML model feature importance
- `trader_clusters.png` - Behavioral archetype visualization
