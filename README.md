# Market Sentiment Analysis: Hyperliquid Trader Behavior

Analyze how market sentiment (Fear/Greed) relates to trader behavior and performance on Hyperliquid.

## 📁 Project Structure

```
.
├── data/
│   ├── bitcoin_sentiment.csv      # Fear/Greed index data
│   └── hyperliquid_trader_data.csv # Historical trader data
├── notebooks/
│   ├── analysis_notebook.py       # Source script
│   ├── analysis_notebook.ipynb    # Jupyter notebook
│   └── analysis_notebook_executed.ipynb  # Executed with outputs
├── output/
│   ├── fear_vs_greed_performance.png
│   ├── drawdown_by_sentiment.png
│   ├── behavior_by_sentiment.png
│   ├── segment_performance_heatmap.png
│   ├── feature_importance.png
│   └── trader_clusters.png
└── README.md
```

## 🚀 Setup & Installation

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupytext streamlit

# Run the analysis
cd notebooks
python analysis_notebook.py

# Or run as Jupyter notebook
jupyter notebook analysis_notebook.ipynb

# Run interactive dashboard
streamlit run dashboard.py
```

## 📊 Analysis Overview

### Part A: Data Preparation
- Loaded 2,645 days of sentiment data + 211,225 trades
- Aligned datasets by date with timestamp conversion
- Created metrics: daily PnL, win rate, trade size, long/short ratio

### Part B: Analysis
- **Fear vs Greed Performance**: Compared PnL, win rates, and drawdowns
- **Behavioral Changes**: Trade frequency, position sizing, directional bias
- **Trader Segments**: Size-based, frequency-based, consistency-based

### Part C: Actionable Strategies
1. **Sentiment-Based Sizing**: Reduce positions during Fear for large traders
2. **Frequency Optimization**: Adjust trade count based on sentiment

### Bonus
- Random Forest model for profitability prediction
- K-Means clustering for trader archetypes

## 📈 Key Findings

| Metric | Fear Days | Greed Days |
|--------|-----------|------------|
| Avg PnL/Trade | Variable | Variable |
| Win Rate | ~30-40% | ~30-40% |
| Long Ratio | Lower | Higher |

## 📧 Contact

Created for Data Science Intern assessment.
