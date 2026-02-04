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

# Analysis notebook
cd notebooks && python3 analysis_notebook.py

# Interactive dashboard
streamlit run dashboard.py



