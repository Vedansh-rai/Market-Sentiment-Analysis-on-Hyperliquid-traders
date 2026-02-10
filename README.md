# Market Sentiment Analysis on Hyperliquid Traders

## 📂 Project Structure 

This repository follows the structure:

```
ds_<Vedansh>/
├── notebook_1.ipynb               # Google Colab Notebook (Analysis & Code)
├── csv_files/                     # Processed Data Files
│   ├── bitcoin_sentiment.csv
│   ├── hyperliquid_trader_data.csv
│   ├── processed_merged_data.csv
│   ├── processed_daily_metrics.csv
│   └── ...
├── outputs/                       # Visual Outputs (Charts/Graphs)
│   ├── fear_vs_greed_performance.png
│   ├── drawdown_by_sentiment.png
│   ├── behavior_by_sentiment.png
│   ├── segment_performance_heatmap.png
│   ├── feature_importance.png
│   └── trader_clusters.png
├── ds_report.pdf                  # Final Summarized Insights Report
└── README.md                      # Setup Instructions & Notes
```

## 🔹 Google Colab Link

**[View Analysis Notebook in Google Colab](https://colab.research.google.com/drive/1mC9SGgvRVYUi3UuFJRQR_kNQk-npaF8i)**  
*(Access set to 'Anyone with the link can view')*

## 🚀 Setup & Usage

1. **Data:** Raw and processed data is available in `csv_files/`.
2. **Analysis:** The core analysis is in `notebook_1.ipynb`.
3. **Report:** Read the full findings in `ds_report.pdf`.
4. **Processing:** Use `process_data.py` (in `extras/` or root if moved) to reproduce the processed CSVs.

## 📊 Key Findings

<<<<<<< HEAD
# Analysis notebook
cd notebooks && python3 analysis_notebook.py

# Interactive dashboard
streamlit run dashboard.py



=======
- **Sentiment Impact:** Market sentiment (Fear/Greed) significantly impacts trader profitability.
- **Behavioral Shifts:** Traders adjust frequency and leverage based on sentiment.
- **Strategies:** Recommended creating sentiment-adjusted sizing rules for large traders.

---
*Submitted by Vedansh Rai*
>>>>>>> c2207f4 (Initial commit of structured project)
