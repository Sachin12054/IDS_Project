# Time Series Models Comparison Report
## IDS Attack Prediction

Generated: 2025-12-30 15:19:57

---

## Models Implemented

### 1. LINEAR MODEL: SARIMA
- **Model**: SARIMA(1,1,1)(1,0,1,24)
- **Type**: Statistical Time Series
- **RMSE**: 992.4197384002548
- **Best For**: Capturing seasonal patterns (24-hour cycles)

### 2. NON-LINEAR MODEL: XGBoost
- **Model**: XGBoost Regressor (100 trees)
- **Type**: Gradient Boosting
- **RMSE**: 621.7220289245025
- **R² Score**: 0.6006768941879272
- **Best For**: Complex non-linear relationships

### 3. DEEP LEARNING: LSTM
- **Model**: 2-Layer LSTM (64 hidden units)
- **Type**: Recurrent Neural Network
- **RMSE**: 590.564824976903
- **Best For**: Sequential pattern recognition

---

## Results Summary

| Model | Type | RMSE | MAE |
|-------|------|------|-----|
| SARIMA | Linear | 992.4197 | 651.6987 |
| XGBoost | Non-Linear | 621.7220 | 269.4376 |
| LSTM | Deep Learning | 590.5648 | 255.1005 |

---

## Best Model: LSTM

The LSTM model achieved the lowest RMSE of 590.5648

---

## Files Generated

- `model_comparison.png` - Forecast comparison plots
- `model_metrics_comparison.png` - Performance metrics bar chart
- `time_series_models_report.md` - This report
