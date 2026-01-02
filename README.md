# 🔐 Time Series Intrusion Detection System

## Project Overview
Intelligent Intrusion Detection System using Time-Series Deep Learning and Machine Learning on CSE-CIC-IDS2018 dataset for computer security analysis.

---

## 📚 TIME SERIES MODELS COMPARISON

### Efficient 3-Model Approach ✅

| Model Type | Model | RMSE | MAE | File |
|------------|-------|------|-----|------|
| **LINEAR** | SARIMA(1,1,1)(1,0,1,24) | 992.42 | 651.70 | `time_series_models.py` |
| **NON-LINEAR** | XGBoost (100 trees, R²=0.60) | 621.72 | 269.44 | `time_series_models.py` |
| **DEEP LEARNING** | LSTM (2-layer, 64 hidden) 🏆 | **590.56** | **255.10** | `time_series_models.py` |

### Why These Models?

| Category | Model | Reason |
|----------|-------|--------|
| Linear | **SARIMA** | Best for seasonal patterns (24-hour cycles in network traffic) |
| Non-Linear | **XGBoost** | Captures complex relationships, handles lag features efficiently |
| Deep Learning | **LSTM** | Sequential pattern recognition, memory of past attack patterns |

### Key Features Used
- Lag features (1-24 hours)
- Rolling statistics (6hr, 24hr windows)
- Time features (hour, day_of_week, is_weekend)
- Seasonal decomposition

### Syllabus Coverage
| Unit | Topics Covered |
|------|---------------|
| Unit 1 | Stationarity (ADF test), ACF/PACF analysis |
| Unit 2 | SARIMA model with seasonal components |
| Unit 3 | Non-linear patterns via XGBoost |
| Unit 4 | Deep Learning via LSTM neural network |

---

## 📋 PROJECT TODO CHECKLIST

### Phase 1: Environment Setup ✅ COMPLETED
- [x] Create Python virtual environment
- [x] Install all dependencies from requirements.txt
- [x] Verify GPU availability (using CPU - PyTorch)

### Phase 2: Data Preprocessing ✅ COMPLETED
- [x] Load and merge CSV files from raw_csv/ (10% sample = 1.6M rows)
- [x] Clean column names and handle missing values
- [x] Remove duplicates and infinite values
- [x] Encode labels (attack types)
- [x] Feature selection and scaling (71 features)
- [x] Save processed data to parquet format

### Phase 3: Time Series Feature Engineering ✅ COMPLETED
- [x] Add temporal features (hour, day, cyclical encoding)
- [x] Create rolling window statistics (5, 10)
- [x] Generate lag features for time series (1, 3, 5)
- [x] Create difference features (velocity of change)
- [x] Statistical anomaly detection features
- [x] Create LSTM sequences (length=30) - 99,970 sequences
- [x] Save time series features and sequences

### Phase 4: Model Training ✅ COMPLETED
- [x] Train Random Forest classifier - **100% Accuracy, AUC 1.0**
- [x] Train XGBoost classifier - **100% Accuracy, AUC 1.0**
- [x] Train Isolation Forest (anomaly detection)
- [x] Train LSTM classifier - **94.33% Accuracy, AUC 0.9858**
- [ ] Train GRU model (optional)
- [x] Save all trained models

### Phase 5: Model Evaluation ✅ COMPLETED
- [x] Evaluate all models on test data
- [x] Generate confusion matrices
- [x] Plot ROC curves
- [x] Plot Precision-Recall curves
- [x] Create evaluation report
- [x] Compare model performances

### Phase 6: Dashboard & Visualization ✅ COMPLETED
- [x] Test dashboard with sample data
- [x] Connect dashboard to real predictions
- [x] Verify all visualizations work

### Phase 7: Documentation & Final Testing ✅ COMPLETED
- [x] Run complete pipeline end-to-end
- [x] Document results and findings
- [x] Prepare presentation materials

---

## 🏆 MODEL RESULTS SUMMARY (Final Evaluation - No Data Leakage)

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **98.64%** | 0.9630 | 0.9518 | 0.9574 | 0.9958 |
| **XGBoost** | **98.68%** | 0.9633 | 0.9536 | 0.9584 | 0.9961 |
| LSTM | 94.06% | 0.8290 | 0.7936 | 0.8109 | 0.9853 |

> **Note:** Initial results showed 100% accuracy due to data leakage (`Label_encoded` feature). 
> After removing leaky features, realistic ~99% accuracy achieved - indicating a properly trained model.

---

## System Requirements
- **CPU**: i5-12500H or better
- **RAM**: 8GB recommended (4GB minimum)
- **GPU**: RTX 3050 or better (4GB VRAM) - Optional for deep learning
- **Storage**: 20GB+ available space

## Installation

### 1. Environment Setup
```bash
# Create virtual environment
conda create -n ids_timeseries python=3.10
conda activate ids_timeseries

# Install requirements
pip install -r requirements.txt
```

### 2. GPU Setup (Optional)
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Project Structure
```
IDS_TIMESERIES_SECURITY/
├── data/
│   ├── raw_csv/           # Original CSV files
│   ├── processed/         # Cleaned + scaled data
│   └── time_series/       # Time series datasets
├── preprocessing/         # Data preprocessing scripts
├── models/               # Trained ML/DL models
├── notebooks/            # Jupyter notebooks for analysis
├── dashboard/            # Web dashboard application
└── README.md
```

## Execution Pipeline

### Stage 1: Data Preprocessing
```bash
python preprocessing/data_cleaning.py
```
Output: `data/processed/cleaned_features.parquet` (1.6M rows, 75 columns)

### Stage 2: Time Series Feature Engineering
```bash
python preprocessing/time_series_features.py
```
Output: `data/time_series/time_series_features.parquet` (200K rows, 129 features)
Output: `data/time_series/lstm_sequences.npz` (99,970 sequences)

### Stage 3: Model Training
```bash
python models/train_models.py
```
Output: Trained models in `models/` directory

### Stage 4: Evaluation
```bash
python models/run_evaluation.py
```
Output: `evaluation_results/` with plots and reports

### Stage 5: Dashboard
```bash
python dashboard/app.py
```
Access: http://localhost:8050

---

## Dataset
- **Source**: CSE-CIC-IDS2018
- **Original Size**: ~16 million rows
- **Processed Size**: 1.6M rows (10% sample)
- **Format**: CSV files with flow-based features
- **Attack Types**: DoS, DDoS, Botnet, Brute Force, Web Attacks, Infiltration
- **Time Period**: February-March 2018

## Models Implemented

### Time Series Deep Learning
| Model | Architecture | Parameters |
|-------|-------------|------------|
| LSTM | Bidirectional, 64 hidden | ~400K |
| GRU | Bidirectional, 64 hidden | ~300K |

### Traditional ML Models
| Model | Key Parameters |
|-------|---------------|
| Random Forest | 100 trees, max_depth=20 |
| XGBoost | 100 estimators |
| Isolation Forest | 100 estimators |

## Final Results

### Performance Metrics
| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| **Random Forest** | **100%** | 1.00 | 1.00 | 1.00 | 1.00 |
| **XGBoost** | **100%** | 1.00 | 1.00 | 1.00 | 1.00 |
| LSTM | 94.06% | 0.83 | 0.79 | 0.81 | 0.99 |

### Key Findings
1. Traditional ML models (RF, XGBoost) achieve perfect classification on this dataset
2. LSTM achieves 94%+ accuracy with time-series patterns
3. Feature engineering significantly improves model performance
4. Rolling statistics and lag features are most predictive

---

## 📊 TIME SERIES ANALYSIS RESULTS

### Stationarity Analysis
| Test | Statistic | p-value | Result |
|------|-----------|---------|--------|
| ADF Test | -4.7518 | 0.0001 | **STATIONARY** |
| KPSS Test | 0.2263 | 0.1000 | **STATIONARY** |

### ARIMA Models Performance
| Model | AIC | BIC | Description |
|-------|-----|-----|-------------|
| AR(5) | 6615.69 | 6644.31 | Autoregressive with 5 lags |
| ARIMA(1,1,1) | 6703.22 | 6715.51 | Integrated model |
| SARIMA(1,1,1)(1,0,1,24) | 6273.39 | 6293.58 | **Best** - Seasonal patterns |

### GARCH Volatility Analysis
| Model | AIC | Result |
|-------|-----|--------|
| ARCH(1) | 6889.67 | Volatility clustering detected |
| GARCH(1,1) | 6889.39 | Time-varying variance modeled |

### Nonlinearity Tests
| Test | Statistic | p-value | Conclusion |
|------|-----------|---------|------------|
| Runs Test | -19.24 | 0.0000 | **Non-Random** patterns |
| McLeod-Li | 1217.99 | 0.0000 | **ARCH effects present** |
| Levene Test | 1.58 | 0.2092 | Variance stable |

---

## Output Files

```
evaluation_results/
├── confusion_matrices.png      # Confusion matrices for all models
├── roc_curves.png              # ROC curves comparison
├── precision_recall_curves.png # Precision-Recall curves
├── metrics_comparison.png      # Bar chart of all metrics
├── evaluation_report.csv       # Numeric results
├── evaluation_report.md        # Markdown summary
└── time_series_analysis/       # 📊 TIME SERIES ANALYSIS
    ├── attack_count_acf_pacf.png      # ACF/PACF plots
    ├── attack_count_arima.png         # ARIMA diagnostics
    ├── attack_count_sarima.png        # SARIMA forecast
    ├── attack_count_decomposition.png # Seasonal decomposition
    ├── traffic_volume_garch.png       # GARCH volatility
    ├── attack_forecast_comparison.png # Forecast comparison
    └── time_series_report.md          # Full analysis report

models/
├── random_forest.pkl           # Random Forest model
├── xgboost.pkl                 # XGBoost model
├── lstm_best.pth               # Best LSTM model (PyTorch)
├── isolation_forest.pkl        # Isolation Forest model
└── scaler_*.pkl                # Feature scalers

notebooks/
├── EDA.ipynb                   # Exploratory Data Analysis
└── time_series_analysis.py     # 📊 COMPREHENSIVE TS ANALYSIS

data/
├── processed/
│   └── cleaned_features.parquet
└── time_series/
    ├── time_series_features.parquet
    └── lstm_sequences.npz
```

---

## Authors
- Student Name
- University: Amrita Vishwa Vidyapeetham
- Course: Computer Security (Semester 6)

## License
Academic Project - Amrita University