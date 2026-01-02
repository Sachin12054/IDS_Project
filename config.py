# Configuration for IDS Time Series Project

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV_DIR = DATA_DIR / "raw_csv"
PROCESSED_DIR = DATA_DIR / "processed"
TIMESERIES_DIR = DATA_DIR / "time_series"
MODELS_DIR = PROJECT_ROOT / "models"
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"

# Data Processing Parameters
CHUNK_SIZE = 10000
BATCH_SIZE = 64
TIME_WINDOW = "5 minutes"  # For time series aggregation
SEQUENCE_LENGTH = 30  # For LSTM sequences

# Model Parameters
LSTM_CONFIG = {
    "sequence_length": 30,
    "hidden_units": 128,
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "epochs": 50
}

AUTOENCODER_CONFIG = {
    "encoding_dim": 32,
    "input_dim": None,  # Will be set based on features
    "epochs": 100,
    "batch_size": 32
}

# Attack Label Mapping
ATTACK_LABELS = {
    "BENIGN": 0,
    "DoS Hulk": 1,
    "DoS GoldenEye": 2,
    "DoS Slowloris": 3,
    "DoS slowhttptest": 4,
    "DDoS LOIC-HTTP": 5,
    "DDoS LOIC-UDP": 6,
    "DDoS HOIC": 7,
    "Brute Force -Web": 8,
    "Brute Force -XSS": 9,
    "SQL Injection": 10,
    "Infiltration": 11,
    "Bot": 12
}

# Feature Columns (will be updated after data exploration)
FEATURE_COLUMNS = [
    'Dst Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Tot Fwd Pkts',
    'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
    'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max',
    'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s',
    'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min'
]

# GPU Configuration
GPU_CONFIG = {
    "use_gpu": True,
    "memory_growth": True,
    "mixed_precision": True
}