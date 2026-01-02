"""
Time Series Feature Engineering for IDS
Creates time-series specific features and sequences for deep learning
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, TIMESERIES_DIR, SEQUENCE_LENGTH, TIME_WINDOW

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample size for time series features (memory optimization)
SAMPLE_SIZE = 200000  # 200K samples for time series processing

class TimeSeriesFeatureEngineer:
    def __init__(self):
        """Initialize time series feature engineer"""
        self.sequence_scaler = MinMaxScaler()
        
        # Ensure output directories exist
        os.makedirs(TIMESERIES_DIR, exist_ok=True)
        
        logger.info("Time Series Feature Engineer initialized")
    
    def load_processed_data(self, filename="cleaned_features.parquet", sample_size=SAMPLE_SIZE):
        """Load processed data with optional sampling for memory efficiency"""
        data_path = os.path.join(PROCESSED_DIR, filename)
        logger.info(f"Loading processed data from: {data_path}")
        
        df = pd.read_parquet(data_path)
        logger.info(f"Full data shape: {df.shape}")
        
        # Sample data for memory efficiency
        if sample_size and len(df) > sample_size:
            # Stratified sampling to maintain attack/benign ratio
            if 'is_attack' in df.columns:
                attack_ratio = df['is_attack'].mean()
                n_attack = int(sample_size * attack_ratio)
                n_benign = sample_size - n_attack
                
                attack_samples = df[df['is_attack'] == 1].sample(n=min(n_attack, len(df[df['is_attack'] == 1])), random_state=42)
                benign_samples = df[df['is_attack'] == 0].sample(n=min(n_benign, len(df[df['is_attack'] == 0])), random_state=42)
                df = pd.concat([attack_samples, benign_samples]).sort_index()
            else:
                df = df.sample(n=sample_size, random_state=42)
            
            logger.info(f"Sampled data shape: {df.shape}")
        
        return df
    
    def add_temporal_features(self, df):
        """Add temporal features based on row order"""
        logger.info("Adding temporal features...")
        
        # Create timestamp based on row order (simulating time sequence)
        df = df.copy()
        df['timestamp'] = pd.to_datetime('2018-02-14') + pd.to_timedelta(df.index, unit='s')
        
        # Extract time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Time-based cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        logger.info("Temporal features added")
        return df
    
    def create_rolling_statistics(self, df, windows=[5, 10]):
        """Create rolling window statistics (optimized for memory)"""
        logger.info("Creating rolling window statistics...")
        
        # Select numeric features (excluding labels and temporal features)
        exclude_cols = ['is_attack', 'timestamp', 'hour', 'day_of_week', 'is_weekend', 'is_business_hours', 
                        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'Label_encoded']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        # Use only top 5 features for rolling statistics (memory optimization)
        key_features = feature_cols[:5]
        
        # Build new columns in a list, then concat at once (avoids fragmentation)
        new_cols = {}
        for feature in key_features:
            for window in windows:
                new_cols[f'{feature}_rolling_mean_{window}'] = df[feature].rolling(window=window, min_periods=1).mean()
                new_cols[f'{feature}_rolling_std_{window}'] = df[feature].rolling(window=window, min_periods=1).std()
        
        # Concat all new columns at once
        new_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
        # Fill any remaining NaN values
        new_df = new_df.ffill().bfill()
        
        logger.info(f"Rolling statistics created. New shape: {new_df.shape}")
        return new_df
    
    def create_lag_features(self, df, lags=[1, 3, 5]):
        """Create lag features for time series analysis (optimized)"""
        logger.info("Creating lag features...")
        
        # Select key features for lag creation (limit to 5 features)
        exclude_cols = ['is_attack', 'timestamp', 'Label_encoded']
        feature_cols = [col for col in df.columns if col not in exclude_cols and 'rolling' not in col 
                        and df[col].dtype in ['int64', 'float64']]
        key_features = feature_cols[:5]  # Reduced to 5 features
        
        # Build new columns in a dict, then concat
        new_cols = {}
        for feature in key_features:
            for lag in lags:
                new_cols[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        # Concat all new columns at once
        new_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
        # Fill NaN values created by lagging
        new_df = new_df.bfill().fillna(0)
        
        logger.info(f"Lag features created. New shape: {new_df.shape}")
        return new_df
    
    def create_difference_features(self, df):
        """Create difference features to capture changes (optimized)"""
        logger.info("Creating difference features...")
        
        # Select key numeric features (limit to 5)
        exclude_cols = ['is_attack', 'timestamp', 'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
                        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'Label_encoded']
        numeric_cols = [col for col in df.columns if col not in exclude_cols and 'lag' not in col 
                        and 'rolling' not in col and df[col].dtype in ['int64', 'float64']]
        key_features = numeric_cols[:5]  # Reduced to 5
        
        # Build new columns in dict
        new_cols = {}
        for feature in key_features:
            # First difference only (reduced from 3 calculations)
            new_cols[f'{feature}_diff_1'] = df[feature].diff()
        
        # Concat all new columns at once
        new_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
        # Fill NaN values
        new_df = new_df.fillna(0)
        
        logger.info(f"Difference features created. New shape: {new_df.shape}")
        return new_df
    
    def detect_anomalies_statistical(self, df):
        """Statistical anomaly detection for additional features (optimized)"""
        logger.info("Creating statistical anomaly features...")
        
        # Select key features for anomaly detection (limit to 3 features)
        exclude_cols = ['is_attack', 'timestamp', 'Label_encoded']
        feature_cols = [col for col in df.columns if col not in exclude_cols and 
                       not any(x in col for x in ['lag', 'rolling', 'diff', 'pct_change', 'sin', 'cos'])
                       and df[col].dtype in ['int64', 'float64']]
        
        key_features = feature_cols[:3]  # Reduced to 3 features
        
        # Build new columns in dict
        new_cols = {}
        for feature in key_features:
            # Z-score based anomaly detection
            feature_data = df[feature].values
            mean_val = np.mean(feature_data)
            std_val = np.std(feature_data)
            if std_val > 0:
                z_scores = np.abs((feature_data - mean_val) / std_val)
                new_cols[f'{feature}_zscore_anomaly'] = (z_scores > 3).astype(np.int8)
        
        # Concat all new columns at once
        new_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
        # Create composite anomaly score
        anomaly_cols = [col for col in new_df.columns if 'zscore_anomaly' in col]
        if anomaly_cols:
            new_df['total_anomaly_score'] = new_df[anomaly_cols].sum(axis=1).astype(np.int8)
            new_df['is_statistical_anomaly'] = (new_df['total_anomaly_score'] > 0).astype(np.int8)
        
        logger.info("Statistical anomaly features created")
        return new_df
    
    def create_sequences_for_lstm(self, df, sequence_length=SEQUENCE_LENGTH):
        """Create sequences for LSTM training"""
        logger.info(f"Creating LSTM sequences with length: {sequence_length}")
        
        # Select only numeric features for LSTM (exclude categorical, object, and target variables)
        exclude_cols = ['is_attack', 'timestamp', 'is_weekend', 'is_business_hours', 'source_file', 
                        'Label', 'Label_encoded', 'total_anomaly_score', 'is_statistical_anomaly']
        
        # Get only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols and 'anomaly' not in col]
        
        logger.info(f"Using {len(feature_cols)} features for LSTM sequences")
        
        # Sort by timestamp to ensure proper sequence order
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        else:
            df_sorted = df.reset_index(drop=True)
        
        # Prepare data for sequences
        features = df_sorted[feature_cols].values.astype(np.float32)
        labels = df_sorted['is_attack'].values.astype(np.int8)
        
        # Scale features for LSTM
        features_scaled = self.sequence_scaler.fit_transform(features)
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(features_scaled)):
            X_sequences.append(features_scaled[i-sequence_length:i])
            y_sequences.append(labels[i])
        
        X_sequences = np.array(X_sequences, dtype=np.float32)
        y_sequences = np.array(y_sequences, dtype=np.int8)
        
        logger.info(f"Created {len(X_sequences)} sequences with shape: {X_sequences.shape}")
        
        return X_sequences, y_sequences, feature_cols
    
    def create_time_windows(self, df, window_size_minutes=5):
        """Create time-based windows for aggregation"""
        logger.info(f"Creating time windows of {window_size_minutes} minutes...")
        
        # Set timestamp as index
        df_time = df.set_index('timestamp')
        
        # Resample to create time windows
        agg_functions = {
            'is_attack': 'max',  # If any attack in window, mark as attack
        }
        
        # Add aggregation functions for numeric features
        numeric_cols = df_time.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'is_attack'][:20]  # Limit features
        
        for col in numeric_cols:
            agg_functions[col + '_mean'] = (col, 'mean')
            agg_functions[col + '_std'] = (col, 'std')
            agg_functions[col + '_max'] = (col, 'max')
            agg_functions[col + '_min'] = (col, 'min')
        
        # Create time windows
        windowed_df = df_time.resample(f'{window_size_minutes}T').agg(agg_functions)
        windowed_df = windowed_df.fillna(0)
        
        # Flatten column names
        windowed_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                              for col in windowed_df.columns.values]
        
        # Reset index to get timestamp as column
        windowed_df = windowed_df.reset_index()
        
        logger.info(f"Created {len(windowed_df)} time windows. Shape: {windowed_df.shape}")
        
        return windowed_df
    
    def save_time_series_features(self, df, X_sequences=None, y_sequences=None, feature_cols=None):
        """Save time series features and sequences"""
        logger.info("Saving time series features...")
        
        # Save enhanced features
        features_path = os.path.join(TIMESERIES_DIR, "time_series_features.parquet")
        df.to_parquet(features_path, index=False)
        logger.info(f"Time series features saved to: {features_path}")
        
        # Save LSTM sequences if provided
        if X_sequences is not None and y_sequences is not None:
            sequences_path = os.path.join(TIMESERIES_DIR, "lstm_sequences.npz")
            np.savez_compressed(sequences_path, 
                               X=X_sequences, 
                               y=y_sequences, 
                               feature_names=feature_cols)
            logger.info(f"LSTM sequences saved to: {sequences_path}")
            
            # Save sequence scaler
            import joblib
            scaler_path = os.path.join(TIMESERIES_DIR, "sequence_scaler.pkl")
            joblib.dump(self.sequence_scaler, scaler_path)
            logger.info(f"Sequence scaler saved to: {scaler_path}")
    
    def generate_feature_summary(self, df):
        """Generate summary of created features"""
        logger.info("=== TIME SERIES FEATURE SUMMARY ===")
        
        print(f"Total features: {len(df.columns)}")
        print(f"Dataset shape: {df.shape}")
        
        # Feature categories
        categories = {
            'Original': [col for col in df.columns if not any(x in col for x in 
                        ['rolling', 'lag', 'diff', 'pct_change', 'anomaly', 'sin', 'cos'])],
            'Rolling Stats': [col for col in df.columns if 'rolling' in col],
            'Lag Features': [col for col in df.columns if 'lag' in col],
            'Difference': [col for col in df.columns if 'diff' in col or 'pct_change' in col],
            'Anomaly': [col for col in df.columns if 'anomaly' in col],
            'Temporal': [col for col in df.columns if any(x in col for x in ['sin', 'cos', 'hour', 'day'])]
        }
        
        print("\nFeature Categories:")
        for category, features in categories.items():
            print(f"  {category}: {len(features)} features")
        
        # Attack distribution
        if 'is_attack' in df.columns:
            attack_dist = df['is_attack'].value_counts()
            print(f"\nAttack Distribution:")
            print(f"  Benign: {attack_dist.get(0, 0)} ({attack_dist.get(0, 0)/len(df)*100:.2f}%)")
            print(f"  Attack: {attack_dist.get(1, 0)} ({attack_dist.get(1, 0)/len(df)*100:.2f}%)")

def main():
    """Main time series feature engineering pipeline"""
    import gc
    
    try:
        # Initialize feature engineer
        ts_engineer = TimeSeriesFeatureEngineer()
        
        logger.info("=== STARTING TIME SERIES FEATURE ENGINEERING ===")
        
        # Step 1: Load processed data
        logger.info("=== STEP 1: LOADING PROCESSED DATA ===")
        df = ts_engineer.load_processed_data()
        gc.collect()
        
        # Step 2: Add temporal features
        logger.info("=== STEP 2: ADDING TEMPORAL FEATURES ===")
        df = ts_engineer.add_temporal_features(df)
        gc.collect()
        
        # Step 3: Create rolling statistics
        logger.info("=== STEP 3: CREATING ROLLING STATISTICS ===")
        df = ts_engineer.create_rolling_statistics(df)
        gc.collect()
        
        # Step 4: Create lag features
        logger.info("=== STEP 4: CREATING LAG FEATURES ===")
        df = ts_engineer.create_lag_features(df)
        gc.collect()
        
        # Step 5: Create difference features
        logger.info("=== STEP 5: CREATING DIFFERENCE FEATURES ===")
        df = ts_engineer.create_difference_features(df)
        gc.collect()
        
        # Step 6: Statistical anomaly detection
        logger.info("=== STEP 6: STATISTICAL ANOMALY DETECTION ===")
        df = ts_engineer.detect_anomalies_statistical(df)
        gc.collect()
        
        # Step 7: Save intermediate time series features first
        logger.info("=== STEP 7: SAVING INTERMEDIATE FEATURES ===")
        intermediate_path = os.path.join(TIMESERIES_DIR, "time_series_features.parquet")
        df.to_parquet(intermediate_path, index=False)
        logger.info(f"Time series features saved to: {intermediate_path}")
        gc.collect()
        
        # Step 8: Create LSTM sequences (use subset for memory efficiency)
        logger.info("=== STEP 8: CREATING LSTM SEQUENCES ===")
        # Use only 100K samples for LSTM sequences due to memory constraints
        sample_size = min(100000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42).sort_index()
        X_sequences, y_sequences, feature_cols = ts_engineer.create_sequences_for_lstm(df_sample)
        del df_sample
        gc.collect()
        
        # Save sequences
        logger.info("=== STEP 9: SAVING LSTM SEQUENCES ===")
        ts_engineer.save_time_series_features(df, X_sequences, y_sequences, feature_cols)
        
        # Step 10: Generate summary
        logger.info("=== STEP 10: FEATURE SUMMARY ===")
        ts_engineer.generate_feature_summary(df)
        
        logger.info("=== TIME SERIES FEATURE ENGINEERING COMPLETED ===")
        
        return df, X_sequences, y_sequences
        
    except Exception as e:
        logger.error(f"Error in time series feature engineering: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()