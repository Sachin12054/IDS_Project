"""
Model Training Pipeline for Time Series IDS (PyTorch Version)
Implements Random Forest, XGBoost, LSTM, and GRU models for intrusion detection
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LSTM_CONFIG, AUTOENCODER_CONFIG, TIMESERIES_DIR, PROCESSED_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class LSTMClassifier(nn.Module):
    """LSTM model for time series classification"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Take last time step
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        return out

class GRUClassifier(nn.Module):
    """GRU model for time series classification"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        return out

class TimeSeriesIDSTrainer:
    def __init__(self):
        """Initialize model trainer"""
        self.models = {}
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        # Ensure output directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        logger.info("Time Series IDS Trainer initialized")
    
    def load_data(self):
        """Load engineered features and sequences"""
        logger.info("Loading features and sequences...")
        
        # Load processed features
        processed_path = os.path.join(PROCESSED_DIR, "cleaned_features.parquet")
        if os.path.exists(processed_path):
            self.flow_data = pd.read_parquet(processed_path)
            logger.info(f"Processed data shape: {self.flow_data.shape}")
        else:
            self.flow_data = None
            logger.warning("No processed data found")
        
        # Load time series features
        ts_path = os.path.join(TIMESERIES_DIR, "time_series_features.parquet")
        if os.path.exists(ts_path):
            self.ts_data = pd.read_parquet(ts_path)
            logger.info(f"Time-series data shape: {self.ts_data.shape}")
        else:
            self.ts_data = None
            logger.warning("No time-series data found")
        
        # Load LSTM sequences
        sequences_path = os.path.join(TIMESERIES_DIR, "lstm_sequences.npz")
        if os.path.exists(sequences_path):
            sequences = np.load(sequences_path, allow_pickle=True)
            self.X_sequences = sequences['X']
            self.y_sequences = sequences['y']
            logger.info(f"LSTM sequences shape: X={self.X_sequences.shape}, y={self.y_sequences.shape}")
        else:
            self.X_sequences = None
            self.y_sequences = None
            logger.warning("No LSTM sequences found")
        
        return self.flow_data, self.ts_data
    
    def prepare_flow_data(self, df):
        """Prepare flow data for machine learning"""
        logger.info("Preparing data for ML...")
        
        # Separate features and labels
        label_column = 'is_attack'
        
        if label_column not in df.columns:
            logger.error("No label column found!")
            return None, None, None
        
        # Get feature columns (exclude labels, leaky features, and non-features)
        # IMPORTANT: Label_encoded causes data leakage - it's derived from the target!
        exclude_columns = ['is_attack', 'Label', 'Label_encoded', 'source_file', 'Timestamp', 'timestamp']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        X = df[feature_columns].copy()
        y = df[label_column].copy()
        
        # Remove infinite and NaN values
        X = X.replace([np.inf, -np.inf], np.nan)
        valid_mask = ~X.isnull().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Fill any remaining NaN
        X = X.fillna(0)
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Labels distribution - Benign: {(y==0).sum()}, Attack: {(y==1).sum()}")
        
        return X, y, feature_columns
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest Classifier"""
        logger.info("Training Random Forest Classifier...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['rf'] = scaler
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        logger.info("Fitting Random Forest...")
        rf.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf
        
        # Predictions
        y_pred = rf.predict(X_test_scaled)
        y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info("=" * 50)
        logger.info("RANDOM FOREST RESULTS:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"ROC AUC Score: {auc:.4f}")
        logger.info("=" * 50)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return rf, auc
    
    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost Classifier"""
        logger.info("Training XGBoost Classifier...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['xgb'] = scaler
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc'
        )
        
        logger.info("Fitting XGBoost...")
        model.fit(X_train_scaled, y_train, 
                 eval_set=[(X_test_scaled, y_test)],
                 verbose=False)
        
        self.models['xgboost'] = model
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info("=" * 50)
        logger.info("XGBOOST RESULTS:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"ROC AUC Score: {auc:.4f}")
        logger.info("=" * 50)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return model, auc
    
    def train_isolation_forest(self, X_train):
        """Train Isolation Forest for anomaly detection"""
        logger.info("Training Isolation Forest...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['isolation_forest'] = scaler
        
        # Train Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        iso_forest.fit(X_train_scaled)
        self.models['isolation_forest'] = iso_forest
        
        # Get anomaly predictions
        predictions = iso_forest.predict(X_train_scaled)
        anomalies = (predictions == -1).sum()
        
        logger.info(f"Isolation Forest trained. Detected {anomalies} anomalies out of {len(predictions)}")
        
        return iso_forest
    
    def train_lstm_pytorch(self, X_sequences, y_sequences, epochs=30, batch_size=64, max_samples=20000):
        """Train LSTM classifier using PyTorch (memory optimized)"""
        logger.info("Training LSTM Classifier (PyTorch)...")
        
        # Subsample if too large (memory optimization)
        if len(X_sequences) > max_samples:
            indices = np.random.choice(len(X_sequences), max_samples, replace=False)
            X_sequences = X_sequences[indices]
            y_sequences = y_sequences[indices]
            logger.info(f"Subsampled to {max_samples} sequences for memory efficiency")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_sequences, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences
        )
        
        # Convert to PyTorch tensors (keep test data on CPU for batched validation)
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Create test DataLoader for batched validation
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = X_train.shape[2]
        model = LSTMClassifier(input_size, hidden_size=64, num_layers=1, dropout=0.1).to(device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        # Training loop
        best_auc = 0
        patience_counter = 0
        
        logger.info("Starting LSTM training...")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Batched validation
            model.eval()
            val_outputs_list = []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X).cpu().numpy()
                    val_outputs_list.append(outputs)
            
            val_outputs = np.vstack(val_outputs_list)
            val_pred = (val_outputs > 0.5).astype(int).flatten()
            
            val_accuracy = accuracy_score(y_test, val_pred)
            val_auc = roc_auc_score(y_test, val_outputs.flatten())
                
            avg_loss = total_loss / len(train_loader)
            scheduler.step(avg_loss)
            
            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'lstm_best.pth'))
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}")
            
            if patience_counter >= 7:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'lstm_best.pth')))
        self.models['lstm'] = model
        
        # Final evaluation (batched)
        model.eval()
        test_outputs_list = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X).cpu().numpy()
                test_outputs_list.append(outputs)
        
        test_outputs = np.vstack(test_outputs_list)
        test_pred = (test_outputs > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_test, test_pred)
        f1 = f1_score(y_test, test_pred)
        auc = roc_auc_score(y_test, test_outputs.flatten())
        
        logger.info("=" * 50)
        logger.info("LSTM CLASSIFIER RESULTS:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"ROC AUC Score: {auc:.4f}")
        logger.info("=" * 50)
        print("\nClassification Report:")
        print(classification_report(y_test, test_pred))
        
        return model, auc
    
    def save_models(self):
        """Save trained models and scalers"""
        logger.info(f"Saving models to: {MODELS_DIR}")
        
        # Save sklearn models
        for name, model in self.models.items():
            if name in ['random_forest', 'isolation_forest']:
                joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))
            elif name == 'xgboost':
                # Save XGBoost model using joblib instead of native save_model
                joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))
            elif name == 'lstm':
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{name}_final.pth"))
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(MODELS_DIR, f"scaler_{name}.pkl"))
        
        # Save label encoder
        joblib.dump(self.label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
        
        logger.info("All models saved successfully")

def main():
    """Main training pipeline"""
    try:
        trainer = TimeSeriesIDSTrainer()
        
        logger.info("=" * 60)
        logger.info("STARTING TIME SERIES MODEL TRAINING PIPELINE")
        logger.info("=" * 60)
        
        # Load data
        flow_data, ts_data = trainer.load_data()
        
        if flow_data is None:
            logger.error("No data available. Please run preprocessing first!")
            logger.info("Run: python preprocessing/data_cleaning.py")
            return
        
        # Prepare data
        X, y, feature_columns = trainer.prepare_flow_data(flow_data)
        
        if X is None:
            logger.error("Failed to prepare data!")
            return
        
        # Train-test split
        logger.info("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Train traditional ML models
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING TRADITIONAL ML MODELS")
        logger.info("=" * 60)
        
        # Train Random Forest
        rf_model, rf_auc = trainer.train_random_forest(X_train, X_test, y_train, y_test)
        
        # Train XGBoost
        xgb_model, xgb_auc = trainer.train_xgboost(X_train, X_test, y_train, y_test)
        
        # Train Isolation Forest
        iso_forest = trainer.train_isolation_forest(X_train)
        
        # Train LSTM if sequences are available
        if trainer.X_sequences is not None and trainer.y_sequences is not None:
            logger.info("\n" + "=" * 60)
            logger.info("TRAINING DEEP LEARNING MODELS")
            logger.info("=" * 60)
            
            lstm_model, lstm_auc = trainer.train_lstm_pytorch(
                trainer.X_sequences, trainer.y_sequences, epochs=30
            )
        
        # Save all models
        trainer.save_models()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETED - MODEL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Random Forest AUC:    {rf_auc:.4f}")
        logger.info(f"XGBoost AUC:          {xgb_auc:.4f}")
        if trainer.X_sequences is not None:
            logger.info(f"LSTM AUC:             {lstm_auc:.4f}")
        logger.info("=" * 60)
        logger.info("All models saved to: " + str(MODELS_DIR))
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()