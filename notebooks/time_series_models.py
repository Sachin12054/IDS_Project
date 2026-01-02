"""
================================================================================
EFFICIENT TIME SERIES MODELS FOR IDS
================================================================================
Three Model Categories:
1. LINEAR MODEL      → SARIMA (Best for seasonal patterns)
2. NON-LINEAR MODEL  → XGBoost Time Series (Handles complex patterns)
3. DEEP LEARNING     → LSTM (Sequential pattern recognition)

Author: Computer Security Project - Amrita University
================================================================================
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Statistical Models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# Machine Learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          "evaluation_results", "time_series_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')


class TimeSeriesIDS:
    """
    Efficient Time Series Analysis for Intrusion Detection
    Implements: SARIMA (Linear) + XGBoost (Non-Linear) + LSTM (Deep Learning)
    """
    
    def __init__(self):
        self.results = {}
        self.attack_series = None
        self.scaler = MinMaxScaler()
        
    def load_and_prepare_data(self):
        """Load data and create hourly attack time series"""
        print("=" * 70)
        print("LOADING AND PREPARING TIME SERIES DATA")
        print("=" * 70)
        
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "processed", "cleaned_features.parquet")
        
        df = pd.read_parquet(data_path)
        print(f"Loaded {len(df):,} records")
        
        # Create timestamp
        df['timestamp'] = pd.date_range(start='2018-02-14', periods=len(df), freq='1s')
        
        # Aggregate to hourly attack counts
        df['hour'] = df['timestamp'].dt.floor('h')
        self.attack_series = df.groupby('hour')['is_attack'].sum()
        
        print(f"Created hourly time series: {len(self.attack_series)} observations")
        print(f"Attack range: [{self.attack_series.min()}, {self.attack_series.max()}]")
        print(f"Mean attacks/hour: {self.attack_series.mean():.2f}")
        
        # Stationarity Test
        adf_result = adfuller(self.attack_series)
        print(f"\nADF Test: Statistic={adf_result[0]:.4f}, p-value={adf_result[1]:.4f}")
        print(f"Series is {'STATIONARY' if adf_result[1] < 0.05 else 'NON-STATIONARY'}")
        
        return self.attack_series
    
    def train_test_split(self, series, train_ratio=0.8):
        """Split time series into train and test"""
        train_size = int(len(series) * train_ratio)
        train = series.iloc[:train_size]
        test = series.iloc[train_size:]
        print(f"\nTrain: {len(train)} samples | Test: {len(test)} samples")
        return train, test
    
    # =========================================================================
    # MODEL 1: SARIMA (LINEAR MODEL)
    # =========================================================================
    
    def train_sarima(self, train, test):
        """
        SARIMA - Seasonal AutoRegressive Integrated Moving Average
        Best LINEAR model for time series with seasonal patterns
        
        Parameters: SARIMA(p,d,q)(P,D,Q,s)
        - p: AR order, d: differencing, q: MA order
        - P,D,Q: Seasonal components, s: Seasonal period (24 for hourly)
        """
        print("\n" + "=" * 70)
        print("MODEL 1: SARIMA (LINEAR MODEL)")
        print("=" * 70)
        
        try:
            # SARIMA(1,1,1)(1,0,1,24) - Simple but effective
            print("Fitting SARIMA(1,1,1)(1,0,1,24)...")
            
            model = SARIMAX(train, 
                           order=(1, 1, 1),
                           seasonal_order=(1, 0, 1, 24),
                           enforce_stationarity=False,
                           enforce_invertibility=False)
            
            model_fit = model.fit(disp=False)
            
            # Forecast
            forecast = model_fit.forecast(steps=len(test))
            
            # Metrics
            mse = mean_squared_error(test, forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test, forecast)
            
            print(f"\n📊 SARIMA Results:")
            print(f"   AIC: {model_fit.aic:.2f}")
            print(f"   BIC: {model_fit.bic:.2f}")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   MAE: {mae:.4f}")
            
            self.results['SARIMA'] = {
                'model': 'SARIMA(1,1,1)(1,0,1,24)',
                'type': 'LINEAR',
                'aic': model_fit.aic,
                'rmse': rmse,
                'mae': mae,
                'forecast': forecast
            }
            
            return model_fit, forecast
            
        except Exception as e:
            print(f"SARIMA Error: {e}")
            return None, None
    
    # =========================================================================
    # MODEL 2: XGBOOST TIME SERIES (NON-LINEAR MODEL)
    # =========================================================================
    
    def create_features(self, series, n_lags=24):
        """Create lag features for supervised learning"""
        df = pd.DataFrame({'target': series.values})
        
        # Lag features
        for i in range(1, n_lags + 1):
            df[f'lag_{i}'] = df['target'].shift(i)
        
        # Rolling statistics
        df['rolling_mean_6'] = df['target'].shift(1).rolling(6).mean()
        df['rolling_std_6'] = df['target'].shift(1).rolling(6).std()
        df['rolling_mean_24'] = df['target'].shift(1).rolling(24).mean()
        
        # Time features
        df['hour'] = series.index.hour
        df['day_of_week'] = series.index.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        df = df.dropna()
        return df
    
    def train_xgboost(self, train, test):
        """
        XGBoost Time Series - Gradient Boosting for Non-Linear Patterns
        Best NON-LINEAR model for complex relationships
        """
        print("\n" + "=" * 70)
        print("MODEL 2: XGBOOST (NON-LINEAR MODEL)")
        print("=" * 70)
        
        try:
            # Combine for feature creation
            full_series = pd.concat([train, test])
            df = self.create_features(full_series)
            
            # Split back
            train_size = len(train) - 24  # Account for lag features
            
            X = df.drop('target', axis=1)
            y = df['target']
            
            X_train = X.iloc[:train_size]
            y_train = y.iloc[:train_size]
            X_test = X.iloc[train_size:]
            y_test = y.iloc[train_size:]
            
            print(f"Training XGBoost on {len(X_train)} samples...")
            
            # XGBoost model
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
            
            model.fit(X_train, y_train)
            
            # Predict
            forecast = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, forecast)
            r2 = r2_score(y_test, forecast)
            
            print(f"\n📊 XGBoost Results:")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   MAE: {mae:.4f}")
            print(f"   R² Score: {r2:.4f}")
            
            # Feature importance
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            print(f"\n   Top Features:")
            for _, row in importance.head(5).iterrows():
                print(f"   - {row['feature']}: {row['importance']:.4f}")
            
            self.results['XGBoost'] = {
                'model': 'XGBoost Regressor',
                'type': 'NON-LINEAR',
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'forecast': forecast,
                'y_test': y_test.values
            }
            
            return model, forecast, y_test
            
        except Exception as e:
            print(f"XGBoost Error: {e}")
            return None, None, None
    
    # =========================================================================
    # MODEL 3: LSTM (DEEP LEARNING MODEL)
    # =========================================================================
    
    def create_sequences(self, data, seq_length=24):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def train_lstm(self, train, test, seq_length=24, epochs=50):
        """
        LSTM - Long Short-Term Memory Network
        Best DEEP LEARNING model for sequential patterns
        """
        print("\n" + "=" * 70)
        print("MODEL 3: LSTM (DEEP LEARNING MODEL)")
        print("=" * 70)
        
        try:
            # Normalize data
            train_scaled = self.scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
            test_scaled = self.scaler.transform(test.values.reshape(-1, 1)).flatten()
            
            # Create sequences
            X_train, y_train = self.create_sequences(train_scaled, seq_length)
            
            # For test, we need some history
            combined = np.concatenate([train_scaled[-seq_length:], test_scaled])
            X_test, y_test = self.create_sequences(combined, seq_length)
            
            # Convert to PyTorch tensors
            X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)
            y_train_t = torch.FloatTensor(y_train)
            X_test_t = torch.FloatTensor(X_test).unsqueeze(-1)
            y_test_t = torch.FloatTensor(y_test)
            
            # DataLoader
            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # LSTM Model
            class LSTMModel(nn.Module):
                def __init__(self, input_size=1, hidden_size=64, num_layers=2):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                       batch_first=True, dropout=0.2)
                    self.fc = nn.Linear(hidden_size, 1)
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    return self.fc(lstm_out[:, -1, :]).squeeze()
            
            model = LSTMModel()
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            print(f"Training LSTM for {epochs} epochs...")
            
            # Training loop
            model.train()
            for epoch in range(epochs):
                total_loss = 0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    print(f"   Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")
            
            # Predict
            model.eval()
            with torch.no_grad():
                predictions_scaled = model(X_test_t).numpy()
            
            # Inverse transform
            forecast = self.scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
            y_test_actual = self.scaler.inverse_transform(y_test_t.numpy().reshape(-1, 1)).flatten()
            
            # Metrics
            mse = mean_squared_error(y_test_actual, forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_actual, forecast)
            
            print(f"\n📊 LSTM Results:")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   MAE: {mae:.4f}")
            
            self.results['LSTM'] = {
                'model': 'LSTM (2 layers, 64 hidden)',
                'type': 'DEEP LEARNING',
                'rmse': rmse,
                'mae': mae,
                'forecast': forecast,
                'y_test': y_test_actual
            }
            
            return model, forecast, y_test_actual
            
        except Exception as e:
            print(f"LSTM Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    # =========================================================================
    # VISUALIZATION & COMPARISON
    # =========================================================================
    
    def plot_results(self, train, test):
        """Create comparison plots for all models"""
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Time Series Overview
        ax1 = axes[0, 0]
        ax1.plot(train.index, train.values, label='Training', alpha=0.7)
        ax1.plot(test.index, test.values, label='Actual Test', linewidth=2)
        ax1.set_title('Attack Time Series - Overview', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Attack Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. SARIMA Forecast
        ax2 = axes[0, 1]
        if 'SARIMA' in self.results:
            ax2.plot(test.index, test.values, label='Actual', linewidth=2, color='blue')
            ax2.plot(test.index, self.results['SARIMA']['forecast'], 
                    label=f"SARIMA (RMSE: {self.results['SARIMA']['rmse']:.2f})", 
                    linewidth=2, color='red', linestyle='--')
        ax2.set_title('LINEAR: SARIMA Forecast', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. XGBoost Forecast
        ax3 = axes[1, 0]
        if 'XGBoost' in self.results:
            y_test = self.results['XGBoost']['y_test']
            forecast = self.results['XGBoost']['forecast']
            ax3.plot(range(len(y_test)), y_test, label='Actual', linewidth=2, color='blue')
            ax3.plot(range(len(forecast)), forecast, 
                    label=f"XGBoost (RMSE: {self.results['XGBoost']['rmse']:.2f})", 
                    linewidth=2, color='green', linestyle='--')
        ax3.set_title('NON-LINEAR: XGBoost Forecast', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. LSTM Forecast
        ax4 = axes[1, 1]
        if 'LSTM' in self.results:
            y_test = self.results['LSTM']['y_test']
            forecast = self.results['LSTM']['forecast']
            ax4.plot(range(len(y_test)), y_test, label='Actual', linewidth=2, color='blue')
            ax4.plot(range(len(forecast)), forecast, 
                    label=f"LSTM (RMSE: {self.results['LSTM']['rmse']:.2f})", 
                    linewidth=2, color='purple', linestyle='--')
        ax4.set_title('DEEP LEARNING: LSTM Forecast', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Model Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(self.results.keys())
        rmse_values = [self.results[m]['rmse'] for m in models]
        mae_values = [self.results[m]['mae'] for m in models]
        colors = ['#ef4444', '#10b981', '#6366f1']
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, rmse_values, width, label='RMSE', color=colors, alpha=0.8)
        bars2 = ax.bar(x + width/2, mae_values, width, label='MAE', color=colors, alpha=0.5)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Error', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m}\n({self.results[m]['type']})" for m in models])
        ax.legend()
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'model_metrics_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Plots saved to: {OUTPUT_DIR}")
    
    def generate_report(self):
        """Generate summary report"""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 70)
        
        # Find best model
        best_model = min(self.results.keys(), key=lambda x: self.results[x]['rmse'])
        
        print(f"\n{'Model':<25} {'Type':<15} {'RMSE':<12} {'MAE':<12}")
        print("-" * 65)
        
        for name, result in self.results.items():
            marker = "🏆" if name == best_model else "  "
            print(f"{marker} {name:<23} {result['type']:<15} {result['rmse']:<12.4f} {result['mae']:<12.4f}")
        
        print("-" * 65)
        print(f"\n🏆 BEST MODEL: {best_model} ({self.results[best_model]['type']})")
        print(f"   RMSE: {self.results[best_model]['rmse']:.4f}")
        
        # Save report
        report = f"""# Time Series Models Comparison Report
## IDS Attack Prediction

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Models Implemented

### 1. LINEAR MODEL: SARIMA
- **Model**: SARIMA(1,1,1)(1,0,1,24)
- **Type**: Statistical Time Series
- **RMSE**: {self.results.get('SARIMA', {}).get('rmse', 'N/A')}
- **Best For**: Capturing seasonal patterns (24-hour cycles)

### 2. NON-LINEAR MODEL: XGBoost
- **Model**: XGBoost Regressor (100 trees)
- **Type**: Gradient Boosting
- **RMSE**: {self.results.get('XGBoost', {}).get('rmse', 'N/A')}
- **R² Score**: {self.results.get('XGBoost', {}).get('r2', 'N/A')}
- **Best For**: Complex non-linear relationships

### 3. DEEP LEARNING: LSTM
- **Model**: 2-Layer LSTM (64 hidden units)
- **Type**: Recurrent Neural Network
- **RMSE**: {self.results.get('LSTM', {}).get('rmse', 'N/A')}
- **Best For**: Sequential pattern recognition

---

## Results Summary

| Model | Type | RMSE | MAE |
|-------|------|------|-----|
| SARIMA | Linear | {self.results.get('SARIMA', {}).get('rmse', 'N/A'):.4f} | {self.results.get('SARIMA', {}).get('mae', 'N/A'):.4f} |
| XGBoost | Non-Linear | {self.results.get('XGBoost', {}).get('rmse', 'N/A'):.4f} | {self.results.get('XGBoost', {}).get('mae', 'N/A'):.4f} |
| LSTM | Deep Learning | {self.results.get('LSTM', {}).get('rmse', 'N/A'):.4f} | {self.results.get('LSTM', {}).get('mae', 'N/A'):.4f} |

---

## Best Model: {best_model}

The {best_model} model achieved the lowest RMSE of {self.results[best_model]['rmse']:.4f}

---

## Files Generated

- `model_comparison.png` - Forecast comparison plots
- `model_metrics_comparison.png` - Performance metrics bar chart
- `time_series_models_report.md` - This report
"""
        
        report_path = os.path.join(OUTPUT_DIR, 'time_series_models_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ Report saved to: {report_path}")
        
        return self.results


def main():
    """Main execution"""
    print("=" * 70)
    print("EFFICIENT TIME SERIES MODELS FOR IDS")
    print("LINEAR (SARIMA) | NON-LINEAR (XGBoost) | DEEP LEARNING (LSTM)")
    print("=" * 70)
    
    # Initialize
    ts_ids = TimeSeriesIDS()
    
    # Load data
    series = ts_ids.load_and_prepare_data()
    
    # Train-test split
    train, test = ts_ids.train_test_split(series)
    
    # Train models
    ts_ids.train_sarima(train, test)
    ts_ids.train_xgboost(train, test)
    ts_ids.train_lstm(train, test, epochs=30)
    
    # Visualize
    ts_ids.plot_results(train, test)
    
    # Report
    results = ts_ids.generate_report()
    
    print("\n" + "=" * 70)
    print("✅ TIME SERIES ANALYSIS COMPLETED!")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
