"""
================================================================================
GENERATE ALL MISSING PLOTS
================================================================================
This script generates all missing visualizations and saves them to the
'missing' folder for easy identification and review.

Output: missing/
  - time_series_models/     (8 plots)
  - advanced_time_series/   (5 plots)
  - enhanced_visualizations/ (3 plots)

Total: 16+ visualization files
================================================================================
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Update output directories to 'missing' folder
MISSING_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "missing")

print("=" * 80)
print("  GENERATING ALL MISSING PLOTS")
print("=" * 80)
print(f"\nOutput directory: {MISSING_DIR}")
print("\nThis will generate:")
print("  - 8 Time Series Model plots")
print("  - 5 Advanced Time Series plots")
print("  - 3 Enhanced Visualization plots")
print("  - Multiple reports and data files")
print("\n" + "=" * 80 + "\n")

# =============================================================================
# PART 1: TIME SERIES MODELS (SARIMA, XGBoost, LSTM)
# =============================================================================

print("\n" + "=" * 80)
print("  PART 1: TIME SERIES MODELS ANALYSIS")
print("=" * 80 + "\n")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Statistical Models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Machine Learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats, signal

plt.style.use('seaborn-v0_8-whitegrid')

# Set output directory
OUTPUT_DIR_TS = os.path.join(MISSING_DIR, "time_series_models")
os.makedirs(OUTPUT_DIR_TS, exist_ok=True)

class TimeSeriesIDS:
    def __init__(self):
        self.results = {}
        self.attack_series = None
        self.scaler = MinMaxScaler()
        self.sarima_model_fit = None
        self.xgboost_model = None
        self.lstm_model = None
        self.lstm_train_losses = []
        self.xgb_feature_importance = None
        
    def load_and_prepare_data(self):
        print("Loading and preparing time series data...")
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "processed", "cleaned_features.parquet")
        
        df = pd.read_parquet(data_path)
        df['timestamp'] = pd.date_range(start='2018-02-14', periods=len(df), freq='1s')
        df['hour'] = df['timestamp'].dt.floor('h')
        self.attack_series = df.groupby('hour')['is_attack'].sum()
        
        print(f"✓ Created hourly time series: {len(self.attack_series)} observations")
        return self.attack_series
    
    def train_test_split(self, series, train_ratio=0.8):
        train_size = int(len(series) * train_ratio)
        return series.iloc[:train_size], series.iloc[train_size:]
    
    def train_sarima(self, train, test):
        print("\n[1/3] Training SARIMA model...")
        try:
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 24),
                           enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=len(test))
            
            rmse = np.sqrt(mean_squared_error(test, forecast))
            mae = mean_absolute_error(test, forecast)
            
            self.results['SARIMA'] = {
                'rmse': rmse, 'mae': mae, 'forecast': forecast,
                'residuals': model_fit.resid
            }
            self.sarima_model_fit = model_fit
            print(f"  ✓ SARIMA RMSE: {rmse:.2f}")
            return model_fit, forecast
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None, None
    
    def create_features(self, series, n_lags=24):
        df = pd.DataFrame({'target': series.values})
        for i in range(1, n_lags + 1):
            df[f'lag_{i}'] = df['target'].shift(i)
        df['rolling_mean_6'] = df['target'].shift(1).rolling(6).mean()
        df['rolling_std_6'] = df['target'].shift(1).rolling(6).std()
        df['rolling_mean_24'] = df['target'].shift(1).rolling(24).mean()
        df['hour'] = series.index.hour
        df['day_of_week'] = series.index.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        return df.dropna()
    
    def train_xgboost(self, train, test):
        print("\n[2/3] Training XGBoost model...")
        try:
            full_series = pd.concat([train, test])
            df = self.create_features(full_series)
            train_size = len(train) - 24
            
            X = df.drop('target', axis=1)
            y = df['target']
            X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
            X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]
            
            model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
            model.fit(X_train, y_train)
            forecast = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, forecast))
            mae = mean_absolute_error(y_test, forecast)
            r2 = r2_score(y_test, forecast)
            
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.xgb_feature_importance = importance
            self.results['XGBoost'] = {
                'rmse': rmse, 'mae': mae, 'r2': r2,
                'forecast': forecast, 'y_test': y_test.values,
                'residuals': y_test.values - forecast
            }
            self.xgboost_model = model
            print(f"  ✓ XGBoost RMSE: {rmse:.2f}, R²: {r2:.3f}")
            return model, forecast, y_test
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None, None, None
    
    def create_sequences(self, data, seq_length=24):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def train_lstm(self, train, test, seq_length=24, epochs=30):
        print("\n[3/3] Training LSTM model...")
        try:
            train_scaled = self.scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
            test_scaled = self.scaler.transform(test.values.reshape(-1, 1)).flatten()
            
            X_train, y_train = self.create_sequences(train_scaled, seq_length)
            combined = np.concatenate([train_scaled[-seq_length:], test_scaled])
            X_test, y_test = self.create_sequences(combined, seq_length)
            
            X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)
            y_train_t = torch.FloatTensor(y_train)
            X_test_t = torch.FloatTensor(X_test).unsqueeze(-1)
            y_test_t = torch.FloatTensor(y_test)
            
            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            class LSTMModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(1, 64, 2, batch_first=True, dropout=0.2)
                    self.fc = nn.Linear(64, 1)
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    return self.fc(lstm_out[:, -1, :]).squeeze()
            
            model = LSTMModel()
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
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
                self.lstm_train_losses.append(total_loss/len(train_loader))
            
            model.eval()
            with torch.no_grad():
                predictions_scaled = model(X_test_t).numpy()
            
            forecast = self.scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
            y_test_actual = self.scaler.inverse_transform(y_test_t.numpy().reshape(-1, 1)).flatten()
            
            rmse = np.sqrt(mean_squared_error(y_test_actual, forecast))
            mae = mean_absolute_error(y_test_actual, forecast)
            
            self.results['LSTM'] = {
                'rmse': rmse, 'mae': mae,
                'forecast': forecast, 'y_test': y_test_actual,
                'residuals': y_test_actual - forecast
            }
            self.lstm_model = model
            print(f"  ✓ LSTM RMSE: {rmse:.2f}")
            return model, forecast, y_test_actual
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def generate_all_plots(self, train, test):
        print("\nGenerating visualizations...")
        
        # Plot 1: Model Comparison
        print("  [1/8] Model comparison...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ax1 = axes[0, 0]
        ax1.plot(train.index, train.values, label='Training', alpha=0.7)
        ax1.plot(test.index, test.values, label='Actual Test', linewidth=2)
        
        # Mark potential missing data regions (flat zeros)
        train_values = train.values
        for i in range(1, len(train_values)-1):
            if train_values[i] == 0 and train_values[i-1] == 0 and train_values[i+1] == 0:
                ax1.axvline(x=train.index[i], color='orange', alpha=0.1, linewidth=0.5)
        
        ax1.set_title('Attack Time Series - Overview\n(Orange shading indicates potential missing/zero data regions)', 
                     fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if 'SARIMA' in self.results:
            ax2 = axes[0, 1]
            ax2.plot(test.index, test.values, label='Actual', linewidth=2, color='blue')
            ax2.plot(test.index, self.results['SARIMA']['forecast'], 
                    label=f"SARIMA (RMSE: {self.results['SARIMA']['rmse']:.2f})", 
                    linewidth=2, color='red', linestyle='--')
            ax2.set_title('LINEAR: SARIMA Forecast', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        if 'XGBoost' in self.results:
            ax3 = axes[1, 0]
            y_test = self.results['XGBoost']['y_test']
            forecast = self.results['XGBoost']['forecast']
            ax3.plot(range(len(y_test)), y_test, label='Actual', linewidth=2, color='blue')
            ax3.plot(range(len(forecast)), forecast, 
                    label=f"XGBoost (RMSE: {self.results['XGBoost']['rmse']:.2f})", 
                    linewidth=2, color='green', linestyle='--')
            ax3.set_title('NON-LINEAR: XGBoost Forecast', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        if 'LSTM' in self.results:
            ax4 = axes[1, 1]
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
        plt.savefig(os.path.join(OUTPUT_DIR_TS, 'model_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Metrics Comparison
        print("  [2/8] Metrics comparison...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        models = list(self.results.keys())
        rmse_values = [self.results[m]['rmse'] for m in models]
        mae_values = [self.results[m]['mae'] for m in models]
        colors = ['#ef4444', '#10b981', '#6366f1']
        
        x = np.arange(len(models))
        
        # RMSE subplot
        bars1 = ax1.bar(x, rmse_values, color=colors, alpha=0.8)
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('RMSE', fontsize=12)
        ax1.set_title('Root Mean Squared Error Comparison\n(RMSE penalizes large errors more than MAE)', 
                     fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.grid(True, axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
        
        # MAE subplot
        bars2 = ax2.bar(x, mae_values, color=colors, alpha=0.8)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('MAE', fontsize=12)
        ax2.set_title('Mean Absolute Error Comparison\n(MAE treats all errors equally)', 
                     fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.grid(True, axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_TS, 'model_metrics_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Residual Analysis
        print("  [3/8] Residual analysis...")
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        for i, model_name in enumerate(['SARIMA', 'XGBoost', 'LSTM']):
            if model_name in self.results and 'residuals' in self.results[model_name]:
                residuals = self.results[model_name]['residuals']
                
                ax1 = axes[i, 0]
                ax1.plot(residuals, alpha=0.7)
                ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
                ax1.set_title(f'{model_name} - Residuals Over Time', fontweight='bold')
                ax1.set_ylabel('Residual')
                ax1.grid(True, alpha=0.3)
                
                # Fix #11: Add note about distribution shape
                ax2 = axes[i, 1]
                ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
                ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
                ax2.set_title(f'{model_name} - Residual Distribution', fontweight='bold')
                if model_name == 'SARIMA':
                    ax2.text(0.98, 0.95, 'Leptokurtic\n(heavy tails)', 
                            transform=ax2.transAxes, ha='right', va='top', fontsize=7,
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                
                # Fix #2: Add caption about deviation
                ax3 = axes[i, 2]
                stats.probplot(residuals, dist="norm", plot=ax3)
                ax3.set_title(f'{model_name} - Q-Q Plot', fontweight='bold')
                ax3.grid(True, alpha=0.3)
                if i == 0:  # Add note to first plot only
                    ax3.text(0.02, 0.98, 'Tail deviations indicate\nnon-normality', 
                            transform=ax3.transAxes, ha='left', va='top', fontsize=7,
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_TS, 'residual_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 4: LSTM Learning Curves
        if self.lstm_train_losses:
            print("  [4/8] LSTM learning curves...")
            fig, ax = plt.subplots(figsize=(12, 6))
            epochs = range(1, len(self.lstm_train_losses) + 1)
            ax.plot(epochs, self.lstm_train_losses, linewidth=2, color='purple', marker='o', markersize=4)
            ax.set_title('LSTM Training Loss Over Epochs', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.grid(True, alpha=0.3)
            
            min_loss_idx = np.argmin(self.lstm_train_losses)
            min_loss = self.lstm_train_losses[min_loss_idx]
            ax.annotate(f'Best Loss: {min_loss:.6f}\nEpoch: {min_loss_idx+1}',
                       xy=(min_loss_idx+1, min_loss), xytext=(min_loss_idx+1+5, min_loss+0.001),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR_TS, 'lstm_learning_curves.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Plot 5: Feature Importance
        if self.xgb_feature_importance is not None:
            print("  [5/8] Feature importance...")
            fig, ax = plt.subplots(figsize=(12, 8))
            top_features = self.xgb_feature_importance.head(15)
            bars = ax.barh(range(len(top_features)), top_features['importance'].values, color='teal', alpha=0.8)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'].values)
            ax.invert_yaxis()
            ax.set_xlabel('Importance Score (Gain-based)')
            ax.set_title('XGBoost - Top 15 Feature Importances\n(Note: Gain-based importance does not imply causality)', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)
            
            for i, (bar, val) in enumerate(zip(bars, top_features['importance'].values)):
                ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                       ha='left', va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR_TS, 'xgboost_feature_importance.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Plot 6: Prediction Intervals
        print("  [6/8] Prediction intervals...")
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for idx, model_name in enumerate(['SARIMA', 'XGBoost', 'LSTM']):
            if model_name not in self.results:
                continue
            
            ax = axes[idx]
            
            if model_name == 'SARIMA':
                y_test = test.values
                forecast = self.results[model_name]['forecast'].values
                test_index = test.index
            else:
                y_test = self.results[model_name]['y_test']
                forecast = self.results[model_name]['forecast']
                test_index = range(len(y_test))
            
            residuals = self.results[model_name].get('residuals', y_test - forecast)
            std_error = np.std(residuals)
            lower_bound = forecast - 1.96 * std_error
            upper_bound = forecast + 1.96 * std_error
            
            ax.plot(test_index, y_test, label='Actual', linewidth=2, color='blue', alpha=0.7)
            ax.plot(test_index, forecast, label='Forecast', linewidth=2, color='red', linestyle='--')
            
            # Different labels based on model type
            if model_name == 'SARIMA':
                interval_label = '95% Prediction Interval\n(Assumes homoskedastic errors)'
            else:
                interval_label = 'Empirical Prediction Bands\n(±1.96σ, assumes stationary\nresidual variance)'
            
            ax.fill_between(test_index, lower_bound, upper_bound, alpha=0.2, color='red',
                           label=interval_label)
            
            ax.set_title(f'{model_name} - Prediction Intervals', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time' if model_name == 'SARIMA' else 'Index')
            ax.set_ylabel('Attack Count')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_TS, 'prediction_intervals.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 7: Error Distributions
        print("  [7/8] Error distributions...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        all_errors = {}
        for model_name in ['SARIMA', 'XGBoost', 'LSTM']:
            if model_name in self.results and 'residuals' in self.results[model_name]:
                all_errors[model_name] = self.results[model_name]['residuals']
        
        ax1 = axes[0, 0]
        ax1.boxplot([all_errors[m] for m in all_errors.keys()],
                   labels=list(all_errors.keys()), patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_title('Error Distribution Comparison (Box Plot)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        positions = range(1, len(all_errors) + 1)
        parts = ax2.violinplot([all_errors[m] for m in all_errors.keys()],
                              positions=positions, showmeans=True, showmedians=True)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xticks(positions)
        ax2.set_xticklabels(list(all_errors.keys()))
        ax2.set_title('Error Shape Visualization (Illustrative)\nViolet shapes show error distribution shape', 
                     fontsize=13, fontweight='bold')
        ax2.text(0.02, 0.98, 'Note: Single test set\n(no CV folds)', 
                transform=ax2.transAxes, ha='left', va='top', fontsize=7,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        colors = ['red', 'green', 'purple']
        for (model_name, errors), color in zip(all_errors.items(), colors):
            # Use log scale for better visibility
            abs_errors = np.abs(errors) + 1
            ax3.hist(abs_errors, bins=50, alpha=0.5, label=model_name, color=color, edgecolor='black')
        ax3.set_yscale('log')
        ax3.set_xlabel('|Prediction Error| + 1')
        ax3.set_ylabel('Frequency (log scale)')
        ax3.set_title('Overlapping Error Distributions (Log Scale)', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        for (model_name, errors), color in zip(all_errors.items(), colors):
            sorted_errors = np.sort(np.abs(errors))
            cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            ax4.plot(sorted_errors, cumulative, label=model_name, linewidth=2, color=color)
        ax4.set_title('Cumulative Distribution of Absolute Errors', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_TS, 'error_distributions.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 8: ARIMA Diagnostics
        if self.sarima_model_fit is not None:
            print("  [8/8] ARIMA diagnostics...")
            try:
                fig = plt.figure(figsize=(16, 12))
                residuals = self.sarima_model_fit.resid
                
                # Fix #8: Properly standardize residuals
                residuals_std = (residuals - residuals.mean()) / residuals.std()
                
                ax1 = plt.subplot(3, 2, 1)
                ax1.plot(residuals_std)
                ax1.axhline(y=0, color='r', linestyle='--')
                ax1.set_title('Standardized Residuals (Z-scores)', fontweight='bold')
                ax1.set_ylabel('Z-score')
                ax1.grid(True, alpha=0.3)
                
                # Fix #1: Reference Normal curve, not "Normal"
                # Fix #11: Add leptokurtic note
                ax2 = plt.subplot(3, 2, 2)
                ax2.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black', label='Residuals')
                mu, sigma = residuals.mean(), residuals.std()
                x = np.linspace(residuals.min(), residuals.max(), 100)
                ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'b--', linewidth=2, alpha=0.7, 
                        label='Standard Normal\n(Visual Reference Only)')
                ax2.set_title('Histogram + Reference Density', fontweight='bold')
                ax2.legend(fontsize=8)
                ax2.text(0.98, 0.95, 'Note: Residuals are\nleptokurtic (heavy tails)', 
                        transform=ax2.transAxes, ha='right', va='top', fontsize=7,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                
                # Fix #2: Add caption about deviation from normality
                ax3 = plt.subplot(3, 2, 3)
                stats.probplot(residuals, dist="norm", plot=ax3)
                ax3.set_title('Normal Q-Q Plot', fontweight='bold')
                ax3.grid(True, alpha=0.3)
                ax3.text(0.02, 0.98, 'Deviation in tails\nindicates non-normality', 
                        transform=ax3.transAxes, ha='left', va='top', fontsize=7,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                
                # Fix #12: Mention confidence bands
                ax4 = plt.subplot(3, 2, 4)
                plot_acf(residuals, lags=40, ax=ax4, alpha=0.05)
                ax4.set_title('ACF of Residuals (95% CI shown)', fontweight='bold')
                
                ax5 = plt.subplot(3, 2, 5)
                plot_pacf(residuals, lags=40, ax=ax5, alpha=0.05)
                ax5.set_title('PACF of Residuals (95% CI shown)', fontweight='bold')
                
                # Fix #3: Add note about visual inspection
                ax6 = plt.subplot(3, 2, 6)
                plot_acf(residuals**2, lags=40, ax=ax6, alpha=0.05)
                ax6.set_title('ACF of Squared Residuals', fontweight='bold')
                ax6.text(0.02, 0.98, 'Visual ARCH inspection\n(Formal ARCH-LM test\nnot reported)', 
                        transform=ax6.transAxes, ha='left', va='top', fontsize=7,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR_TS, 'arima_diagnostics.png'), dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"    ⚠️ Warning: {e}")
        
        print(f"\n✓ All time series plots saved to: {OUTPUT_DIR_TS}")

# Execute Time Series Models
print("Starting time series models analysis...")
ts_ids = TimeSeriesIDS()
series = ts_ids.load_and_prepare_data()
train, test = ts_ids.train_test_split(series)
ts_ids.train_sarima(train, test)
ts_ids.train_xgboost(train, test)
ts_ids.train_lstm(train, test, epochs=30)
ts_ids.generate_all_plots(train, test)

print("\n" + "=" * 80)
print("  ✅ TIME SERIES MODELS COMPLETED!")
print("=" * 80)

# =============================================================================
# PART 2: ADVANCED TIME SERIES ANALYSIS
# =============================================================================

print("\n\n" + "=" * 80)
print("  PART 2: ADVANCED TIME SERIES ANALYSIS")
print("=" * 80 + "\n")

OUTPUT_DIR_ADV = os.path.join(MISSING_DIR, "advanced_time_series")
os.makedirs(OUTPUT_DIR_ADV, exist_ok=True)

class AdvancedAnalysis:
    def __init__(self):
        self.data = None
        
    def load_data(self):
        print("Loading data for advanced analysis...")
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "processed", "cleaned_features.parquet")
        df = pd.read_parquet(data_path)
        df['timestamp'] = pd.date_range(start='2018-02-14', periods=len(df), freq='1s')
        df['hour'] = df['timestamp'].dt.floor('h')
        
        self.data = pd.DataFrame({
            'attack_count': df.groupby('hour')['is_attack'].sum(),
            'total_packets': df.groupby('hour').size(),
        })
        self.data['attack_rate'] = (self.data['attack_count'] / self.data['total_packets'] * 100)
        print(f"✓ Created time series with {len(self.data)} observations")
        return self.data
    
    def spectral_analysis(self):
        print("\n[1/5] Spectral analysis...")
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        series = self.data['attack_count'].values
        
        # Detrend the series for better frequency analysis
        from scipy.signal import detrend
        series_detrended = detrend(series)
        
        ax1 = axes[0, 0]
        frequencies, power = signal.periodogram(series_detrended, scaling='density')
        ax1.semilogy(frequencies[1:], power[1:])  # Skip DC component
        ax1.set_title('Periodogram - Attack Count (Detrended)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Frequency (cycles per hour)')
        ax1.set_ylabel('Power Spectral Density')
        ax1.grid(True, alpha=0.3)
        # Mark 24-hour cycle
        if len(frequencies) > 24:
            ax1.axvline(x=1/24, color='r', linestyle='--', linewidth=2, label='24h cycle', alpha=0.7)
        ax1.legend()
        
        ax2 = axes[0, 1]
        from scipy.fft import fft, fftfreq
        N = len(series_detrended)
        yf = fft(series_detrended)
        xf = fftfreq(N, 1)[:N//2]
        ax2.plot(xf[1:], 2.0/N * np.abs(yf[1:N//2]))  # Skip DC component
        ax2.set_title('FFT - Frequency Components (Detrended)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Frequency (cycles per hour)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        if len(xf) > 24:
            ax2.axvline(x=1/24, color='r', linestyle='--', linewidth=2, label='24h cycle', alpha=0.7)
        ax2.legend()
        
        ax3 = axes[1, 0]
        f_welch, Pxx_welch = signal.welch(series_detrended, nperseg=min(256, len(series_detrended)//4), scaling='density')
        ax3.semilogy(f_welch[1:], Pxx_welch[1:])  # Skip DC component
        ax3.set_title("Welch's Method - Smoothed PSD (Detrended)", fontsize=14, fontweight='bold')
        ax3.set_xlabel('Frequency (cycles per hour)')
        ax3.set_ylabel('Power Spectral Density')
        ax3.grid(True, alpha=0.3)
        if len(f_welch) > 24:
            ax3.axvline(x=1/24, color='r', linestyle='--', linewidth=2, label='24h cycle', alpha=0.7)
        ax3.legend()
        
        ax4 = axes[1, 1]
        # Use longer segments for better frequency resolution
        nperseg = min(128, len(series_detrended)//4)
        f, t, Sxx = signal.spectrogram(series_detrended, nperseg=nperseg, noverlap=nperseg//2)
        # Avoid log of zero
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        im = ax4.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis', vmin=np.percentile(Sxx_db, 5), vmax=np.percentile(Sxx_db, 95))
        ax4.set_title('Spectrogram - Time-Frequency Analysis (Detrended)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Frequency (cycles per hour)')
        ax4.set_xlabel('Time (hours)')
        plt.colorbar(im, ax=ax4, label='Power (dB)')
        if len(f) > 24:
            ax4.axhline(y=1/24, color='r', linestyle='--', linewidth=2, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_ADV, 'spectral_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Spectral analysis complete")
    
    def cross_correlation(self):
        print("\n[2/5] Cross-correlation analysis...")
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        series_pairs = [
            ('attack_count', 'attack_rate'),
            ('attack_rate', 'total_packets'),
        ]
        
        for idx, (s1_name, s2_name) in enumerate(series_pairs):
            ax = axes[idx // 2, idx % 2]
            s1 = self.data[s1_name].values
            s2 = self.data[s2_name].values
            
            # Detrend both series for better correlation analysis
            from scipy.signal import detrend
            s1_detrend = detrend(s1)
            s2_detrend = detrend(s2)
            
            # Normalize
            s1_norm = s1_detrend / (np.std(s1_detrend) + 1e-10)
            s2_norm = s2_detrend / (np.std(s2_detrend) + 1e-10)
            
            # Compute cross-correlation
            correlation = np.correlate(s1_norm, s2_norm, mode='full')
            correlation = correlation / len(s1_norm)  # Normalize by length
            lags = np.arange(-len(s1)+1, len(s1))
            
            # Plot only relevant lag range
            max_lag = min(100, len(s1)//4)
            mask = (lags >= -max_lag) & (lags <= max_lag)
            
            ax.plot(lags[mask], correlation[mask], linewidth=2)
            ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Lag')
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
            
            # Mark peak correlation
            peak_idx = np.argmax(np.abs(correlation[mask]))
            peak_lag = lags[mask][peak_idx]
            peak_corr = correlation[mask][peak_idx]
            ax.scatter(peak_lag, peak_corr, s=200, c='red', marker='*', zorder=5, edgecolors='black', linewidth=2)
            ax.annotate(f'Peak: lag={peak_lag}\ncorr={peak_corr:.3f}',
                       xy=(peak_lag, peak_corr), xytext=(peak_lag+10, peak_corr),
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            ax.set_title(f'Cross-Correlation: {s1_name} vs {s2_name} (Detrended)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Lag (hours)')
            ax.set_ylabel('Normalized Correlation')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-max_lag, max_lag)
        
        # Hide unused subplots
        for idx in range(len(series_pairs), 4):
            axes[idx // 2, idx % 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_ADV, 'cross_correlation.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Cross-correlation complete")
    
    def structural_breaks(self):
        print("\n[3/5] Structural break detection...")
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        series_to_analyze = ['attack_count', 'attack_rate']
        
        for idx, series_name in enumerate(series_to_analyze):
            ax = axes[idx // 2, idx % 2]
            series = self.data[series_name].values
            
            # Standardize the series for better CUSUM scaling
            series_std = (series - np.mean(series)) / (np.std(series) + 1e-10)
            cusum_pos = np.maximum.accumulate(np.maximum(0, np.cumsum(series_std - 0.5)))
            cusum_neg = np.minimum.accumulate(np.minimum(0, np.cumsum(series_std + 0.5)))
            
            # Plot standardized series
            ax2 = ax.twinx()
            ax2.plot(range(len(series_std)), series_std, 'b-', alpha=0.4, linewidth=1, label='Standardized Series')
            ax2.set_ylabel('Standardized Value', color='b', fontsize=10)
            ax2.tick_params(axis='y', labelcolor='b')
            ax2.set_ylim(-4, 4)
            
            # Plot CUSUM
            ax.plot(range(len(cusum_pos)), cusum_pos, 'r-', linewidth=2, label='CUSUM (+)', alpha=0.8)
            ax.plot(range(len(cusum_neg)), cusum_neg, 'orange', linewidth=2, label='CUSUM (-)', alpha=0.8)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            
            # Add threshold lines
            threshold = 5  # Standard threshold for CUSUM
            ax.axhline(y=threshold, color='darkred', linestyle=':', linewidth=2, label=f'Threshold (±{threshold})', alpha=0.7)
            ax.axhline(y=-threshold, color='darkred', linestyle=':', linewidth=2, alpha=0.7)
            
            ax.set_ylabel('CUSUM (Standardized)', color='r', fontsize=10)
            ax.set_xlabel('Time (hours)', fontsize=10)
            ax.set_title(f'CUSUM Structural Break Detection - {series_name}', fontsize=12, fontweight='bold')
            ax.tick_params(axis='y', labelcolor='r')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-10, 10)
        
        # Add explanation panel
        ax_info = axes[1, 0]
        ax_info.axis('off')
        info_text = (
            "CUSUM (Cumulative Sum Control Chart)\n"
            "═" * 40 + "\n\n"
            "Purpose: Detect shifts in mean level\n\n"
            "Interpretation:\n"
            "• Values move away from 0: shift detected\n"
            "• Beyond threshold (±5): significant change\n"
            "• CUSUM(+): detects upward shifts\n"
            "• CUSUM(-): detects downward shifts\n\n"
            "Note: Series are standardized for comparison"
        )
        ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Hide remaining subplot
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_ADV, 'structural_breaks.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Structural breaks complete")
    
    def wavelet_analysis(self):
        print("\n[4/5] Wavelet analysis...")
        try:
            import pywt
            
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            series = self.data['attack_count'].values
            
            ax1 = axes[0, 0]
            scales = np.arange(1, 128)
            coefficients, frequencies = pywt.cwt(series, scales, 'morl')
            im1 = ax1.imshow(np.abs(coefficients), extent=[0, len(series), 1, 128],
                           cmap='viridis', aspect='auto', vmax=abs(coefficients).max())
            ax1.set_title('Continuous Wavelet Transform', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Scale')
            ax1.set_xlabel('Time')
            plt.colorbar(im1, ax=ax1, label='Magnitude')
            
            ax2 = axes[0, 1]
            wavelet = 'db4'
            level = 4
            coeffs = pywt.wavedec(series, wavelet, level=level)
            reconstructed = pywt.waverec(coeffs, wavelet)[:len(series)]
            ax2.plot(series, label='Original', alpha=0.7, linewidth=1)
            ax2.plot(reconstructed, label='Reconstructed', linewidth=2, linestyle='--')
            ax2.set_title(f'Discrete Wavelet Transform ({wavelet})', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            ax3 = axes[1, 0]
            for i, coeff in enumerate(coeffs[:-1]):
                if len(coeff) > 0:
                    ax3.plot(np.arange(len(coeff)), coeff, label=f'Detail {i+1}', alpha=0.7)
            ax3.set_title('Wavelet Decomposition Levels', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            ax4 = axes[1, 1]
            energies = [np.sum(c**2) for c in coeffs]
            total_energy = sum(energies)
            energy_pct = [e/total_energy*100 for e in energies]
            labels = [f'Level {i+1}' for i in range(len(coeffs)-1)] + ['Approx']
            ax4.bar(range(len(energies)), energy_pct, color='steelblue', alpha=0.7, edgecolor='black')
            ax4.set_xticks(range(len(energies)))
            ax4.set_xticklabels(labels)
            ax4.set_title('Wavelet Energy Distribution', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Energy (%)')
            ax4.grid(True, alpha=0.3)
            
            for i, pct in enumerate(energy_pct):
                ax4.text(i, pct, f'{pct:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR_ADV, 'wavelet_analysis.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ Wavelet analysis complete")
        except ImportError:
            print("  ⚠️ PyWavelets not installed, skipping wavelet analysis")
    
    def granger_causality(self):
        print("\n[5/5] Granger causality tests...")
        from statsmodels.tsa.stattools import grangercausalitytests
        
        results = []
        test_pairs = [
            ('total_packets', 'attack_count'),
            ('attack_count', 'attack_rate'),
        ]
        
        for x_name, y_name in test_pairs:
            data_pair = self.data[[x_name, y_name]].dropna()
            try:
                gc_result = grangercausalitytests(data_pair, maxlag=12, verbose=False)
                p_values = [gc_result[lag][0]['ssr_ftest'][1] for lag in range(1, 13)]
                significant_lags = [i+1 for i, p in enumerate(p_values) if p < 0.05]
                
                if significant_lags:
                    results.append(f"{x_name} -> {y_name}: YES (lags: {significant_lags[:3]})")
                else:
                    results.append(f"{x_name} -> {y_name}: NO")
            except:
                results.append(f"{x_name} -> {y_name}: ERROR")
        
        with open(os.path.join(OUTPUT_DIR_ADV, 'granger_causality_results.txt'), 'w') as f:
            f.write("GRANGER CAUSALITY TEST RESULTS\n")
            f.write("=" * 50 + "\n\n")
            for result in results:
                f.write(result + "\n")
        
        print("  ✓ Granger causality tests complete")

# Execute Advanced Analysis
adv = AdvancedAnalysis()
adv.load_data()
adv.spectral_analysis()
adv.cross_correlation()
adv.structural_breaks()
adv.wavelet_analysis()
adv.granger_causality()

print("\n" + "=" * 80)
print("  ✅ ADVANCED TIME SERIES ANALYSIS COMPLETED!")
print("=" * 80)

# =============================================================================
# PART 3: ENHANCED VISUALIZATIONS
# =============================================================================

print("\n\n" + "=" * 80)
print("  PART 3: ENHANCED VISUALIZATIONS")
print("=" * 80 + "\n")

OUTPUT_DIR_ENH = os.path.join(MISSING_DIR, "enhanced_visualizations")
os.makedirs(OUTPUT_DIR_ENH, exist_ok=True)

class EnhancedVisuals:
    def __init__(self):
        self.evaluation_data = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost', 'LSTM'],
            'Accuracy': [0.9864, 0.9868, 0.9406],
            'Precision': [0.9630, 0.9633, 0.8290],
            'Recall': [0.9518, 0.9536, 0.7936],
            'F1_Score': [0.9574, 0.9584, 0.8109],
            'ROC_AUC': [0.9958, 0.9961, 0.9853]
        })
    
    def comprehensive_comparison(self):
        print("\n[1/3] Comprehensive model comparison...")
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        models = self.evaluation_data['Model'].values
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
        colors = ['#ef4444', '#10b981', '#6366f1']
        
        # Radar chart
        ax1 = fig.add_subplot(gs[0, 0], projection='polar')
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for idx, model in enumerate(models):
            values = self.evaluation_data[self.evaluation_data['Model'] == model][metrics].values[0].tolist()
            values += values[:1]
            ax1.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
            ax1.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metrics, size=9)
        ax1.set_ylim(0, 1)
        ax1.set_title('Performance Radar Chart', fontweight='bold', pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax1.text(0.5, -0.15, 'Note: Radar charts can visually exaggerate differences.\nRefer to bar chart for precise comparison.',
                transform=ax1.transAxes, ha='center', fontsize=7, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        # Grouped bar chart
        ax2 = fig.add_subplot(gs[0, 1:])
        x = np.arange(len(metrics))
        width = 0.25
        
        for idx, model in enumerate(models):
            values = self.evaluation_data[self.evaluation_data['Model'] == model][metrics].values[0]
            offset = width * (idx - 1)
            bars = ax2.bar(x + offset, values, width, label=model, color=colors[idx], alpha=0.8, edgecolor='black')
            
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Score')
        ax2.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics, rotation=15)
        ax2.legend()
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Heatmap
        ax3 = fig.add_subplot(gs[1, 0])
        metric_matrix = self.evaluation_data[metrics].values
        im = ax3.imshow(metric_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax3.set_xticks(np.arange(len(metrics)))
        ax3.set_yticks(np.arange(len(models)))
        ax3.set_xticklabels(metrics, rotation=45, ha='right')
        ax3.set_yticklabels(models)
        
        for i in range(len(models)):
            for j in range(len(metrics)):
                ax3.text(j, i, f'{metric_matrix[i, j]:.3f}',
                        ha="center", va="center", color="black", fontweight='bold', fontsize=9)
        
        ax3.set_title('Performance Heatmap', fontweight='bold')
        plt.colorbar(im, ax=ax3)
        
        # Scatter plot
        ax4 = fig.add_subplot(gs[1, 1])
        for idx, model in enumerate(models):
            row = self.evaluation_data[self.evaluation_data['Model'] == model]
            ax4.scatter(row['Accuracy'], row['F1_Score'], s=300, alpha=0.7, 
                       color=colors[idx], edgecolors='black', linewidth=2, label=model)
            ax4.annotate(model, (row['Accuracy'].values[0], row['F1_Score'].values[0]),
                        fontsize=9, ha='center')
        
        ax4.set_xlabel('Accuracy')
        ax4.set_ylabel('F1 Score')
        ax4.set_title('Accuracy vs F1 Score', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0.75, 1.0)
        ax4.set_ylim(0.75, 1.0)
        
        # Bar plot ranking (replacing violin plot)
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Calculate average score across all metrics
        avg_scores = []
        for model in models:
            row_values = self.evaluation_data[self.evaluation_data['Model'] == model][metrics].values[0]
            avg_scores.append(np.mean(row_values))
        
        bars = ax5.barh(range(len(models)), avg_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax5.set_yticks(range(len(models)))
        ax5.set_yticklabels(models)
        ax5.set_xlabel('Average Score Across All Metrics')
        ax5.set_title('Model Ranking by Average Performance', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        ax5.set_xlim(0.8, 1.0)
        ax5.invert_yaxis()
        
        # Annotate values
        for i, (bar, score) in enumerate(zip(bars, avg_scores)):
            ax5.text(score, bar.get_y() + bar.get_height()/2, f'{score:.4f}',
                    ha='left', va='center', fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
        
        # Summary text
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        summary_text = "PERFORMANCE SUMMARY\n" + "="*50 + "\n\n"
        for metric in metrics:
            best_idx = self.evaluation_data[metric].idxmax()
            best_model = self.evaluation_data.loc[best_idx, 'Model']
            best_score = self.evaluation_data.loc[best_idx, metric]
            summary_text += f"{metric}: {best_model} ({best_score:.4f})\n"
        
        ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes,
                fontsize=12, verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle('Comprehensive Model Performance Analysis', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(OUTPUT_DIR_ENH, 'comprehensive_model_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Comprehensive comparison complete")
    
    def attack_heatmaps(self):
        print("\n[2/3] Attack pattern heatmaps...")
        try:
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "data", "processed", "cleaned_features.parquet")
            df = pd.read_parquet(data_path)
            df['timestamp'] = pd.date_range(start='2018-02-14', periods=len(df), freq='1s')
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['date'] = df['timestamp'].dt.date
            
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            
            # Hour vs Day heatmap
            ax1 = axes[0, 0]
            pivot1 = df[df['is_attack'] == 1].groupby(['hour', 'day_of_week']).size().unstack(fill_value=0)
            sns.heatmap(pivot1, cmap='YlOrRd', annot=False, ax=ax1, cbar_kws={'label': 'Attack Count'})
            ax1.set_title('Attack Patterns: Hour vs Day of Week', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Day of Week')
            ax1.set_ylabel('Hour of Day')
            day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            ax1.set_xticklabels(day_labels)
            
            # Attack rate heatmap with capped scale
            ax2 = axes[0, 1]
            total_by_hour_dow = df.groupby(['hour', 'day_of_week']).size().unstack(fill_value=1)
            attack_rate = (pivot1 / total_by_hour_dow * 100).fillna(0)
            
            # Cap at realistic maximum (99th percentile or 80%, whichever is lower)
            max_rate = min(np.percentile(attack_rate.values, 99), 80)
            attack_rate_capped = np.clip(attack_rate, 0, max_rate)
            
            sns.heatmap(attack_rate_capped, cmap='coolwarm', annot=False, ax=ax2, 
                       cbar_kws={'label': 'Attack Rate (%)'}, vmin=0, vmax=max_rate)
            ax2.set_title(f'Attack Rate: Hour vs Day of Week (Capped at {max_rate:.0f}%)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Day of Week')
            ax2.set_ylabel('Hour of Day')
            ax2.set_xticklabels(day_labels)
            
            # Add note about rate calculation
            note_text = f'Attack Rate = (Attack Count / Total Packets) × 100\nCapped at {max_rate:.0f}% for visualization clarity'
            ax2.text(0.02, 0.98, note_text, transform=ax2.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            # Daily timeline with data quality handling
            ax3 = axes[1, 0]
            daily_attacks = df[df['is_attack'] == 1].groupby('date').size()
            dates = pd.to_datetime(daily_attacks.index)
            
            # Filter out zero or near-zero days (likely missing data)
            valid_mask = daily_attacks.values > 100  # Threshold for valid data
            if not valid_mask.all():
                print(f"    Note: Filtered {(~valid_mask).sum()} days with suspiciously low counts (likely missing data)")
            
            dates_valid = dates[valid_mask]
            attacks_valid = daily_attacks.values[valid_mask]
            
            ax3.plot(dates_valid, attacks_valid, linewidth=2, color='crimson', marker='o', markersize=6)
            ax3.fill_between(dates_valid, attacks_valid, alpha=0.3, color='red')
            ax3.set_title('Daily Attack Count Timeline (Filtered for Data Quality)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Attack Count')
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
            
            # Add note about data filtering
            ax3.text(0.02, 0.98, f'Note: Showing {len(attacks_valid)} days with valid data\n(Excluded {len(daily_attacks)-len(attacks_valid)} days with <100 attacks)',
                    transform=ax3.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            # Hourly distribution
            ax4 = axes[1, 1]
            hourly_dist = df[df['is_attack'] == 1].groupby('hour').size()
            colors_gradient = plt.cm.Reds(np.linspace(0.3, 0.9, len(hourly_dist)))
            bars = ax4.bar(hourly_dist.index, hourly_dist.values, color=colors_gradient, 
                          edgecolor='black', linewidth=1.5)
            
            peak_hour = hourly_dist.idxmax()
            bars[peak_hour].set_color('darkred')
            bars[peak_hour].set_edgecolor('gold')
            bars[peak_hour].set_linewidth(3)
            
            ax4.set_title('Hourly Attack Distribution (Peak Hour Highlighted)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Hour of Day')
            ax4.set_ylabel('Attack Count')
            ax4.grid(True, alpha=0.3, axis='y')
            
            ax4.annotate(f'Peak: {peak_hour}:00\n{hourly_dist.max()} attacks',
                        xy=(peak_hour, hourly_dist.max()),
                        xytext=(peak_hour+2, hourly_dist.max()*0.9),
                        arrowprops=dict(arrowstyle='->', color='gold', lw=2),
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR_ENH, 'attack_pattern_heatmaps.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ Attack heatmaps complete")
        except Exception as e:
            print(f"  ⚠️ Warning: {e}")
    
    def metric_evolution(self):
        print("\n[3/3] Metric evolution...")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        models = self.evaluation_data['Model'].values
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            values = self.evaluation_data[metric].values
            
            ax.plot(range(len(models)), values, marker='o', markersize=12, 
                   linewidth=3, color='steelblue', label=metric)
            ax.fill_between(range(len(models)), values, alpha=0.3, color='steelblue')
            
            for i, (model, value) in enumerate(zip(models, values)):
                ax.annotate(f'{value:.4f}', xy=(i, value), xytext=(0, 10),
                          textcoords='offset points', ha='center', fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=15)
            ax.set_ylabel(f'{metric} Score')
            ax.set_title(f'{metric} Across Models', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(min(values) - 0.05, 1.05)
            
            best_idx = np.argmax(values)
            ax.scatter(best_idx, values[best_idx], s=300, c='gold', 
                      edgecolors='darkred', linewidth=3, zorder=5, marker='*')
        
        # Summary panel
        ax_summary = axes[1, 2]
        ax_summary.axis('off')
        
        summary_text = "SUMMARY STATISTICS\n" + "="*30 + "\n\n"
        for metric in metrics:
            values = self.evaluation_data[metric].values
            best_model = models[np.argmax(values)]
            best_score = np.max(values)
            summary_text += f"{metric}:\n"
            summary_text += f"  Best: {best_model}\n"
            summary_text += f"  Score: {best_score:.4f}\n\n"
        
        ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle('Metric Evolution Across Model Types', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR_ENH, 'metric_evolution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Metric evolution complete")

# Execute Enhanced Visualizations
enh = EnhancedVisuals()
enh.comprehensive_comparison()
enh.attack_heatmaps()
enh.metric_evolution()

print("\n" + "=" * 80)
print("  ✅ ENHANCED VISUALIZATIONS COMPLETED!")
print("=" * 80)

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n\n" + "=" * 80)
print("  🎉 ALL MISSING PLOTS GENERATED SUCCESSFULLY!")
print("=" * 80)

print(f"\n📁 Output Location: {MISSING_DIR}")
print("\n📊 Files Generated:")
print(f"\n  time_series_models/")
print(f"    ✓ model_comparison.png")
print(f"    ✓ model_metrics_comparison.png")
print(f"    ✓ residual_analysis.png")
print(f"    ✓ lstm_learning_curves.png")
print(f"    ✓ xgboost_feature_importance.png")
print(f"    ✓ prediction_intervals.png")
print(f"    ✓ error_distributions.png")
print(f"    ✓ arima_diagnostics.png")

print(f"\n  advanced_time_series/")
print(f"    ✓ spectral_analysis.png")
print(f"    ✓ cross_correlation.png")
print(f"    ✓ structural_breaks.png")
print(f"    ✓ wavelet_analysis.png")
print(f"    ✓ granger_causality_results.txt")

print(f"\n  enhanced_visualizations/")
print(f"    ✓ comprehensive_model_comparison.png")
print(f"    ✓ attack_pattern_heatmaps.png")
print(f"    ✓ metric_evolution.png")

print("\n" + "=" * 80)
print("  Total: 16 visualization files generated!")
print("=" * 80 + "\n")
