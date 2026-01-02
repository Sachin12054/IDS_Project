"""
Model Evaluation Pipeline for IDS
Comprehensive evaluation of trained models with metrics and visualizations
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, TIMESERIES_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMClassifier(nn.Module):
    """LSTM model for time series classification"""
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
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
        out = lstm_out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        return out

class ModelEvaluator:
    def __init__(self, models_dir=MODELS_DIR):
        """Initialize model evaluator"""
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # Create output directory for plots
        self.output_dir = os.path.join(os.path.dirname(models_dir), 'evaluation_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Model Evaluator initialized with models from: {models_dir}")
    
    def load_models(self):
        """Load all trained models and scalers"""
        logger.info("Loading trained models...")
        
        # Load sklearn models
        rf_path = os.path.join(self.models_dir, "random_forest.pkl")
        if os.path.exists(rf_path):
            self.models['random_forest'] = joblib.load(rf_path)
            logger.info("Random Forest model loaded")
        
        # Load XGBoost model
        xgb_path = os.path.join(self.models_dir, "xgboost.pkl")
        if os.path.exists(xgb_path):
            self.models['xgboost'] = joblib.load(xgb_path)
            logger.info("XGBoost model loaded")
        
        # Load Isolation Forest
        iso_path = os.path.join(self.models_dir, "isolation_forest.pkl")
        if os.path.exists(iso_path):
            self.models['isolation_forest'] = joblib.load(iso_path)
            logger.info("Isolation Forest model loaded")
        
        # Load scalers
        for scaler_name in ['rf', 'xgb', 'isolation_forest']:
            scaler_path = os.path.join(self.models_dir, f"scaler_{scaler_name}.pkl")
            if os.path.exists(scaler_path):
                self.scalers[scaler_name] = joblib.load(scaler_path)
        
        # Load label encoder
        le_path = os.path.join(self.models_dir, "label_encoder.pkl")
        if os.path.exists(le_path):
            self.label_encoder = joblib.load(le_path)
        
        logger.info(f"Loaded {len(self.models)} models and {len(self.scalers)} scalers")
        
    def load_test_data(self, flow_features_path, ts_features_path=None):
        """Load test data for evaluation"""
        logger.info("Loading test data...")
        
        # Load flow-level features
        self.flow_test = pd.read_parquet(flow_features_path)
        logger.info(f"Flow test data shape: {self.flow_test.shape}")
        
        # Load time-series features if provided
        if ts_features_path and os.path.exists(ts_features_path):
            self.ts_test = pd.read_parquet(ts_features_path)
            logger.info(f"Time-series test data shape: {self.ts_test.shape}")
        else:
            self.ts_test = None
    
    def prepare_test_data(self, df, test_size=0.2):
        """Prepare test data (using last portion as test set)"""
        # Sort by timestamp if available
        if 'Timestamp' in df.columns:
            df = df.sort_values('Timestamp')
        elif 'window_start' in df.columns:
            df = df.sort_values('window_start')
        
        # Use last portion as test set
        test_start = int(len(df) * (1 - test_size))
        test_df = df.iloc[test_start:].copy()
        
        # Prepare features and labels
        label_column = 'Label' if 'Label' in test_df.columns else 'is_attack'
        if label_column not in test_df.columns:
            label_column = 'is_statistical_anomaly'
        
        drop_columns = ['source_file', 'Timestamp', 'row_id'] + \
                      [col for col in test_df.columns if col.startswith('window') or 'time' in col.lower()]
        feature_columns = [col for col in test_df.columns if col not in drop_columns + [label_column]]
        
        X_test = test_df[feature_columns].copy()
        y_test = test_df[label_column].copy()
        
        # Handle categorical labels
        if y_test.dtype == 'object' and hasattr(self, 'label_encoder'):
            y_test = self.label_encoder.transform(y_test)
        
        # Remove infinite and NaN values
        X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()
        y_test = y_test.iloc[X_test.index]
        
        return X_test, y_test
    
    def evaluate_sklearn_model(self, model_name, X_test, y_test):
        """Evaluate sklearn-based models"""
        logger.info(f"Evaluating {model_name}...")
        
        model = self.models[model_name]
        scaler_name = model_name.replace('random_forest', 'rf')
        
        # Scale features
        if scaler_name in self.scalers:
            X_test_scaled = self.scalers[scaler_name].transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Predictions
        if model_name == 'isolation_forest':
            # For anomaly detection
            y_pred = model.predict(X_test_scaled)
            y_pred_binary = (y_pred == -1).astype(int)  # -1 is anomaly
            y_pred_proba = -model.decision_function(X_test_scaled)  # Higher is more anomalous
        else:
            # For classification
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred_binary = y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        
        if len(np.unique(y_test)) > 1:  # Check if we have both classes
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            auc = None
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_score': auc,
            'predictions': y_pred_binary,
            'probabilities': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred_binary),
            'classification_report': classification_report(y_test, y_pred_binary, output_dict=True)
        }
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f if auc else 'N/A'}")
        
        return self.results[model_name]
    
    def evaluate_xgboost_model(self, X_test, y_test):
        """Evaluate XGBoost model"""
        logger.info("Evaluating XGBoost...")
        
        model = self.models['xgboost']
        
        # Scale features
        if 'xgb' in self.scalers:
            X_test_scaled = self.scalers['xgb'].transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Create DMatrix
        dtest = xgb.DMatrix(X_test_scaled)
        
        # Predictions
        y_pred_proba = model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            auc = None
        
        # Store results
        self.results['xgboost'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_score': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"XGBoost - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f if auc else 'N/A'}")
        
        return self.results['xgboost']
    
    def evaluate_lstm_model(self, X_test, y_test):
        """Evaluate LSTM model"""
        logger.info("Evaluating LSTM...")
        
        model = self.models['lstm_classifier']
        
        # Predictions
        y_pred_proba = model.predict(X_test).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            auc = None
        
        # Store results
        self.results['lstm_classifier'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_score': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"LSTM - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f if auc else 'N/A'}")
        
        return self.results['lstm_classifier']
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        if n_models == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (model_name, results) in enumerate(self.results.items()):
            if i >= 4:  # Maximum 4 plots
                break
                
            cm = results['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name.replace("_", " ").title()} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, 4):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Confusion matrices plotted and saved")
    
    def plot_roc_curves(self, y_test):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            if results['auc_score'] is not None:
                fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
                plt.plot(fpr, tpr, label=f"{model_name.replace('_', ' ').title()} (AUC = {results['auc_score']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        plt.savefig(os.path.join(self.models_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("ROC curves plotted and saved")
    
    def plot_precision_recall_curves(self, y_test):
        """Plot Precision-Recall curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            if results['auc_score'] is not None:
                precision, recall, _ = precision_recall_curve(y_test, results['probabilities'])
                plt.plot(recall, precision, label=f"{model_name.replace('_', ' ').title()}")
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.models_dir, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Precision-Recall curves plotted and saved")
    
    def create_evaluation_report(self):
        """Create comprehensive evaluation report"""
        logger.info("Creating evaluation report...")
        
        # Create results summary
        summary_data = []
        for model_name, results in self.results.items():
            summary_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': results['accuracy'],
                'F1 Score': results['f1_score'],
                'AUC Score': results['auc_score'] if results['auc_score'] else 'N/A'
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create detailed report
        report = "# IDS Model Evaluation Report\n\n"
        report += "## Model Performance Summary\n\n"
        report += summary_df.to_markdown(index=False) + "\n\n"
        
        # Add detailed results for each model
        for model_name, results in self.results.items():
            report += f"## {model_name.replace('_', ' ').title()} Detailed Results\n\n"
            
            # Classification report
            class_report = results['classification_report']
            report += "### Classification Report\n\n"
            report += f"- **Precision (Class 0)**: {class_report['0']['precision']:.4f}\n"
            report += f"- **Recall (Class 0)**: {class_report['0']['recall']:.4f}\n"
            report += f"- **F1-Score (Class 0)**: {class_report['0']['f1-score']:.4f}\n"
            
            if '1' in class_report:
                report += f"- **Precision (Class 1)**: {class_report['1']['precision']:.4f}\n"
                report += f"- **Recall (Class 1)**: {class_report['1']['recall']:.4f}\n"
                report += f"- **F1-Score (Class 1)**: {class_report['1']['f1-score']:.4f}\n"
            
            report += f"- **Macro Avg F1**: {class_report['macro avg']['f1-score']:.4f}\n"
            report += f"- **Weighted Avg F1**: {class_report['weighted avg']['f1-score']:.4f}\n\n"
            
            # Confusion matrix
            cm = results['confusion_matrix']
            report += "### Confusion Matrix\n\n"
            report += "```\n"
            report += f"[[{cm[0,0]}, {cm[0,1]}],\n"
            report += f" [{cm[1,0]}, {cm[1,1]}]]\n"
            report += "```\n\n"
        
        # Save report
        report_path = os.path.join(self.models_dir, 'evaluation_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to: {report_path}")
        
        return summary_df
    
    def save_results(self):
        """Save evaluation results to JSON"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, results in self.results.items():
            serializable_results[model_name] = {
                'accuracy': float(results['accuracy']),
                'f1_score': float(results['f1_score']),
                'auc_score': float(results['auc_score']) if results['auc_score'] else None,
                'confusion_matrix': results['confusion_matrix'].tolist(),
                'classification_report': results['classification_report']
            }
        
        results_path = os.path.join(self.models_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")

def main():
    """Main evaluation pipeline"""
    try:
        models_dir = "../models"
        
        # Initialize evaluator
        evaluator = ModelEvaluator(models_dir)
        
        logger.info("=== STARTING MODEL EVALUATION ===")
        
        # Load models
        evaluator.load_models()
        
        if len(evaluator.models) == 0:
            logger.error("No trained models found. Please train models first.")
            return
        
        # Load test data
        flow_features_path = "../data/processed/engineered_features.parquet"
        ts_features_path = "../data/time_series/time_series_features.parquet"
        
        evaluator.load_test_data(flow_features_path, ts_features_path)
        
        # Evaluate flow-based models
        if evaluator.flow_test is not None:
            logger.info("=== EVALUATING FLOW-BASED MODELS ===")
            X_test, y_test = evaluator.prepare_test_data(evaluator.flow_test)
            
            # Evaluate sklearn models
            for model_name in ['random_forest', 'isolation_forest']:
                if model_name in evaluator.models:
                    evaluator.evaluate_sklearn_model(model_name, X_test, y_test)
            
            # Evaluate XGBoost
            if 'xgboost' in evaluator.models:
                evaluator.evaluate_xgboost_model(X_test, y_test)
        
        # Evaluate time-series models
        if evaluator.ts_test is not None and 'lstm_classifier' in evaluator.models:
            logger.info("=== EVALUATING TIME-SERIES MODELS ===")
            X_test_ts, y_test_ts = evaluator.prepare_test_data(evaluator.ts_test)
            
            # Prepare LSTM data (simplified version)
            sequence_length = 10
            if len(X_test_ts) > sequence_length:
                # Create simple sequences
                X_lstm = []
                y_lstm = []
                for i in range(sequence_length, len(X_test_ts)):
                    X_lstm.append(X_test_ts.iloc[i-sequence_length:i].values)
                    y_lstm.append(y_test_ts.iloc[i])
                
                X_lstm = np.array(X_lstm)
                y_lstm = np.array(y_lstm)
                
                evaluator.evaluate_lstm_model(X_lstm, y_lstm)
        
        # Create visualizations and reports
        logger.info("=== CREATING EVALUATION REPORTS ===")
        
        if evaluator.results:
            evaluator.plot_confusion_matrices()
            
            # Use the last y_test for plotting (flow-based)
            if 'flow_test' in dir(evaluator) and evaluator.flow_test is not None:
                X_test, y_test = evaluator.prepare_test_data(evaluator.flow_test)
                evaluator.plot_roc_curves(y_test)
                evaluator.plot_precision_recall_curves(y_test)
            
            summary_df = evaluator.create_evaluation_report()
            evaluator.save_results()
            
            print("\n=== MODEL PERFORMANCE SUMMARY ===")
            print(summary_df.to_string(index=False))
        
        logger.info("=== MODEL EVALUATION COMPLETED ===")
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()