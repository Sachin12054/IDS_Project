"""
Simplified Model Evaluation Script
Evaluates trained models and generates visualizations
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, TIMESERIES_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output directory for plots
OUTPUT_DIR = os.path.join(os.path.dirname(MODELS_DIR), 'evaluation_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def load_data():
    """Load test data"""
    logger.info("Loading processed data...")
    
    # Load flow features
    flow_path = os.path.join(PROCESSED_DIR, "cleaned_features.parquet")
    df = pd.read_parquet(flow_path)
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Load LSTM sequences
    lstm_path = os.path.join(TIMESERIES_DIR, "lstm_sequences.npz")
    sequences = np.load(lstm_path, allow_pickle=True)
    X_seq = sequences['X']
    y_seq = sequences['y']
    logger.info(f"Loaded sequences shape: X={X_seq.shape}, y={y_seq.shape}")
    
    return df, X_seq, y_seq


def prepare_flow_data(df, test_size=0.2, scaler=None):
    """Prepare flow data for evaluation"""
    # Get feature columns from scaler if available
    if scaler is not None and hasattr(scaler, 'feature_names_in_'):
        feature_cols = list(scaler.feature_names_in_)
        # Filter to only columns that exist in df
        feature_cols = [col for col in feature_cols if col in df.columns]
    else:
        # Exclude non-feature columns
        exclude_cols = ['is_attack', 'Label', 'source_file', 'Timestamp', 'timestamp']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    X = df[feature_cols].copy()
    y = df['is_attack'].copy()
    
    # Handle infinite/NaN values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    return X_test, y_test, feature_cols


def evaluate_model(model_name, model, scaler, X_test, y_test):
    """Evaluate a single model"""
    logger.info(f"Evaluating {model_name}...")
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    if model_name == 'isolation_forest':
        y_pred_raw = model.predict(X_test_scaled)
        y_pred = (y_pred_raw == -1).astype(int)
        y_proba = -model.decision_function(X_test_scaled)
        # Normalize to 0-1 range
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
    else:
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc,
        'predictions': y_pred,
        'probabilities': y_proba,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return results


def evaluate_lstm(X_test, y_test, model_path, batch_size=64):
    """Evaluate LSTM model"""
    logger.info("Evaluating LSTM model...")
    
    input_size = X_test.shape[2]
    model = LSTMClassifier(input_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Create DataLoader
    X_tensor = torch.FloatTensor(X_test)
    test_dataset = TensorDataset(X_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Batched predictions
    y_proba_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch_X = batch[0].to(device)
            outputs = model(batch_X).cpu().numpy()
            y_proba_list.append(outputs)
    
    y_proba = np.vstack(y_proba_list).flatten()
    y_pred = (y_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc,
        'predictions': y_pred,
        'probabilities': y_proba,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return results


def plot_confusion_matrices(results_dict, output_dir):
    """Plot confusion matrices for all models"""
    n_models = len(results_dict)
    cols = 2
    rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5*rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (model_name, results) in enumerate(results_dict.items()):
        cm = results['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['Benign', 'Attack'],
                   yticklabels=['Benign', 'Attack'])
        axes[i].set_title(f'{model_name.replace("_", " ").title()}\nAccuracy: {results["accuracy"]:.4f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Hide empty subplots
    for i in range(n_models, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrices saved to {output_dir}")


def plot_roc_curves(results_dict, y_test, output_dir):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    for i, (model_name, results) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
        plt.plot(fpr, tpr, label=f"{model_name.replace('_', ' ').title()} (AUC = {results['auc_score']:.4f})",
                color=colors[i % len(colors)], linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curves saved to {output_dir}")


def plot_precision_recall_curves(results_dict, y_test, output_dir):
    """Plot Precision-Recall curves for all models"""
    plt.figure(figsize=(10, 8))
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    for i, (model_name, results) in enumerate(results_dict.items()):
        precision, recall, _ = precision_recall_curve(y_test, results['probabilities'])
        plt.plot(recall, precision, label=f"{model_name.replace('_', ' ').title()}",
                color=colors[i % len(colors)], linewidth=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves Comparison', fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Precision-Recall curves saved to {output_dir}")


def plot_metrics_comparison(results_dict, output_dir):
    """Plot metrics comparison bar chart"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
    model_names = list(results_dict.keys())
    
    data = {metric: [results_dict[m][metric] for m in model_names] for metric in metrics}
    
    x = np.arange(len(model_names))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, data[metric], width, label=metric.replace('_', ' ').title(), color=colors[i])
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in model_names], fontsize=10)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, metric in enumerate(metrics):
        for j, v in enumerate(data[metric]):
            ax.text(j + i*width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Metrics comparison saved to {output_dir}")


def create_evaluation_report(results_dict, output_dir):
    """Create and save evaluation report"""
    report_data = []
    for model_name, results in results_dict.items():
        report_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': f"{results['accuracy']:.4f}",
            'Precision': f"{results['precision']:.4f}",
            'Recall': f"{results['recall']:.4f}",
            'F1 Score': f"{results['f1_score']:.4f}",
            'ROC AUC': f"{results['auc_score']:.4f}"
        })
    
    report_df = pd.DataFrame(report_data)
    
    # Save to CSV
    report_df.to_csv(os.path.join(output_dir, 'evaluation_report.csv'), index=False)
    
    # Save as markdown (simple format without tabulate)
    with open(os.path.join(output_dir, 'evaluation_report.md'), 'w') as f:
        f.write("# Model Evaluation Report\n\n")
        f.write("## Performance Summary\n\n")
        
        # Create simple markdown table
        f.write("| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |\n")
        f.write("|-------|----------|-----------|--------|----------|----------|\n")
        for _, row in report_df.iterrows():
            f.write(f"| {row['Model']} | {row['Accuracy']} | {row['Precision']} | {row['Recall']} | {row['F1 Score']} | {row['ROC AUC']} |\n")
        
        f.write("\n\n## Generated Visualizations\n\n")
        f.write("- `confusion_matrices.png` - Confusion matrices for all models\n")
        f.write("- `roc_curves.png` - ROC curves comparison\n")
        f.write("- `precision_recall_curves.png` - Precision-Recall curves\n")
        f.write("- `metrics_comparison.png` - Bar chart comparing all metrics\n")
    
    return report_df


def main():
    """Main evaluation pipeline"""
    logger.info("=" * 60)
    logger.info("STARTING MODEL EVALUATION PIPELINE")
    logger.info("=" * 60)
    
    # Load data
    df, X_seq, y_seq = load_data()
    
    # Load scalers first to get feature columns
    rf_scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_rf.pkl"))
    xgb_scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_xgb.pkl"))
    
    # Prepare flow data for traditional ML evaluation
    X_test_flow, y_test_flow, feature_cols = prepare_flow_data(df, scaler=rf_scaler)
    logger.info(f"Test data prepared: {X_test_flow.shape}")
    
    # Prepare LSTM data
    _, X_test_lstm, _, y_test_lstm = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )
    logger.info(f"LSTM test data: {X_test_lstm.shape}")
    
    # Store all results
    all_results = {}
    
    # Evaluate Random Forest
    logger.info("\n" + "=" * 40)
    logger.info("Evaluating Random Forest")
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    all_results['random_forest'] = evaluate_model('random_forest', rf_model, rf_scaler, X_test_flow, y_test_flow)
    print(f"Random Forest - Accuracy: {all_results['random_forest']['accuracy']:.4f}, AUC: {all_results['random_forest']['auc_score']:.4f}")
    
    # Evaluate XGBoost
    logger.info("\n" + "=" * 40)
    logger.info("Evaluating XGBoost")
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))
    all_results['xgboost'] = evaluate_model('xgboost', xgb_model, xgb_scaler, X_test_flow, y_test_flow)
    print(f"XGBoost - Accuracy: {all_results['xgboost']['accuracy']:.4f}, AUC: {all_results['xgboost']['auc_score']:.4f}")
    
    # Evaluate LSTM
    logger.info("\n" + "=" * 40)
    logger.info("Evaluating LSTM")
    lstm_path = os.path.join(MODELS_DIR, "lstm_best.pth")
    all_results['lstm'] = evaluate_lstm(X_test_lstm, y_test_lstm, lstm_path)
    print(f"LSTM - Accuracy: {all_results['lstm']['accuracy']:.4f}, AUC: {all_results['lstm']['auc_score']:.4f}")
    
    # Generate visualizations
    logger.info("\n" + "=" * 40)
    logger.info("Generating Visualizations")
    
    # For flow models (RF, XGBoost), use flow test data
    flow_results = {k: v for k, v in all_results.items() if k != 'lstm'}
    plot_confusion_matrices(all_results, OUTPUT_DIR)
    plot_roc_curves(flow_results, y_test_flow, OUTPUT_DIR)
    plot_precision_recall_curves(flow_results, y_test_flow, OUTPUT_DIR)
    plot_metrics_comparison(all_results, OUTPUT_DIR)
    
    # Create report
    report_df = create_evaluation_report(all_results, OUTPUT_DIR)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(report_df.to_string(index=False))
    print("\n" + "=" * 60)
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    logger.info("Model evaluation completed successfully!")
    
    return all_results


if __name__ == "__main__":
    main()
