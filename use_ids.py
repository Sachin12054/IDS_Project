"""
=================================================================
INTRUSION DETECTION SYSTEM - USER GUIDE
=================================================================

HOW TO USE THIS PROJECT:

1. REAL-TIME NETWORK MONITORING (Production Use)
   - Connect to network traffic capture (Wireshark, tcpdump)
   - Extract flow features from packets
   - Feed to model → Get ATTACK/BENIGN prediction
   
2. ANALYZE CSV LOG FILES (This Demo)
   - Upload network flow CSV file
   - Model analyzes each row
   - Returns predictions for each flow

3. DASHBOARD MONITORING
   - Run dashboard for visual monitoring
   - See attack trends over time

=================================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

# Load trained model
MODEL_PATH = 'models/random_forest.pkl'
SCALER_PATH = 'models/scaler_rf.pkl'

def load_model():
    """Load the trained IDS model"""
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model not found! Run training first.")
        sys.exit(1)
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def get_feature_columns():
    """Get the feature columns used by the model"""
    # These are network flow features - same as training
    exclude_cols = ['is_attack', 'Label', 'Label_encoded', 'source_file', 'Timestamp', 'timestamp']
    
    # Load sample to get column names
    df = pd.read_parquet('data/processed/cleaned_features.parquet', columns=None)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
    return feature_cols

def predict_from_csv(csv_path):
    """
    Analyze a CSV file containing network flows
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file with network flow data
        
    Returns:
    --------
    DataFrame with predictions
    """
    print(f"\n📂 Loading file: {csv_path}")
    
    # Load model
    model, scaler = load_model()
    feature_cols = get_feature_columns()
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"   Found {len(df)} network flows to analyze")
    
    # Check if required columns exist
    available_cols = [col for col in feature_cols if col in df.columns]
    if len(available_cols) < 10:
        print(f"⚠️  Warning: Only {len(available_cols)} features found. Need at least 10.")
        print("   Make sure CSV has network flow features like:")
        print("   Dst_Port, Flow_Duration, Fwd_Pkt_Len_Mean, Protocol, etc.")
        return None
    
    # Prepare features
    X = df[available_cols].fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Add results to dataframe
    df['Prediction'] = ['ATTACK' if p == 1 else 'BENIGN' for p in predictions]
    df['Attack_Probability'] = probabilities
    df['Risk_Level'] = pd.cut(probabilities, 
                               bins=[0, 0.3, 0.7, 1.0], 
                               labels=['LOW', 'MEDIUM', 'HIGH'])
    
    return df

def analyze_results(df):
    """Print analysis of detection results"""
    print("\n" + "=" * 60)
    print("📊 DETECTION RESULTS")
    print("=" * 60)
    
    total = len(df)
    attacks = (df['Prediction'] == 'ATTACK').sum()
    benign = (df['Prediction'] == 'BENIGN').sum()
    
    print(f"\n📈 Summary:")
    print(f"   Total flows analyzed: {total}")
    print(f"   🔴 Attacks detected:  {attacks} ({attacks/total*100:.1f}%)")
    print(f"   🟢 Benign traffic:    {benign} ({benign/total*100:.1f}%)")
    
    # Risk breakdown
    print(f"\n⚠️  Risk Levels:")
    risk_counts = df['Risk_Level'].value_counts()
    for level in ['HIGH', 'MEDIUM', 'LOW']:
        if level in risk_counts.index:
            print(f"   {level}: {risk_counts[level]} flows")
    
    # Show high-risk flows
    high_risk = df[df['Attack_Probability'] > 0.9]
    if len(high_risk) > 0:
        print(f"\n🚨 HIGH CONFIDENCE ATTACKS ({len(high_risk)} detected):")
        print("-" * 60)
        cols_to_show = ['Dst_Port', 'Protocol', 'Attack_Probability', 'Prediction']
        cols_available = [c for c in cols_to_show if c in df.columns]
        if cols_available:
            print(high_risk[cols_available].head(10).to_string(index=False))

def demo_with_sample_data():
    """Demo using sample data from the dataset"""
    print("\n" + "=" * 60)
    print("🔬 DEMO: Analyzing Sample Network Traffic")
    print("=" * 60)
    
    model, scaler = load_model()
    
    # Load sample data
    df = pd.read_parquet('data/processed/cleaned_features.parquet')
    
    # Get feature columns
    exclude_cols = ['is_attack', 'Label', 'Label_encoded', 'source_file', 'Timestamp', 'timestamp']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
    
    # Sample 100 random flows
    sample = df.sample(100, random_state=42)
    X = sample[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Predict
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Results
    sample['Prediction'] = ['ATTACK' if p == 1 else 'BENIGN' for p in predictions]
    sample['Attack_Probability'] = probabilities
    actual_labels = sample['is_attack'].values
    
    # Calculate accuracy
    correct = (predictions == actual_labels).sum()
    
    print(f"\n📊 Results on 100 random samples:")
    print(f"   Correct predictions: {correct}/100 ({correct}%)")
    print(f"   Attacks detected: {predictions.sum()}")
    print(f"   Actual attacks: {actual_labels.sum()}")
    
    # Show some examples
    print("\n📋 Sample Predictions:")
    print("-" * 70)
    print(f"{'Dst_Port':<10} {'Protocol':<10} {'Actual':<10} {'Predicted':<12} {'Confidence'}")
    print("-" * 70)
    
    for i in range(min(15, len(sample))):
        row = sample.iloc[i]
        actual = 'ATTACK' if row['is_attack'] == 1 else 'BENIGN'
        pred = row['Prediction']
        prob = row['Attack_Probability']
        conf = prob if pred == 'ATTACK' else 1 - prob
        port = int(row['Dst_Port']) if 'Dst_Port' in row else 'N/A'
        proto = int(row['Protocol']) if 'Protocol' in row else 'N/A'
        mark = '✓' if actual == pred else '✗'
        print(f"{port:<10} {proto:<10} {actual:<10} {pred:<12} {conf*100:.1f}% {mark}")

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║          INTRUSION DETECTION SYSTEM (IDS)                         ║
║          Time Series + Machine Learning Based                     ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  WHAT THIS DOES:                                                  ║
║  ---------------                                                  ║
║  Analyzes network traffic to detect cyber attacks like:           ║
║  • DDoS (Distributed Denial of Service)                           ║
║  • Brute Force attacks                                            ║
║  • Port scanning                                                  ║
║  • Data infiltration/exfiltration                                 ║
║                                                                   ║
║  HOW TO USE:                                                      ║
║  -----------                                                      ║
║  1. Upload network flow CSV → Get attack predictions              ║
║  2. Run dashboard for real-time monitoring                        ║
║  3. Integrate with network capture tools (production)             ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    print("\nOptions:")
    print("  1. Demo with sample data (recommended)")
    print("  2. Analyze a CSV file")
    print("  3. Show required CSV format")
    print("  4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        demo_with_sample_data()
    
    elif choice == '2':
        csv_path = input("Enter path to CSV file: ").strip()
        if os.path.exists(csv_path):
            results = predict_from_csv(csv_path)
            if results is not None:
                analyze_results(results)
                # Save results
                output_path = csv_path.replace('.csv', '_predictions.csv')
                results.to_csv(output_path, index=False)
                print(f"\n💾 Results saved to: {output_path}")
        else:
            print(f"File not found: {csv_path}")
    
    elif choice == '3':
        print("\n📝 REQUIRED CSV FORMAT:")
        print("-" * 60)
        print("Your CSV file should have network flow features like:")
        print("""
Column Name          | Description
---------------------|----------------------------------------
Dst_Port             | Destination port number (e.g., 80, 443)
Protocol             | Protocol type (6=TCP, 17=UDP)
Flow_Duration        | Duration of flow in microseconds
Fwd_Pkt_Len_Mean     | Mean forward packet length
Bwd_Pkt_Len_Mean     | Mean backward packet length
Flow_Byts_s          | Flow bytes per second
Flow_Pkts_s          | Flow packets per second
Fwd_IAT_Mean         | Forward inter-arrival time mean
ACK_Flag_Cnt         | Count of ACK flags
SYN_Flag_Cnt         | Count of SYN flags
... and more

Example CSV row:
Dst_Port,Protocol,Flow_Duration,Fwd_Pkt_Len_Mean,...
443,6,1200000,512.5,...
        """)
        print("\nYou can generate this data using:")
        print("  • CICFlowMeter (converts PCAP to CSV)")
        print("  • Wireshark + tshark")
        print("  • Custom network capture scripts")
    
    else:
        print("Goodbye!")

if __name__ == "__main__":
    main()
