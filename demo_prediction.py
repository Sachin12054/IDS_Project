"""
IDS Demo - Shows how the model predicts attacks
"""
import pandas as pd
import numpy as np
import joblib

print('=' * 60)
print('INTRUSION DETECTION SYSTEM - LIVE DEMO')
print('=' * 60)
print()

# Load model and scaler
rf = joblib.load('models/random_forest.pkl')
scaler = joblib.load('models/scaler_rf.pkl')

# Load some test data
df = pd.read_parquet('data/processed/cleaned_features.parquet')

# Get feature columns (same as training)
exclude_cols = ['is_attack', 'Label', 'Label_encoded', 'source_file', 'Timestamp', 'timestamp']
feature_cols = [col for col in df.columns if col not in exclude_cols]
feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]

# Get random samples (mix of attack and benign)
np.random.seed(123)
benign_samples = df[df['is_attack'] == 0].sample(5)
attack_samples = df[df['is_attack'] == 1].sample(5)
test_samples = pd.concat([benign_samples, attack_samples])

# Prepare features
X_test = test_samples[feature_cols].fillna(0)
X_test = X_test.replace([np.inf, -np.inf], 0)
X_scaled = scaler.transform(X_test)

# Predict
predictions = rf.predict(X_scaled)
probabilities = rf.predict_proba(X_scaled)[:, 1]
actual = test_samples['is_attack'].values

print('Testing on 10 random network flows:')
print()
print('-' * 65)
print(f"{'No.':<5}{'Actual':<12}{'Predicted':<12}{'Confidence':<15}{'Result'}")
print('-' * 65)

correct = 0
for i in range(len(predictions)):
    actual_label = 'ATTACK' if actual[i] == 1 else 'BENIGN'
    pred_label = 'ATTACK' if predictions[i] == 1 else 'BENIGN'
    conf = probabilities[i] if predictions[i] == 1 else 1 - probabilities[i]
    is_correct = actual[i] == predictions[i]
    if is_correct:
        correct += 1
    result = '✓ Correct' if is_correct else '✗ WRONG'
    print(f"{i+1:<5}{actual_label:<12}{pred_label:<12}{conf*100:.1f}%{'':<10}{result}")

print('-' * 65)
print(f"\nAccuracy on these samples: {correct}/{len(predictions)} = {correct/len(predictions)*100:.0f}%")

print('\n' + '=' * 65)
print('HOW IT WORKS:')
print('=' * 65)
print("""
INPUT (Network Flow Features):
┌─────────────────────────────────────────────────────────────┐
│  Dst_Port         = 443 (HTTPS)                             │
│  Flow_Duration    = 1200000 microseconds                    │
│  Fwd_Pkt_Len_Mean = 512 bytes                               │
│  Protocol         = 6 (TCP)                                 │
│  ACK_Flag_Cnt     = 15                                      │
│  ... 66 more features                                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  ML MODEL     │
                    │  (Random      │
                    │   Forest)     │
                    └───────────────┘
                            │
                            ▼
OUTPUT:
┌─────────────────────────────────────────────────────────────┐
│  Prediction: ATTACK or BENIGN                               │
│  Confidence: 98.5%                                          │
│  Action: Alert security team if ATTACK                      │
└─────────────────────────────────────────────────────────────┘
""")

print('=' * 65)
print('USE CASES:')
print('=' * 65)
print("""
1. CORPORATE NETWORK MONITORING
   - Monitor all incoming/outgoing traffic
   - Alert when attack patterns detected
   
2. DDOS DETECTION
   - Identify distributed denial of service attacks
   - Block malicious IPs automatically
   
3. BRUTE FORCE DETECTION
   - Detect password guessing attempts
   - Lock accounts after suspicious activity
   
4. DATA EXFILTRATION
   - Detect unusual data transfers
   - Prevent sensitive data leaks
""")
