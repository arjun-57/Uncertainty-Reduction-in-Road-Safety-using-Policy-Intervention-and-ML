"""
STAGE 2A: BINARY SEVERITY GATE
This script trains the initial gate of the system:
"Is this a Severe accident (Serious/Fatal) or Slight?"

We use XGBoost, applying exact sample weights, targeting a strong Recall metric 
on Severe crashes so we pass as much context to the Fine Classifier as possible.
"""
import os, sys, joblib, time
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight

print("=" * 70)
print("STAGE 2A: BINARY SEVERITY GATE (SLIGHT vs SEVERE)")
print("=" * 70)


_THIS_FILE   = os.path.abspath(__file__)
_SYS_DIR     = os.path.dirname(_THIS_FILE)
DATA_DIR     = os.path.join(_SYS_DIR, 'data')
MODELS_DIR   = os.path.join(_SYS_DIR, 'models')
GRAPHS_DIR   = os.path.join(_SYS_DIR, 'graphs')
RESULTS_DIR  = os.path.join(_SYS_DIR, 'results')

print("  Loading global dataset...")
X_train = np.load(os.path.join(DATA_DIR, 'X_train_global.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train_global.npy'))
X_test  = np.load(os.path.join(DATA_DIR, 'X_test_global.npy'))
y_test  = np.load(os.path.join(DATA_DIR, 'y_test_global.npy'))


y_train_bin = (y_train > 0).astype(int)
y_test_bin  = (y_test > 0).astype(int)

severe_tr = y_train_bin.sum()
severe_te = y_test_bin.sum()
print(f"  Train: Slight={len(y_train_bin)-severe_tr:,} | Severe={severe_tr:,}")
print(f"  Test:  Slight={len(y_test_bin)-severe_te:,} | Severe={severe_te:,}")


print("  Computing binary sample weights...")
sw = compute_sample_weight('balanced', y_train_bin)


print("  Training XGBoost Binary Gate...")
t0 = time.time()
gate_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=150,
    max_depth=5,
    learning_rate=0.08,
    tree_method='hist',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
gate_model.fit(X_train, y_train_bin, sample_weight=sw)
t1 = time.time()
print(f"  ✓ Training completed in {t1-t0:.1f}s")

y_pred = gate_model.predict(X_test)
y_proba = gate_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test_bin, y_pred)
mac_f1 = f1_score(y_test_bin, y_pred, average='macro')
print(f"\n  [RESULTS: BINARY GATE]")
print(f"   Accuracy:  {acc*100:.2f}%")
print(f"   Macro F1:  {mac_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_bin, y_pred, target_names=['Slight', 'Severe']))


print("  Saving models and graphs...")

joblib.dump(gate_model, os.path.join(MODELS_DIR, 'stage2a_binary_xgb.pkl'))

cm = confusion_matrix(y_test_bin, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Slight (Predicted)', 'Severe (Predicted)'],
            yticklabels=['Slight (Actual)', 'Severe (Actual)'])
ax.set_title('Severity Gate Confusion Matrix\n(Slight vs Serious+Fatal)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'stage2a_binary_cm.png'), dpi=150)
plt.close()

import json
with open(os.path.join(RESULTS_DIR, 'stage2a_binary_metrics.json'), 'w') as f:
    json.dump({
        'acc': float(acc), 'macro_f1': float(mac_f1), 
        'recall_severe': float(cm[1,1] / (cm[1,0] + cm[1,1]))
    }, f)

print("=" * 70)
