"""
STAGE 4: INTELLIGENT FUSION ENGINE
This script integrates the Binary Gate (Stage 2A) and Fine-Grained Models (Stage 2B) 
into a unified, rule-based decision mechanism that honors the accuracy-vs-fatal tradeoff.

Logic:
1. P(Severe) < 0.5 --> Predict 0 (Slight)
2. If Severe:
    2a. Compute P(Fatal) from Balanced RF
    2b. Compute P(Fatal) from Focal MLP
    2c. If RF P(Fatal) > Threshold (e.g. 0.5) --> Predict 2 (Fatal) - Priority Safety Override
    2d. Else --> Predict 1 (Serious) using MLP stabilization
"""
import os, sys, joblib, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STAGE 4: INTELLIGENT FUSION ENGINE")
print("=" * 70)

_THIS_FILE   = os.path.abspath(__file__)
_SYS_DIR     = os.path.dirname(_THIS_FILE)
DATA_DIR     = os.path.join(_SYS_DIR, 'data')
MODELS_DIR   = os.path.join(_SYS_DIR, 'models')
GRAPHS_DIR   = os.path.join(_SYS_DIR, 'graphs')
RESULTS_DIR  = os.path.join(_SYS_DIR, 'results')


print("  Loading global test data...")
X_test = np.load(os.path.join(DATA_DIR, 'X_test_global.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test_global.npy'))

print("  Loading specialized models...")
gate_model = joblib.load(os.path.join(MODELS_DIR, 'stage2a_binary_xgb.pkl'))
rf_fine    = joblib.load(os.path.join(MODELS_DIR, 'stage2b_fine_rf.pkl'))

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout_p=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.net(x))

class FineMLP(nn.Module):
    def __init__(self, in_d, h=256, n_classes=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.ReLU())
        self.blocks = nn.Sequential(ResidualMLPBlock(h), ResidualMLPBlock(h))
        self.head = nn.Linear(h, n_classes)
    def forward(self, x):
        x = self.proj(x)
        x = self.blocks(x)
        return self.head(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mlp_fine = FineMLP(X_test.shape[1], h=256).to(device)
mlp_fine.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'stage2b_fine_mlp.pth'), map_location=device))
mlp_fine.eval()


print("\n  Executing Fusion Routing...")
t0 = time.time()


y_proba_gate = gate_model.predict_proba(X_test)[:, 1]
is_severe    = y_proba_gate >= 0.50


y_pred_fused = np.full(len(X_test), -1)


y_pred_fused[~is_severe] = 0


severe_indices = np.where(is_severe)[0]
if len(severe_indices) > 0:
    X_test_severe = X_test[severe_indices]
    

    y_proba_rf_fatal = rf_fine.predict_proba(X_test_severe)[:, 1]
    

    with torch.no_grad():
        X_tes_t = torch.tensor(X_test_severe, dtype=torch.float32).to(device)
        logits = mlp_fine(X_tes_t)
        y_proba_mlp_fatal = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        
   
    RF_FATAL_THRESH = 0.45
    
    y_severe_pred = np.zeros(len(severe_indices))
    

    fatal_mask = y_proba_rf_fatal >= RF_FATAL_THRESH
    

    mlp_decisions = np.where(y_proba_mlp_fatal >= 0.5, 2, 1)
    
    y_severe_pred[fatal_mask]  = 2
    y_severe_pred[~fatal_mask] = mlp_decisions[~fatal_mask]
    

    y_pred_fused[severe_indices] = y_severe_pred

t1 = time.time()


print(f"  ✓ Fusion execution complete in {t1-t0:.2f}s")
acc = accuracy_score(y_test, y_pred_fused)
mac_f1 = f1_score(y_test, y_pred_fused, average='macro')
weighted_f1 = f1_score(y_test, y_pred_fused, average='weighted')
cls_f1s = f1_score(y_test, y_pred_fused, average=None)

print(f"\n  [RESULTS: INTELLIGENT MULTI-STAGE FUSION]")
print(f"   Accuracy:    {acc*100:.2f}%")
print(f"   Macro F1:    {mac_f1:.4f}")
print(f"   Weighted F1: {weighted_f1:.4f}")
print(f"   F1-Slight:   {cls_f1s[0]:.4f}")
print(f"   F1-Serious:  {cls_f1s[1]:.4f}")
print(f"   F1-Fatal:    {cls_f1s[2]:.4f}")


print("\n  Class Report:")
print(classification_report(y_test, y_pred_fused, target_names=['Slight (0)', 'Serious (1)', 'Fatal (2)']))


cm = confusion_matrix(y_test, y_pred_fused)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Slight', 'Serious', 'Fatal'],
            yticklabels=['Slight', 'Serious', 'Fatal'])
ax.set_title('Fusion Engine Core Matrix\n(Binary Gate + RF Override + MLP Stabilizer)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'stage4_fusion_cm.png'), dpi=150)
plt.close()

import json
with open(os.path.join(RESULTS_DIR, 'stage4_fusion_metrics.json'), 'w') as f:
    json.dump({
        'acc': float(acc), 'macro_f1': float(mac_f1),
        'f1_fatal': float(cls_f1s[2])
    }, f)

print("=" * 70)
