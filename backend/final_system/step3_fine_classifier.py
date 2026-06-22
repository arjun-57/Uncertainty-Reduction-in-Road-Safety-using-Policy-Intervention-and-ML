"""
STAGE 2B: FINE-GRAINED CLASSIFIER
This script trains the specialist models that only activate if the Stage 2A
Binary Gate flags an accident as "Severe".
It distinguishes between Serious (0) and Fatal (1).

We train two specialists:
1. BalancedRandomForest (Optimized for Fatal maximization)
2. Focal Loss MLP (Optimized for feature interactions and calibration)
"""
import os, sys, joblib, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.ensemble import BalancedRandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STAGE 2B: FINE-GRAINED CLASSIFIER (SERIOUS vs FATAL)")
print("=" * 70)

_THIS_FILE   = os.path.abspath(__file__)
_SYS_DIR     = os.path.dirname(_THIS_FILE)
DATA_DIR     = os.path.join(_SYS_DIR, 'data')
MODELS_DIR   = os.path.join(_SYS_DIR, 'models')
GRAPHS_DIR   = os.path.join(_SYS_DIR, 'graphs')
RESULTS_DIR  = os.path.join(_SYS_DIR, 'results')

print("  Loading specialized severe dataset...")
X_train_fine = np.load(os.path.join(DATA_DIR, 'X_train_fine.npy'))
y_train_fine = np.load(os.path.join(DATA_DIR, 'y_train_fine.npy'))
X_test_fine  = np.load(os.path.join(DATA_DIR, 'X_test_fine.npy'))
y_test_fine  = np.load(os.path.join(DATA_DIR, 'y_test_fine.npy'))

print(f"  Train shape: {X_train_fine.shape} | Test shape: {X_test_fine.shape}")


print("\n  [1/2] Training BalancedRandomForest...")
t0 = time.time()
rf_model = BalancedRandomForestClassifier(
    n_estimators=150, 
    random_state=42, 
    n_jobs=-1, 
    sampling_strategy='all'
)
rf_model.fit(X_train_fine, y_train_fine)
t1 = time.time()

y_pred_rf = rf_model.predict(X_test_fine)
print(f"   ✓ RF Training complete in {t1-t0:.1f}s")
print(f"   RF Accuracy: {accuracy_score(y_test_fine, y_pred_rf)*100:.2f}%")
print(f"   RF Macro F1: {f1_score(y_test_fine, y_pred_rf, average='macro'):.4f}")

joblib.dump(rf_model, os.path.join(MODELS_DIR, 'stage2b_fine_rf.pkl'))

print("\n  [2/2] Training Focal Loss MLP...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

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

cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train_fine)
cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)

Xtr_t = torch.tensor(X_train_fine, dtype=torch.float32)
ytr_t = torch.tensor(y_train_fine, dtype=torch.long)
Xte_t = torch.tensor(X_test_fine, dtype=torch.float32)

dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=2048, shuffle=True, num_workers=0)
fl = FocalLoss(gamma=2.5, weight=cw_tensor)
mlp_model = FineMLP(Xtr_t.shape[1], h=256).to(device)
opt = optim.Adam(mlp_model.parameters(), lr=5e-4, weight_decay=1e-4)

t0 = time.time()
for epoch in range(15):
    mlp_model.train()
    for Xb, yb in dl:
        Xb, yb = Xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = fl(mlp_model(Xb), yb)
        loss.backward()
        opt.step()
t1 = time.time()

mlp_model.eval()
mlp_preds = []
with torch.no_grad():
    for i in range(0, len(Xte_t), 2048):
        logits = mlp_model(Xte_t[i:i+2048].to(device))
        p = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
        mlp_preds.append(p)
y_pred_mlp = np.concatenate(mlp_preds)

print(f"   ✓ MLP Training complete in {t1-t0:.1f}s")
print(f"   MLP Accuracy: {accuracy_score(y_test_fine, y_pred_mlp)*100:.2f}%")
print(f"   MLP Macro F1: {f1_score(y_test_fine, y_pred_mlp, average='macro'):.4f}")

torch.save(mlp_model.state_dict(), os.path.join(MODELS_DIR, 'stage2b_fine_mlp.pth'))


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm_rf = confusion_matrix(y_test_fine, y_pred_rf)
cm_mlp = confusion_matrix(y_test_fine, y_pred_mlp)

for ax, cm, title in zip(axes, [cm_rf, cm_mlp], ['Balanced RF', 'Residual MLP (Focal)']):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Serious (0)', 'Fatal (1)'],
                yticklabels=['Serious (Actual)', 'Fatal (Actual)'], ax=ax)
    ax.set_title(title, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'stage2b_fine_cm_comparison.png'), dpi=150)
plt.close()

import json
with open(os.path.join(RESULTS_DIR, 'stage2b_fine_metrics.json'), 'w') as f:
    json.dump({
        'rf_f1_fatal': float(f1_score(y_test_fine, y_pred_rf, average=None)[1]),
        'mlp_f1_fatal': float(f1_score(y_test_fine, y_pred_mlp, average=None)[1])
    }, f)

print("\n  ✓ Stage 2B completed. Specialized Fine models saved.")
print("=" * 70)
