"""
STAGE 3: DEEP UNCERTAINTY ENGINE
This script establishes the framework for reasoning about model confidence.
It utilizes a Deep Ensemble to explicitly decouple Epistemic Uncertainty 
(Model Confidence) from Aleatoric Uncertainty (Data Noise/Ambiguity).

We train a small ensemble of 3 MLPs on the global dataset to provide continuous outputs.
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
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STAGE 3: DEEP UNCERTAINTY & CALIBRATION ENGINE")
print("=" * 70)

_THIS_FILE   = os.path.abspath(__file__)
_SYS_DIR     = os.path.dirname(_THIS_FILE)
DATA_DIR     = os.path.join(_SYS_DIR, 'data')
MODELS_DIR   = os.path.join(_SYS_DIR, 'models')
GRAPHS_DIR   = os.path.join(_SYS_DIR, 'graphs')
RESULTS_DIR  = os.path.join(_SYS_DIR, 'results')


X_train = np.load(os.path.join(DATA_DIR, 'X_train_global.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train_global.npy'))
X_test  = np.load(os.path.join(DATA_DIR, 'X_test_global.npy'))
y_test  = np.load(os.path.join(DATA_DIR, 'y_test_global.npy'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleGlobalMLP(nn.Module):
    def __init__(self, in_dim, h=128, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, h//2), nn.BatchNorm1d(h//2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h//2, 3)
        )
    def forward(self, x): return self.net(x)

def train_ensemble_member(model, X_tr, y_tr, seed, epochs=10):
    torch.manual_seed(seed)
    dl = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), 
                                  torch.tensor(y_tr, dtype=torch.long)), 
                    batch_size=2048, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    for _ in range(epochs):
        for Xb, yb in dl:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model(Xb), yb).backward()
            opt.step()
    return model

N_ENSEMBLE = 3
ensemble = []
print(f"  Training Deep Ensemble ({N_ENSEMBLE} models) for Epistemic Uncertainty tracking...")
t0 = time.time()
for i in range(N_ENSEMBLE):
    print(f"   > Training model {i+1}/{N_ENSEMBLE} (Seed {i*42})...")
    m = SimpleGlobalMLP(X_train.shape[1])
    m = train_ensemble_member(m, X_train, y_train, seed=i*42, epochs=8)
    ensemble.append(m)
    torch.save(m.state_dict(), os.path.join(MODELS_DIR, f'stage3_ens_mlp_{i}.pth'))
print(f"  ✓ Deep Ensemble trained in {time.time()-t0:.1f}s")


print("\n  Computing outputs and calibrating...")
X_te_t = torch.tensor(X_test, dtype=torch.float32)

all_probas = []
for m in ensemble:
    m.eval()
    p_batches = []
    with torch.no_grad():
        for i in range(0, len(X_te_t), 4096):
            logits = m(X_te_t[i:i+4096].to(device))
            p_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
    all_probas.append(np.concatenate(p_batches))


probas_stack = np.stack(all_probas)


mean_probas = probas_stack.mean(axis=0)

epistemic = probas_stack.var(axis=0).mean(axis=1)


aleatoric = 1.0 - mean_probas.max(axis=1)

y_pred_ens = mean_probas.argmax(axis=1)



print("  Correlating Uncertainty with Misclassifications...")
correct = (y_pred_ens == y_test).astype(int)

df_u = pd.DataFrame({
    'Correct': correct,
    'Epistemic': epistemic, 
    'Aleatoric': aleatoric,
    'True_Class': y_test
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.kdeplot(data=df_u, x='Epistemic', hue='Correct', fill=True, ax=axes[0], 
            palette={1: "green", 0: "red"}, common_norm=False)
axes[0].set_title('Epistemic Uncertainty (Model Doubt)')

sns.kdeplot(data=df_u, x='Aleatoric', hue='Correct', fill=True, ax=axes[1], 
            palette={1: "green", 0: "red"}, common_norm=False)
axes[1].set_title('Aleatoric Uncertainty (Data Noise)')

plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'stage3_uncertainty_distributions.png'), dpi=150)
plt.close()


mean_u_corr = df_u[df_u['Correct']==1]['Epistemic'].mean()
mean_u_incorr = df_u[df_u['Correct']==0]['Epistemic'].mean()
print(f"   Average Epistemic Uncertainty (Correct):   {mean_u_corr:.6f}")
print(f"   Average Epistemic Uncertainty (Incorrect): {mean_u_incorr:.6f}")
print(f"   Ratio (Incorrect/Correct): {mean_u_incorr/(mean_u_corr+1e-8):.2f}x (Higher is better)")

import json
with open(os.path.join(RESULTS_DIR, 'stage3_uncertainty_metrics.json'), 'w') as f:
    json.dump({
        'epi_correct': float(mean_u_corr), 
        'epi_incorrect': float(mean_u_incorr),
        'alea_correct': float(df_u[df_u['Correct']==1]['Aleatoric'].mean()),
        'alea_incorrect': float(df_u[df_u['Correct']==0]['Aleatoric'].mean()),
    }, f)

print("  ✓ Stage 3 completed. Uncertainty tracking artifacts saved.")
print("=" * 70)
