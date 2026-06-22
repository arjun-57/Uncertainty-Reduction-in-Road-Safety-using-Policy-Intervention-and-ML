"""
FINAL SYSTEM METRICS GENERATOR
This script loads the pre-configured arrays and pre-trained models from the 
Risk Intelligence System stages (1-5) and evaluates them sequentially.
It saves the complete metric outputs (Accuracies, F1 scores, Classification Reports)
to a text file for documentation purposes.
"""
import os, sys, joblib, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

_THIS_FILE   = os.path.abspath(__file__)
_SYS_DIR     = os.path.dirname(_THIS_FILE)
DATA_DIR     = os.path.join(_SYS_DIR, 'data')
MODELS_DIR   = os.path.join(_SYS_DIR, 'models')
RESULTS_DIR  = os.path.join(_SYS_DIR, 'results')

REPORT_FILE = os.path.join(_SYS_DIR, 'FULL_EXECUTION_RESULTS.txt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout_p=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(x + self.net(x))

class FineMLP(nn.Module):
    def __init__(self, in_d, h=256, n_classes=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.ReLU())
        self.blocks = nn.Sequential(ResidualMLPBlock(h), ResidualMLPBlock(h))
        self.head = nn.Linear(h, n_classes)
    def forward(self, x): return self.head(self.blocks(self.proj(x)))

class SimpleGlobalMLP(nn.Module):
    def __init__(self, in_dim, h=128, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, h//2), nn.BatchNorm1d(h//2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h//2, 3)
        )
    def forward(self, x): return self.net(x)

def redirect_to_string(func, *args, **kwargs):
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        func(*args, **kwargs)
    return f.getvalue()

output_log = []
def log(msg):
    print(msg)
    output_log.append(msg)

log("=" * 70)
log("  AI RISK INTELLIGENCE SYSTEM — FULL METRICS EVALUATION")
log("=" * 70)


log("\n[1] Loading Test Data...")
X_test_global = np.load(os.path.join(DATA_DIR, 'X_test_global.npy'))
y_test_global = np.load(os.path.join(DATA_DIR, 'y_test_global.npy'))
X_test_fine   = np.load(os.path.join(DATA_DIR, 'X_test_fine.npy'))
y_test_fine   = np.load(os.path.join(DATA_DIR, 'y_test_fine.npy'))


log("\n" + "-"*50)
log("STAGE 2A: BINARY SEVERITY GATE (Slight vs Severe)")
log("-" * 50)
y_test_bin = (y_test_global > 0).astype(int)
gate_model = joblib.load(os.path.join(MODELS_DIR, 'stage2a_binary_xgb.pkl'))
y_pred_gate = gate_model.predict(X_test_global)
y_proba_gate = gate_model.predict_proba(X_test_global)[:, 1]

log(f"   Accuracy: {accuracy_score(y_test_bin, y_pred_gate)*100:.2f}%")
log(f"   Macro F1: {f1_score(y_test_bin, y_pred_gate, average='macro'):.4f}")
log("\nClassification Report:")
log(classification_report(y_test_bin, y_pred_gate, target_names=['Slight (0)', 'Severe (1)']))


log("-" * 50)
log("STAGE 2B: FINE-GRAINED CLASSIFIER (Serious vs Fatal)")
log("-" * 50)
rf_fine = joblib.load(os.path.join(MODELS_DIR, 'stage2b_fine_rf.pkl'))
y_pred_rf = rf_fine.predict(X_test_fine)
rf_cm = confusion_matrix(y_test_fine, y_pred_rf)

mlp_fine = FineMLP(X_test_fine.shape[1], h=256).to(device)
mlp_fine.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'stage2b_fine_mlp.pth'), map_location=device))
mlp_fine.eval()
mlp_preds = []
with torch.no_grad():
    X_tes_t = torch.tensor(X_test_fine, dtype=torch.float32).to(device)
    p = torch.softmax(mlp_fine(X_tes_t), dim=1).argmax(dim=1).cpu().numpy()
    mlp_preds = p

log(f"   Balanced RF Accuracy: {accuracy_score(y_test_fine, y_pred_rf)*100:.2f}%")
log(f"   Balanced RF F1-Fatal: {f1_score(y_test_fine, y_pred_rf, average=None)[1]:.4f}")
log("\n   Balanced RF Classification Report:")
log(classification_report(y_test_fine, y_pred_rf, target_names=['Serious (0)', 'Fatal (1)']))

log(f"\n   Focal MLP Accuracy: {accuracy_score(y_test_fine, mlp_preds)*100:.2f}%")
log(f"   Focal MLP F1-Fatal: {f1_score(y_test_fine, mlp_preds, average=None)[1]:.4f}")
log("\n   Focal MLP Classification Report:")
log(classification_report(y_test_fine, mlp_preds, target_names=['Serious (0)', 'Fatal (1)']))


log("-" * 50)
log("STAGE 3: EPISTEMIC UNCERTAINTY (Deep Ensemble)")
log("-" * 50)
N_ENSEMBLE = 3
ensemble = []
for i in range(N_ENSEMBLE):
    m = SimpleGlobalMLP(X_test_global.shape[1]).to(device)
    m.load_state_dict(torch.load(os.path.join(MODELS_DIR, f'stage3_ens_mlp_{i}.pth'), map_location=device))
    m.eval()
    ensemble.append(m)

X_te_t_global = torch.tensor(X_test_global, dtype=torch.float32).to(device)
all_probas = []
with torch.no_grad():
    for m in ensemble:
        all_probas.append(torch.softmax(m(X_te_t_global), dim=1).cpu().numpy())

probas_stack = np.stack(all_probas)
mean_probas = probas_stack.mean(axis=0)
epistemic = probas_stack.var(axis=0).mean(axis=1)
y_pred_ens = mean_probas.argmax(axis=1)
correct = (y_pred_ens == y_test_global).astype(int)

df_u = pd.DataFrame({'Correct': correct, 'Epistemic': epistemic})
mean_u_corr = df_u[df_u['Correct']==1]['Epistemic'].mean()
mean_u_incorr = df_u[df_u['Correct']==0]['Epistemic'].mean()

log(f"   Average Epistemic Uncertainty (When Model is Correct):   {mean_u_corr:.6f}")
log(f"   Average Epistemic Uncertainty (When Model is Incorrect): {mean_u_incorr:.6f}")
log(f"   Ratio (Incorrect/Correct): {mean_u_incorr/(mean_u_corr+1e-8):.2f}x (Higher is better)")


log("\n" + "-" * 50)
log("STAGE 4: INTELLIGENT FUSION ENGINE")
log("-" * 50)

is_severe = y_proba_gate >= 0.50
y_pred_fused = np.full(len(X_test_global), -1)
y_pred_fused[~is_severe] = 0

severe_indices = np.where(is_severe)[0]
if len(severe_indices) > 0:
    X_test_severe = X_test_global[severe_indices]
    y_proba_rf_fatal = rf_fine.predict_proba(X_test_severe)[:, 1]
    with torch.no_grad():
        X_tes_sev = torch.tensor(X_test_severe, dtype=torch.float32).to(device)
        y_proba_mlp_fatal = torch.softmax(mlp_fine(X_tes_sev), dim=1)[:, 1].cpu().numpy()
        
    RF_FATAL_THRESH = 0.45
    y_severe_pred = np.zeros(len(severe_indices))
    fatal_mask = y_proba_rf_fatal >= RF_FATAL_THRESH
    mlp_decisions = np.where(y_proba_mlp_fatal >= 0.5, 2, 1)
    
    y_severe_pred[fatal_mask]  = 2
    y_severe_pred[~fatal_mask] = mlp_decisions[~fatal_mask]
    y_pred_fused[severe_indices] = y_severe_pred

cls_f1s = f1_score(y_test_global, y_pred_fused, average=None)
log(f"   Fused Accuracy:    {accuracy_score(y_test_global, y_pred_fused)*100:.2f}%")
log(f"   Fused Macro F1:    {f1_score(y_test_global, y_pred_fused, average='macro'):.4f}")
log(f"   F1-Slight:         {cls_f1s[0]:.4f}")
log(f"   F1-Serious:        {cls_f1s[1]:.4f}")
log(f"   F1-Fatal:          {cls_f1s[2]:.4f}")
log("\nCore Fusion Classification Report:")
log(classification_report(y_test_global, y_pred_fused, target_names=['Slight', 'Serious', 'Fatal']))

with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write("\n".join(output_log))

print(f"\n[!] Success. Complete log successfully dumped to:")
print(f"    {REPORT_FILE}")
