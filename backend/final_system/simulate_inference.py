import os, sys, joblib, time
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print(" PRODUCTION INFERENCE SIMULATION: AI RISK INTELLIGENCE SYSTEM ")
print("=" * 70)


_THIS_FILE   = os.path.abspath(__file__)
_SYS_DIR     = os.path.dirname(_THIS_FILE)
MODELS_DIR   = os.path.join(_SYS_DIR, 'models')


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


print("[1] Loading Final Predictive Models...")
try:
    gate_model = joblib.load(os.path.join(MODELS_DIR, 'stage2a_binary_xgb.pkl'))
    rf_fine    = joblib.load(os.path.join(MODELS_DIR, 'stage2b_fine_rf.pkl'))
    print("    -> Loaded XGBoost Gate & RandomForest Specialist.")
except Exception as e:
    print(f"Error loading tree models: {e}")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_FEATURES = 31

mlp_fine = FineMLP(NUM_FEATURES, h=256).to(device)
try:
    mlp_fine.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'stage2b_fine_mlp.pth'), map_location=device))
    mlp_fine.eval()
    print(f"    -> Loaded Deep Focal MLP Specialist on {device}.")
except Exception as e:
    print(f"Error loading MLP model: {e}")
    sys.exit(1)


print("\n[2] Listening for new real-time incidents...")

NUM_INCIDENTS = 1500
np.random.seed(42)
X_incoming = np.random.randn(NUM_INCIDENTS, NUM_FEATURES)

print(f"    -> Intercepted {NUM_INCIDENTS} new incident data-points (simulated).")

print("\n[3] Executing Intelligence Fusion Routing...")
t0 = time.time()


y_proba_gate = gate_model.predict_proba(X_incoming)[:, 1]
is_severe    = y_proba_gate >= 0.50

y_pred_fused = np.full(NUM_INCIDENTS, -1)

y_pred_fused[~is_severe] = 0

severe_indices = np.where(is_severe)[0]


if len(severe_indices) > 0:
    X_severe = X_incoming[severe_indices]
    

    y_proba_rf_fatal = rf_fine.predict_proba(X_severe)[:, 1]
    
    with torch.no_grad():
        X_test_t = torch.tensor(X_severe, dtype=torch.float32).to(device)
        logits = mlp_fine(X_test_t)
        y_proba_mlp_fatal = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        
    RF_FATAL_THRESH = 0.45
    
    y_severe_pred = np.zeros(len(severe_indices))
    fatal_mask = y_proba_rf_fatal >= RF_FATAL_THRESH
    
    mlp_decisions = np.where(y_proba_mlp_fatal >= 0.5, 2, 1)
    
    y_severe_pred[fatal_mask]  = 2
    y_severe_pred[~fatal_mask] = mlp_decisions[~fatal_mask]

    y_pred_fused[severe_indices] = y_severe_pred

t1 = time.time()
print(f"    -> Fusion resolved in {(t1-t0)*1000:.2f} ms")

print("\n[4] FINAL RISK DISTRIBUTION SUMMARY")
print("-" * 50)
slight_count  = np.sum(y_pred_fused == 0)
serious_count = np.sum(y_pred_fused == 1)
fatal_count   = np.sum(y_pred_fused == 2)
error_count   = np.sum(y_pred_fused == -1)

print(f"  [o] Slight  (Class 0) : {slight_count} incidents")
print(f"  [o] Serious (Class 1) : {serious_count} incidents")
print(f"  [x] Fatal   (Class 2) : {fatal_count} incidents")

if error_count > 0:
    print(f"  [!] Routing Errors    : {error_count} unassigned")

print("-" * 50)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("\n[5] Generating Visualizations...")
GRAPHS_DIR = os.path.join(_SYS_DIR, 'graphs')
os.makedirs(GRAPHS_DIR, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Live Inference Distribution (N={NUM_INCIDENTS})', fontsize=16, fontweight='bold')

labels = ['Slight', 'Serious', 'Fatal']
counts = [slight_count, serious_count, fatal_count]
colors = ['#5cb85c', '#f0ad4e', '#d9534f']

sns.barplot(x=labels, y=counts, palette=colors, ax=ax1)
ax1.set_title('Absolute Incident Counts', fontweight='bold')
ax1.set_ylabel('Number of Incidents')
ax1.set_xlabel('Severity Classification')
for i, count in enumerate(counts):
    ax1.text(i, count + (max(counts)*0.02), str(count), ha='center', fontweight='bold')

wedges, texts, autotexts = ax2.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=[0.05, 0.05, 0.05][:len(labels)])
ax2.set_title('Proportional Risk Distribution', fontweight='bold')
for text in autotexts:
    text.set_color('white')
    text.set_weight('bold')

plt.tight_layout()
graph_path = os.path.join(GRAPHS_DIR, 'live_inference_distribution.png')
plt.savefig(graph_path, dpi=150)
plt.close()

print(f"    -> Saved visual representation to: {graph_path}")
print("=" * 70)
print("Pipeline Ready. System standing by for real-world deployment.")
print("=" * 70)
