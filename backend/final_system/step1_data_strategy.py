"""
STAGE 1: HYBRID DATA STRATEGY & PIPELINE PREPARATION
This script establishes the foundational dataset for the multi-stage Risk Intelligence System.
It outputs:
1. Pure original split (for binary gate + calibration).
2. Specialized fine-grained subset (Serious vs Fatal) augmented specifically for the extreme minority.
"""
import os, sys, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTENC

print("=" * 70)
print("STAGE 1: HYBRID DATA STRATEGY")
print("=" * 70)

_THIS_FILE   = os.path.abspath(__file__)                     
_SYS_DIR     = os.path.dirname(_THIS_FILE)                   
_BACKEND_DIR = os.path.dirname(_SYS_DIR)                     
_PROJECT_DIR = os.path.dirname(_BACKEND_DIR)                 

BASE_DIR      = _PROJECT_DIR
DATASET_DIR   = os.path.join(BASE_DIR, 'daatset')
FINAL_SYS_DIR = _SYS_DIR

for d in ['data', 'models', 'graphs', 'results']:
    os.makedirs(os.path.join(FINAL_SYS_DIR, d), exist_ok=True)
print(f"  ✓ Directory structure initialized at {FINAL_SYS_DIR}")

eng_csv_path = os.path.join(DATASET_DIR, 'engineered_dataset_full.csv')
print(f"  Loading engineered dataset: {eng_csv_path}")
df = pd.read_csv(eng_csv_path)


df['collision_severity'] = df['collision_severity'].map({3: 0, 2: 1, 1: 2})
class_map = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}


feature_names = np.load(os.path.join(DATASET_DIR, 'feature_names.npy'), allow_pickle=True).tolist()
cat_indices   = np.load(os.path.join(DATASET_DIR, 'categorical_feature_indices.npy'), allow_pickle=True).tolist()
categorical_cols = [feature_names[i] for i in cat_indices]
numerical_cols   = [c for c in feature_names if c not in categorical_cols]


X = df[feature_names].copy()
y = df['collision_severity'].values


print("  Encoding categorical features...")
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

print("  Performing global stratified split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols]  = scaler.transform(X_test[numerical_cols])

print(f"\n[GLOBAL DATASET: {len(X_train)} Train | {len(X_test)} Test]")
for i in [0, 1, 2]:
    print(f"  {class_map[i]}: {(y_train==i).sum()}/{len(y_train)} ({(y_train==i).mean()*100:.1f}%)")


data_out_dir = os.path.join(FINAL_SYS_DIR, 'data')
np.save(os.path.join(data_out_dir, 'X_train_global.npy'), X_train_scaled.values.astype(np.float32))
np.save(os.path.join(data_out_dir, 'y_train_global.npy'), y_train.astype(int))
np.save(os.path.join(data_out_dir, 'X_test_global.npy'), X_test_scaled.values.astype(np.float32))
np.save(os.path.join(data_out_dir, 'y_test_global.npy'), y_test.astype(int))

joblib.dump(scaler, os.path.join(FINAL_SYS_DIR, 'models', 'global_scaler.pkl'))
joblib.dump(le_dict, os.path.join(FINAL_SYS_DIR, 'models', 'global_label_encoders.pkl'))


print("\n[SPECIALIZED FINE SUBSET: Serious vs Fatal ONLY]")

severe_mask_tr = y_train > 0 
severe_mask_te = y_test > 0

X_train_severe = X_train_scaled[severe_mask_tr].copy()
y_train_severe = y_train[severe_mask_tr].copy()
X_test_severe  = X_test_scaled[severe_mask_te].copy()
y_test_severe  = y_test[severe_mask_te].copy()


y_train_severe_bin = y_train_severe - 1
y_test_severe_bin  = y_test_severe - 1

print(f"  Pre-Augmentation Severe Set:")
print(f"   - Serious (0): {(y_train_severe_bin==0).sum()} ({np.mean(y_train_severe_bin==0)*100:.1f}%)")
print(f"   - Fatal   (1): {(y_train_severe_bin==1).sum()} ({np.mean(y_train_severe_bin==1)*100:.1f}%)")


from imblearn.over_sampling import SMOTE

print("  Applying Strategic SMOTE to Fatal class (boosting to 50% parity)...")
target_fatal_count = int((y_train_severe_bin==0).sum() * 0.5)


if target_fatal_count > (y_train_severe_bin==1).sum():
    smote = SMOTE(sampling_strategy={0: (y_train_severe_bin==0).sum(), 1: target_fatal_count}, random_state=42)
    X_train_severe_sm, y_train_severe_sm = smote.fit_resample(X_train_severe, y_train_severe_bin)
else:
    X_train_severe_sm, y_train_severe_sm = X_train_severe, y_train_severe_bin

print(f"  Post-Augmentation Severe Train Set (SMOTE):")
print(f"   - Serious (0): {(y_train_severe_sm==0).sum()} ({np.mean(y_train_severe_sm==0)*100:.1f}%)")
print(f"   - Fatal   (1): {(y_train_severe_sm==1).sum()} ({np.mean(y_train_severe_sm==1)*100:.1f}%)")


np.save(os.path.join(data_out_dir, 'X_train_fine.npy'), X_train_severe_sm.values.astype(np.float32))
np.save(os.path.join(data_out_dir, 'y_train_fine.npy'), y_train_severe_sm.astype(int))
np.save(os.path.join(data_out_dir, 'X_test_fine.npy'), X_test_severe.values.astype(np.float32))
np.save(os.path.join(data_out_dir, 'y_test_fine.npy'), y_test_severe_bin.astype(int))

print("\n  ✓ Stage 1 Data Strategy Complete. All arrays and transforms saved to final_system/data/")
print("=" * 70)
