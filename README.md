# Uncertainty Reduction in Road Safety Using Policy Intervention and Machine Learning

A research-grade machine learning and uncertainty quantification pipeline designed to predict road traffic accident severity, decompose predictive uncertainties, and simulate counterfactual policy interventions on the UK Road Safety Dataset (503,475 records).

---

## Overview

Road traffic accident severity prediction is traditionally treated as a standard multi-class classification problem. However, real-world deployment for policymaking requires two critical components usually absent from conventional models:
1. **Uncertainty Quantification**: Distinguishing between data noise (aleatoric uncertainty) and model ignorance (epistemic uncertainty) to prevent overconfident decisions in high-risk scenarios.
2. **Actionable Policy Interventions**: Evaluating counterfactual adjustments (e.g., speed reductions, illumination upgrades, surface treatments) to estimate measurable risk reductions before municipal resource allocation.

This framework implements an end-to-end 12-phase machine learning pipeline integrating balanced gradient boosting ensembles, Bayesian neural networks with Monte Carlo dropout, SHAP interpretability, spatial kernel density estimation, and a counterfactual policy simulation engine.

---

## Architecture and Pipeline

```
                                ┌─────────────────────────────────────────┐
                                │    UK Road Safety Dataset (503,475)     │
                                └────────────────────┬────────────────────┘
                                                     │
                                                     ▼
                                ┌─────────────────────────────────────────┐
                                │  Phase 1: Exploratory Analysis & Audit  │
                                │  Phase 2: Spatial & Cyclical Features   │
                                │  Phase 3: SMOTE-NC Resampling (926.8K)  │
                                └────────────────────┬────────────────────┘
                                                     │
                         ┌───────────────────────────┴───────────────────────────┐
                         ▼                                                       ▼
        ┌─────────────────────────────────┐                     ┌─────────────────────────────────┐
        │     Phase 4: Ensemble Model     │                     │     Phase 5: Bayesian BNN       │
        │  • XGBoost + LightGBM + CatBoost│                     │  • PyTorch Architecture         │
        │  • Soft Voting & CV Calibration │                     │  • 50 MC Dropout Passes         │
        │  • Test Accuracy: 78.5%         │                     │  • Epistemic vs Aleatoric Split │
        └────────────────┬────────────────┘                     └────────────────┬────────────────┘
                         │                                                       │
                         └───────────────────────────┬───────────────────────────┘
                                                     │
                                                     ▼
                                ┌─────────────────────────────────────────┐
                                │   Phase 6: SHAP Interpretability        │
                                │   Phase 7: Policy Intervention Engine   │
                                │   Phase 8: Spatial Kernel Density (KDE) │
                                │   Phase 9-12: Validation & API Wrapper  │
                                └─────────────────────────────────────────┘
```

---

## Key Methodology by Phase

| Phase | Component | Methodology and Technical Details |
|---|---|---|
| **Phase 1** | Exploratory Data Analysis | Class distribution profiling (Slight: 76.7%, Serious: 21.8%, Fatal: 1.5%), spatial bounding validation, and feature leakage audit. |
| **Phase 2** | Feature Engineering | BallTree KNN spatial blackspot density (200m radius), trigonometric cyclical encodings for hour/month/day, compound environmental hazard indices, and speed-road interaction features. |
| **Phase 3** | Class Balancing | Stratified 80/20 train-test partition with SMOTE-NC oversampling, balancing training data to 926,800 samples (308.8K per class). |
| **Phase 4** | Ensemble Classification | Soft-voting ensemble of XGBoost, LightGBM, and CatBoost models with 3-fold cross-validated isotonic probability calibration. |
| **Phase 5** | Uncertainty Quantification | 33 -> 512 -> 256 -> 3 PyTorch Bayesian Neural Network with Monte Carlo Dropout (p=0.40, 50 stochastic passes) isolating epistemic and aleatoric variances. |
| **Phase 6** | Model Interpretability | SHAP TreeExplainer quantifying global and localized feature attribution, identifying key risk drivers (temporal, spatial, demographic). |
| **Phase 7** | Policy Intervention Engine | Simulation of 7 discrete counterfactual levers (speed limits, lighting, road surface, congestion management, blackspot mitigation) to rank risk-reduction interventions. |
| **Phase 8** | Spatial Risk Surface | 2D Kernel Density Estimation over 131,000 Serious and Fatal incidents across UK coordinates (49.91N-60.54N, -7.50E-1.76E). |
| **Phase 9** | Unified Inference Engine | Integrated scoring wrapper providing class probabilities, uncertainty decompositions, and confidence intervals for individual scenarios. |
| **Phase 10** | Manifold Analysis | PCA and t-SNE latent space projections capturing decision boundaries across accident severity tiers. |
| **Phase 11** | Statistical Validation | Conformal prediction sets with distribution-free coverage guarantees and Expected Calibration Error (ECE) auditing. |
| **Phase 12** | Artifact Export | Serialization of calibrated models, scalers, encoders, and inference pipelines for deployment. |

---

## Performance and Metrics

- **Test Set Accuracy**: 78.5% across three severity classes
- **Expected Calibration Error (ECE)**: 0.0742 post-isotonic calibration
- **Uncertainty Decomposition**:
  - Epistemic Mean: 0.00650 (Model uncertainty)
  - Aleatoric Mean: 0.31500 (Inherent data noise)
  - Combined Mean: 0.32150
- **Spatial Coverage**: Complete United Kingdom territorial coordinates mapped at 20x20 grid resolution

---

## Repository Structure

```
.
├── backend/
│   ├── phase1_full.py          # Data exploration and leakage audit
│   ├── phase2_full.py          # Feature engineering and spatial metrics
│   ├── phase3_full.py          # SMOTE-NC class balancing
│   ├── phase4_full.py          # Ensemble training and isotonic calibration
│   ├── phase5_full.py          # Bayesian neural network with MC dropout
│   ├── phase6_full.py          # SHAP interpretability and feature importance
│   ├── phase7_full.py          # Policy intervention simulation engine
│   ├── phase8_full.py          # Spatial risk surface and KDE modeling
│   ├── phase9_full.py          # Unified inference wrapper
│   ├── phase10_full.py         # PCA / t-SNE latent space projections
│   ├── phase11_full.py         # Conformal prediction and calibration checks
│   └── phase12_full.py         # Production artifact serialization
├── daatset/
│   └── final_merged_accident_dataset.csv
├── EXECUTION_GUIDE.md          # Step-by-step runtime manual
├── QUICKSTART.txt              # Quickstart commands and environment instructions
└── SESSION_SUMMARY.txt         # Empirical execution summary and logs
```

---

## Setup and Execution

### 1. Prerequisites
- Python 3.10+ (tested with Python 3.13)
- PyTorch, XGBoost, LightGBM, CatBoost, Scikit-learn, Imbalanced-learn, SHAP

### 2. Installation
```bash
pip install torch xgboost lightgbm catboost scikit-learn imbalanced-learn shap pandas numpy scipy
```

### 3. Running Pipeline Phases
To execute the pipeline sequentially, run the respective phase scripts from the `backend/` directory:

```bash
cd backend

# Execute feature engineering
python phase2_full.py

# Run class balancing
python phase3_full.py

# Train ensemble classifiers
python phase4_full.py

# Quantify Bayesian uncertainties
python phase5_full.py

# Run counterfactual policy engine
python phase7_full.py
```

Refer to `EXECUTION_GUIDE.md` for full instructions on running all 12 phases.
