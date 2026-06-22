# Final Model Comparison — Conclusion

**Total experiments run:** 21

## Master Leaderboard

| Rank | ID | Model | Acc% | Macro F1 | F1-Slight | F1-Serious | F1-Fatal | ROC-AUC | Time |
|------|----|-------|------|----------|-----------|------------|----------|---------|------|
| 1 | E2a | ResidualMLP (BN+Dropout+FocalLoss) | 60.60% | 0.3861 | 0.7163 | 0.3942 | 0.0478 | 0.6561 | 721.9s |
| 2 | E4d | BalancedRandomForest (100 trees) | 56.64% | 0.3835 | 0.7068 | 0.3424 | 0.1012 | 0.6792 | 5.1s |
| 3 | E3b | MC Dropout BNN (T=50) | 54.78% | 0.3738 | 0.6928 | 0.3426 | 0.0861 | 0.6709 | 9.6s |
| 4 | E5d | Stacking (XGB+LGB+CB → LogReg meta) | 53.77% | 0.3738 | 0.6829 | 0.3456 | 0.0930 | 0.6864 | 8.6s |
| 5 | E1a | XGBoost + Class Weights | 54.51% | 0.3719 | 0.7014 | 0.3263 | 0.0879 | 0.6922 | 16.3s |
| 6 | E1b | XGBoost + Focal-like (gamma=1.5) | 54.46% | 0.3710 | 0.7015 | 0.3242 | 0.0874 | 0.6919 | 12.8s |
| 7 | E1c | LightGBM GOSS | 53.97% | 0.3695 | 0.6965 | 0.3231 | 0.0887 | 0.6928 | 8.4s |
| 8 | E5c | Blending (w=[np.float64(0.4), np.float64(0.3), np.float64(0.1), np.float64(0.2)]) | 54.24% | 0.3685 | 0.7020 | 0.3169 | 0.0867 | 0.6888 | 0.1s |
| 9 | E4b | Class-Weighted XGBoost (no SMOTE) | 54.04% | 0.3684 | 0.6989 | 0.3205 | 0.0859 | 0.6897 | 5.6s |
| 10 | E4a | SMOTE + XGBoost (baseline) | 54.07% | 0.3682 | 0.6991 | 0.3198 | 0.0856 | 0.6892 | 12.9s |
| 11 | E5b | Soft Voting 4-model (XGB+LGB+CB+MLP) | 54.13% | 0.3670 | 0.7023 | 0.3126 | 0.0861 | 0.6869 | 0.1s |
| 12 | E5a | Soft Voting (XGB+LGB+CB) | 53.86% | 0.3662 | 0.6997 | 0.3132 | 0.0856 | 0.6886 | 202.1s |
| 13 | E3a | Deep Ensemble (5 MLPs) | 53.05% | 0.3627 | 0.6882 | 0.3163 | 0.0836 | 0.6709 | 588.5s |
| 14 | E1e | CatBoost Ordered + Native Cats | 53.41% | 0.3627 | 0.6970 | 0.3061 | 0.0850 | 0.6836 | 177.6s |
| 15 | E1d | LightGBM DART | 53.36% | 0.3597 | 0.7004 | 0.2968 | 0.0820 | 0.6808 | 26.4s |
| 16 | E2b | TabNet (n_d=32, n_steps=4) | 53.70% | 0.3394 | 0.7151 | 0.2308 | 0.0722 | 0.6391 | 301.9s |
| 17 | E6d | XGBoost + Grid Cell (0.1° resolution) | 76.93% | 0.3160 | 0.8692 | 0.0775 | 0.0013 | 0.7243 | 4.8s |
| 18 | E6a | XGBoost (raw lat/lon baseline) | 76.93% | 0.3151 | 0.8691 | 0.0763 | 0.0000 | 0.7242 | 7.4s |
| 19 | E6b | XGBoost + KMeans Cluster (k=50) | 76.92% | 0.3151 | 0.8690 | 0.0762 | 0.0000 | 0.7243 | 5.5s |
| 20 | E6c | XGBoost + KMeans + Dist-to-Centre | 76.91% | 0.3148 | 0.8690 | 0.0755 | 0.0000 | 0.7244 | 4.9s |
| 21 | E4c | Focal Loss MLP (no SMOTE, gamma=2) | 16.79% | 0.1378 | 0.0357 | 0.2988 | 0.0788 | 0.6334 | 84.7s |

## Winners

- **Best Overall (Macro F1):** [E2a] ResidualMLP (BN+Dropout+FocalLoss) — Acc=60.60%, Macro F1=0.3861
- **Best Fatal Detection:** [E4d] BalancedRandomForest (100 trees) — F1-Fatal=0.1012
- **Highest Accuracy:** [E6d] XGBoost + Grid Cell (0.1° resolution) — Acc=76.93%

## Experiment Set Summaries

### E1: Best = XGBoost + Class Weights
- Accuracy: 54.51%  |  Macro F1: 0.3719  |  Fatal F1: 0.0879

### E2: Best = ResidualMLP (BN+Dropout+FocalLoss)
- Accuracy: 60.60%  |  Macro F1: 0.3861  |  Fatal F1: 0.0478

### E3: Best = MC Dropout BNN (T=50)
- Accuracy: 54.78%  |  Macro F1: 0.3738  |  Fatal F1: 0.0861

### E4: Best = BalancedRandomForest (100 trees)
- Accuracy: 56.64%  |  Macro F1: 0.3835  |  Fatal F1: 0.1012

### E5: Best = Stacking (XGB+LGB+CB → LogReg meta)
- Accuracy: 53.77%  |  Macro F1: 0.3738  |  Fatal F1: 0.0930

### E6: Best = XGBoost + Grid Cell (0.1° resolution)
- Accuracy: 76.93%  |  Macro F1: 0.3160  |  Fatal F1: 0.0013

## Recommendation

For production deployment, use **ResidualMLP (BN+Dropout+FocalLoss)** (experiment E2a) which achieves the best balance of accuracy and generalisation across all severity classes.

For maximum **Fatal accident detection** (highest priority in road safety), use **BalancedRandomForest (100 trees)** (experiment E4d).

## Generated Graphs

- `FINAL_leaderboard.png` — Master ranking chart
- `FINAL_f1_heatmap.png` — Per-class F1 heatmap across all experiments
- `FINAL_set_comparison.png` — Best model per experiment set
- Per-experiment confusion matrices and F1 bar charts in `graphs/`
