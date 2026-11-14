# Evaluation summary
Best model (validation macro-F1): **SVM (RBF)**. Runner-up: Gradient Boosting (XGBoost).

                         Model  Accuracy  Precision  Recall     F1  ROC_AUC  HighRisk_Recall  Val_F1  Val_Accuracy  CV5_F1_mean  CV5_F1_std  Train_Time_s  Inference_ms_per_pred
                 Random Forest    0.6747     0.6778  0.6878 0.6824   0.8954           0.8264  0.7174        0.7058       0.6965      0.0090          1.74                168.804
   Gradient Boosting (XGBoost)    0.6902     0.6878  0.7080 0.6968   0.9039           0.8347  0.7274        0.7173       0.7060      0.0046          2.24                  3.453
                     SVM (RBF)    0.6978     0.7139  0.7102 0.7119   0.9113           0.8194  0.7408        0.7280       0.7141      0.0070         28.01                  2.714
Logistic Regression (baseline)    0.6858     0.6869  0.7128 0.6981   0.9052           0.8667  0.7273        0.7173       0.7093      0.0054          0.30                  1.369

Paired t-test (5-fold CV): t = 5.742, p = 0.0046
McNemar (chi-square with continuity correction): p = 0.2707 (b = 114, c = 97)
High-risk false negatives: 130 (18.1%)
