# Figure interpretations

**FIG01** - The four risk tiers are imbalanced on purpose: Low 35%, Moderate 33%, High 22%, Critical 10%. Critical cases are the rarest but the most dangerous, so plain accuracy would hide poor performance on them. This is why class weights and recall-focused metrics are used later.

**FIG02** - PM2.5 and PM10 move together (r = 0.87), which is expected because both come from the same combustion, dust and smog sources. PM2.5 also correlates with NO2 (r = 0.61) and CO (r = 0.52), pointing to traffic and burning. Humidity and wind show weaker links (humidity r = 0.19, wind r = -0.18). The strong pollutant-pollutant correlation is a warning about redundancy, which is why feature selection follows.

**FIG03** - In the multi-year city-day data the six most polluted cities peak in month 11 (mean 180 ug/m3) and bottom out in month 8 (43 ug/m3), the winter-smog versus monsoon-wash pattern. In the Pakistani timeline, Lahore has the highest average PM2.5, and the average of the five cities rises from 42 ug/m3 (first 4 weeks) to 47 ug/m3 (last 4 weeks). A three-month window covers only part of the smog season, a bias noted in the report.

**FIG04** - Median PM2.5 rises from 27 ug/m3 (Low) to 37 (Moderate), 51 (High) and 81 (Critical). The boxes overlap, so PM2.5 alone cannot separate the tiers: a person with a poor clinical profile can reach a high tier on a moderately polluted day. That overlap is the reason a model combining pollution with the patient profile is needed instead of a plain AQI reading.

**FIG05** - Before capping the maximum hourly PM2.5 was 265 ug/m3; after capping it is 149 ug/m3 (fence at 149.0 ug/m3, 84 readings changed out of 11,120). The wide 3 x IQR fence leaves the long right tail of real smog episodes intact and only trims isolated sensor-style spikes.

**FIG06** - The engineered features most correlated with the risk level are Pollution_x_Vulnerability (|r| = 0.67), Smog_Severity_Index (0.57) and Rolling_24h_PM25 (0.47). The weakest is Diurnal_Temp_Range (0.09). Several pollution features are strongly inter-correlated (Smog Severity, Rolling 24h, Lag 1d), so they carry overlapping information; tree models cope with this, but the SVM and logistic regression benefit from the later feature selection.

**FIG07** - 10 of the top 20 mutual-information features are engineered ones. The top three are Pollution_x_Vulnerability, DustExposure, PhysicalActivity. Mutual information captures non-linear dependence on the risk tier, so combined indices such as Pollution_x_Vulnerability rank high because they merge environment and patient factors.

**FIG08** - RFE kept 20 of 36 features, and 16 of them also appear in the mutual-information top 20, so the two methods largely agree. The Random Forest ranks Pollution_x_Vulnerability first. Final features are the 20 with the best average rank across MI, Gini and RFE.

**FIG09** - All three main models reach a macro ROC-AUC between 0.895 and 0.911; the logistic-regression baseline scores 0.905. For SVM (RBF), per-class AUC is Low 0.940, Moderate 0.818, High 0.903, Critical 0.984. The middle tiers (Moderate, High) are hardest because they sit between neighbours, while the extremes separate cleanly.

**FIG10** - Precision-recall curves matter more than ROC here because Critical cases are only about 10% of the test set. For SVM (RBF) the Critical class has a PR-AUC of 0.896. The curves show how much precision is given up when the alert threshold is lowered to catch more high-risk patients, which is the trade-off behind the app's alert threshold of 0.30.

**FIG11** - For SVM (RBF), 680 of 2250 test records are misclassified (30.2%). Nearly all errors are one tier away from the truth. 130 High/Critical patients were predicted Low/Moderate (a high-risk miss rate of 18.1%), and 141 Low/Moderate patients were flagged High/Critical.

**FIG12** - Logistic Regression (baseline) answers fastest (1.37 ms per single prediction) and Random Forest slowest (168.80 ms). Accuracy differs by only 0.023 between all four models, so for a web app every model is fast enough; the choice should be driven by recall and stability rather than latency.