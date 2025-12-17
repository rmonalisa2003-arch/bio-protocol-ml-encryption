## ML-Based Encryption Framework for Bio-Protocol Security in CloudThis repository implements an adaptive security framework that uses Machine Learning to dynamically assign encryption protocols to biological data based on risk assessment and anomaly detection.

## Project OverviewIn cloud-based healthcare systems, static encryption can be inefficient. This framework analyzes biological profiles (e.g., Haematocrit, Hemoglobin, Erythrocyte levels) to determine the sensitivity of each record. By identifying "anomalous" or "high-risk" profiles, the system applies stronger cryptographic measures to sensitive data while maintaining system performance for standard records.

## Technical Workflow:
1. Data Preprocessing & NormalizationFeature Encoding: Categorical variables like SEX and SOURCE are mapped to numerical values for model compatibility.Normalization: Numerical features are processed through StandardScaler to ensure that features with different units (like Age vs. Erythrocyte count) have equal weight.
Missing Values: The pipeline automatically fills missing numerical data with mean values to maintain dataset integrity.
2. Multi-Stage ML LogicUnsupervised Clustering: Uses KMeans to group similar biological profiles into three distinct risk tiers.
3. Anomaly Detection: Employs Isolation Forest to flag statistical outliers that may represent critical health data or security risks.
4. Supervised Refinement: The final implementation utilizes a Random Forest Classifier with SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance and ensure high accuracy in predicting high-risk profiles.
## Adaptive Encryption PolicySecurity protocols are automatically assigned based on the ML output:
# Risk CategoryML ConditionEncryption ProtocolCritical / AnomalyIsolation Forest Anomaly (-1)RSA + OTP + Key RotationHigh RiskKMeans Cluster 2RSAMedium RiskKMeans Cluster 1AES-256Low RiskKMeans Cluster 0 / OthersAES-128

## Evaluation ResultsThe system's performance is verified through comprehensive metrics:
ROC-AUC Curves: Evaluates the model's ability to distinguish between normal and high-risk profiles.

## Confusion Matrices:
Provides a breakdown of correct vs. incorrect risk classifications.Classification Reports: Includes Precision, Recall, and F1-scores for each security tier.

## Getting StartedInstallationEnsure you have Python installed, then run:
Bashpip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
UsagePlace your dataset.csv in the project root.Execute the final optimized model:Bashpython sourcev3.py

## LicenseThis project is licensed under the MIT License - see the LICENSE file for details.
