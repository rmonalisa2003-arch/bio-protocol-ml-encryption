# bio-protocol-ml-encryption
ML-Based Encryption Framework for Bio-Protocol Security
This project implements an intelligent, adaptive security framework that uses Machine Learning to dynamically assign encryption protocols to biological data based on risk assessment and anomaly detection.

**Project Overview:**
The framework analyzes biometric data (such as Haematocrit, Hemoglobin, and Erythrocyte levels) to categorize records into different security tiers. By identifying "anomalous" or "high-risk" profiles, the system applies stronger cryptographic measures to sensitive data while maintaining system efficiency for standard records.

**Technical Workflow:**
1. Data Preprocessing
Feature Encoding: Categorical variables like SEX and SOURCE are transformed into numerical values (e.g., M/F to 1/0) for model compatibility.

Normalization: All numerical features are processed through StandardScaler to ensure that unit differences do not bias the clustering results.

Missing Values: The system automatically fills missing numerical data with mean values to maintain dataset integrity.

2. Multi-Tiered Security Logic
The framework uses a hybrid approach to classify data sensitivity:

Unsupervised Clustering: Uses KMeans (3 clusters) to group similar biological profiles.

Anomaly Detection: Employs Isolation Forest to flag statistical outliers that may represent critical health data or security risks.

3. Adaptive Encryption PolicyBased on the ML output, the following encryption levels are assigned:Risk CategoryML ConditionEncryption ProtocolCritical/AnomalyIsolation Forest = -1RSA + OTP + Key RotationHigh RiskKMeans Cluster 2RSAMedium RiskKMeans Cluster 1AES-256Low RiskKMeans Cluster 0 / OthersAES-128


4. Advanced Model Refinement
Imbalance Handling: The project utilizes SMOTE (Synthetic Minority Over-sampling Technique) to balance the data, ensuring the model is highly sensitive to rare anomalies.

Supervised Learning: The final iteration (sourcev3.py) implements a Random Forest Classifier with stratified train-test splits to achieve robust predictive performance.

## Evaluation & Visualization:
The system provides several metrics to verify the accuracy of the security assignments:

ROC Curves & AUC: Measures the model's effectiveness in distinguishing between normal and high-risk profiles.

Confusion Matrices: Provides a detailed breakdown of correct vs. incorrect risk classifications.

PCA Visualization: Reduces the biological features to two dimensions to visually demonstrate how the clusters are separated.

## Getting Started
Installation:
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
Execution
Place your dataset.csv in the root directory.

Run the final optimized model:
python src/sourcev3.py
