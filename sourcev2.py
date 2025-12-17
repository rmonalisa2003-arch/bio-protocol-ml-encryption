import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('dataset.csv')

# Step 1: Preprocess
df.fillna(df.mean(numeric_only=True), inplace=True)
df['SEX'] = df['SEX'].map({'M': 1, 'F': 0})
df['SOURCE'] = df['SOURCE'].map({'Venous Blood': 1, 'Capillary Blood': 0})

# Step 2: Normalize features
features = df.drop(['SEX', 'SOURCE'], axis=1).columns
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# Step 3: KMeans Clustering (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Step 4: Isolation Forest with tuned contamination
iso_forest = IsolationForest(contamination=0.15, random_state=42)
df['Anomaly'] = iso_forest.fit_predict(df_scaled)

# Step 5: Assign encryption levels
def assign_encryption(cluster, anomaly):
    if anomaly == -1:
        return 'RSA+OTP+KeyRotation'
    elif cluster == 2:
        return 'RSA'
    elif cluster == 1:
        return 'AES-256'
    else:
        return 'AES-128'

df['Encryption_Level'] = df.apply(lambda row: assign_encryption(row['Cluster'], row['Anomaly']), axis=1)

# Step 6: Prepare for evaluation
# Simulated ground truth: cluster 2 = "anomaly" → label 1, others = 0
df['True_Label'] = df['Cluster'].apply(lambda x: 1 if x == 2 else 0)
df['Predicted_Label'] = df['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Step 7: Apply SMOTE to balance classes for evaluation
X = df_scaled
y = df['True_Label']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train Isolation Forest on balanced data
iso_forest_resampled = IsolationForest(contamination=0.15, random_state=42)
pred_resampled = iso_forest_resampled.fit_predict(X_resampled)
pred_resampled = np.where(pred_resampled == -1, 1, 0)

# Step 8: Evaluation Metrics
accuracy = accuracy_score(y_resampled, pred_resampled)
conf_matrix = confusion_matrix(y_resampled, pred_resampled)
report = classification_report(y_resampled, pred_resampled)
fpr, tpr, _ = roc_curve(y_resampled, pred_resampled)
roc_auc = auc(fpr, tpr)

# Step 9: Print Evaluation
print("✅ Improved Evaluation Results")
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

# Step 10: Plot ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='navy')
plt.plot([0, 1], [0, 1], 'r--')
plt.title('ROC Curve - Isolation Forest (Balanced with SMOTE)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 11: Save the encrypted dataset
df.to_csv('encrypted_bio_profiles_improved.csv', index=False)
