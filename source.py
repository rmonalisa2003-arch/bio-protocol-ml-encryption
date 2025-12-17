# ML-Based Encryption Framework for Bio-Protocol Security in Cloud
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# Step 1: Load Dataset
df = pd.read_csv('dataset.csv')

# Step 2: Encode Categorical Features
le_sex = LabelEncoder()
le_source = LabelEncoder()
df['SEX'] = le_sex.fit_transform(df['SEX'])   # F/M → 0/1
df['SOURCE'] = le_source.fit_transform(df['SOURCE'])

# Step 3: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Step 4: KMeans Clustering (Group profiles)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 5: Anomaly Detection (Outlier Profiles)
iso_forest = IsolationForest(contamination=0.02, random_state=42)
df['Anomaly'] = iso_forest.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal

# Step 6: Visualization (optional)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Scatter plot of clusters
plt.figure(figsize=(8, 6))
for cluster in df['Cluster'].unique():
    subset = df[df['Cluster'] == cluster]
    plt.scatter(subset['PCA1'], subset['PCA2'], label=f'Cluster {cluster}')
plt.title('KMeans Clustering of Bio Profiles')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Encryption Policy Assignment (Conceptual Example)
def assign_encryption(cluster, anomaly):
    if anomaly == -1:
        return "RSA+OTP+KeyRotation"  # Strongest encryption
    elif cluster == 0:
        return "AES-128"
    elif cluster == 1:
        return "AES-256"
    elif cluster == 2:
        return "RSA"
    else:
        return "AES-Default"

df['Encryption_Level'] = df.apply(lambda row: assign_encryption(row['Cluster'], row['Anomaly']), axis=1)

# Step 8: Final Output
output = df[['HAEMATOCRIT', 'HAEMOGLOBINS', 'ERYTHROCYTE', 'LEUCOCYTE',
             'THROMBOCYTE', 'MCH', 'MCHC', 'MCV', 'AGE', 'SEX', 'SOURCE',
             'Cluster', 'Anomaly', 'Encryption_Level']]

# Save to CSV (optional)
output.to_csv("encrypted_bio_profiles.csv", index=False)

# Show result sample
print(output.head())

# Step 1: Create pseudo ground truth
# Let's assume Cluster 2 = anomaly (label = 1), others = normal (label = 0)
df['True_Label'] = df['Cluster'].apply(lambda x: 1 if x == 2 else 0)

# Step 2: Convert Isolation Forest output to binary prediction (1 = anomaly, 0 = normal)
df['Predicted_Label'] = df['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Step 3: Accuracy and Confusion Matrix
accuracy = accuracy_score(df['True_Label'], df['Predicted_Label'])
conf_matrix = confusion_matrix(df['True_Label'], df['Predicted_Label'])
report = classification_report(df['True_Label'], df['Predicted_Label'])

# Step 4: ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(df['True_Label'], df['Predicted_Label'])
roc_auc = auc(fpr, tpr)

# Step 5: Plot ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Isolation Forest (simulated)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Output Results
print("✅ Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)
