import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)
from sklearn.model_selection import train_test_split
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

# Step 4: Isolation Forest with tuned contamination (optional)
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

# Step 6: Prepare for supervised classification
df['True_Label'] = df['Cluster'].apply(lambda x: 1 if x == 2 else 0)

# Features and labels
X = df_scaled
y = df['True_Label']

# Step 7: Split data BEFORE SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 8: Apply SMOTE only on training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Step 9: Train Random Forest Classifier on balanced training set
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)

# Step 10: Predict and get probabilities on original test set
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# Step 11: Evaluate
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Step 12: Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest Classifier')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 13: Print Evaluation Results
print("✅ Supervised Classification Results (Random Forest)")
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

# Step 14: Save the dataset with clusters and encryption level
df.to_csv('encrypted_bio_profiles_rf_final.csv', index=False)
