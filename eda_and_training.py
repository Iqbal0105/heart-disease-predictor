# EDA dan Training Model Prediksi Penyakit Jantung (All-in-One Script)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score
)

# ========== 1. Load Dataset ==========
DATA_PATH = 'data.csv'  # ganti dengan nama file CSV-mu
df = pd.read_csv(DATA_PATH)

print("Sample Data:\n", df.head(), "\n")
print("Data Info:\n")
print(df.info(), "\n")

# ========== 2. Cek Missing Values ==========
print("Missing Values:\n", df.isnull().sum(), "\n")

# ========== 3. Visualisasi Distribusi Target ==========
sns.countplot(data=df, x='HeartDisease')
plt.title("Distribusi Label Penyakit Jantung")
plt.savefig("output_label_distribution.png")
plt.clf()

# ========== 4. Visualisasi Fitur Numerik ==========
sns.histplot(data=df, x='BMI', kde=True)
plt.title("Distribusi BMI")
plt.savefig("output_bmi_distribution.png")
plt.clf()

# ========== 5. Siapkan Data ==========
selected_features = [
    "GenHealth", "AgeCategory", "Stroke", "Diabetic", "PhysicalActivity",
    "DiffWalking", "BMI", "Smoking", "KidneyDisease", "Sex"
]

df = df[selected_features + ['HeartDisease']].dropna()
X = df[selected_features]
y = df['HeartDisease'].map({'Yes': 1, 'No': 0})

numerical_features = ['BMI']
categorical_features = [f for f in selected_features if f not in numerical_features]

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 6. Pipeline dan Training ==========
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# ========== 7. Evaluasi ==========
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix")
plt.savefig("output_confusion_matrix.png")
plt.clf()

# ========== 8. ROC Curve ==========
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend(loc="lower right")
plt.savefig("output_roc_curve.png")
plt.clf()

print(f"\nROC AUC Score: {roc_auc:.4f}")

# ========== 9. Simpan Model ==========
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/heart_disease_model.pkl"
joblib.dump(pipeline, MODEL_PATH)
print(f"\n✅ Model berhasil disimpan di {MODEL_PATH}")
