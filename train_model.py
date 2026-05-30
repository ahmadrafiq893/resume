# ============================================
# AI Resume Analyzer - Complete Training Pipeline
# ============================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("static/graphs", exist_ok=True)

print("="*70)
print("🤖 AI RESUME ANALYZER - COMPLETE TRAINING PIPELINE")
print("="*70)

# ============================================
# 1. DATASET LOADING
# ============================================

print("\n📂 STEP 1: Loading Dataset...")
df = pd.read_csv("./professional_resume_dataset_300_records.csv")
print(f"✅ Loaded {len(df)} resumes")
print(f"📊 Dataset shape: {df.shape}")

# ============================================
# 2. DATA PREPROCESSING
# ============================================

print("\n🔧 STEP 2: Data Preprocessing...")

# Remove duplicates
initial_count = len(df)
df.drop_duplicates(inplace=True)
print(f"   Removed {initial_count - len(df)} duplicate records")

# Fill missing values
df.fillna("Unknown", inplace=True)

# Drop unnecessary columns
if "ResumeID" in df.columns:
    df.drop("ResumeID", axis=1, inplace=True)
    print("   Dropped ResumeID column")

# Encode categorical variables
edu_encoder = LabelEncoder()
cert_encoder = LabelEncoder()
role_encoder = LabelEncoder()

df["Education"] = edu_encoder.fit_transform(df["Education"])
df["Certification"] = cert_encoder.fit_transform(df["Certification"])
df["JobRole"] = role_encoder.fit_transform(df["JobRole"])

print(f"   Education levels: {list(edu_encoder.classes_)}")
print(f"   Certification types: {list(cert_encoder.classes_)}")
print(f"   Total job roles: {len(role_encoder.classes_)}")
print(f"   Job roles: {list(role_encoder.classes_)}")

# ============================================
# 3. FEATURE EXTRACTION
# ============================================

print("\n📝 STEP 3: Feature Extraction...")

# TF-IDF Vectorization for Skills
tfidf = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
skills_features = tfidf.fit_transform(df["Skills"])
print(f"   TF-IDF features shape: {skills_features.shape}")

# Numerical features
numerical_cols = ["Education", "YearsExperience", "Projects", "Internships", "CGPA", "Certification"]
numerical = df[numerical_cols]

# Scaling
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(numerical)
print(f"   Numerical features shape: {scaled_numerical.shape}")

# Combine features
from scipy.sparse import hstack
X = hstack([skills_features, scaled_numerical])
X_dense = X.toarray()
print(f"   Combined features shape: {X_dense.shape}")

# PCA for dimensionality reduction
pca = PCA(n_components=0.95, random_state=42)
X_final = pca.fit_transform(X_dense)
print(f"   After PCA shape: {X_final.shape}")

y = df["JobRole"]

# ============================================
# 4. TRAIN-TEST SPLIT
# ============================================

print("\n✂️ STEP 4: Train-Test Split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# ============================================
# 5. MODEL TRAINING
# ============================================

print("\n🤖 STEP 5: Training Multiple Models...")
print("-"*50)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "ANN": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBClassifier(eval_metric="mlogloss", random_state=42)

results = []
best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    print(f"\n   Training: {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    results.append({
        "Model": name,
        "Accuracy": acc * 100,
        "Precision": prec * 100,
        "Recall": rec * 100,
        "F1-Score": f1 * 100
    })
    
    print(f"      ✅ Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# ============================================
# 6. MODEL EVALUATION
# ============================================

print("\n" + "="*70)
print("📊 STEP 6: MODEL EVALUATION & COMPARISON")
print("="*70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Accuracy", ascending=False)

print("\n📈 Model Performance Comparison:")
print("-"*70)
print(results_df.to_string(index=False))
print("-"*70)

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   📈 Accuracy: {best_accuracy*100:.2f}%")

# Save results
results_df.to_csv("model_comparison.csv", index=False)
print("✅ Model comparison saved to 'model_comparison.csv'")

# ============================================
# 7. GRAPHICAL ANALYSIS
# ============================================

print("\n📈 STEP 7: Generating Graphs...")

# 7.1 Accuracy Comparison Graph
plt.figure(figsize=(14, 7))
colors = ['#4CAF50' if i == 0 else '#2196F3' for i in range(len(results_df))]
bars = plt.bar(results_df["Model"], results_df["Accuracy"], color=colors)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("🎯 Model Accuracy Comparison - AI Resume Analyzer", fontsize=14, fontweight='bold')
plt.ylim(0, 100)
for bar, acc in zip(bars, results_df["Accuracy"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig("static/graphs/accuracy_graph.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Accuracy graph saved to 'static/graphs/accuracy_graph.png'")

# 7.2 Confusion Matrix
best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=role_encoder.classes_, 
            yticklabels=role_encoder.classes_,
            annot_kws={'size': 8})
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title(f"Confusion Matrix - {best_model_name}", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("static/graphs/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Confusion matrix saved to 'static/graphs/confusion_matrix.png'")

# 7.3 Metrics Comparison
metrics_df = results_df.melt(id_vars=["Model"], 
                             value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
                             var_name="Metric", value_name="Score")

plt.figure(figsize=(15, 7))
sns.barplot(data=metrics_df, x="Model", y="Score", hue="Metric", palette="viridis")
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.ylabel("Score (%)", fontsize=12)
plt.title("📊 Model Performance Metrics Comparison", fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("static/graphs/metrics_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Metrics comparison saved to 'static/graphs/metrics_comparison.png'")

# ============================================
# 8. SAVE MODELS
# ============================================

print("\n💾 STEP 8: Saving Best Model...")

joblib.dump(best_model, "models/best_resume_model.pkl")
joblib.dump(tfidf, "models/tfidf.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(pca, "models/pca.pkl")
joblib.dump(role_encoder, "models/jobrole_encoder.pkl")
joblib.dump(edu_encoder, "models/education_encoder.pkl")
joblib.dump(cert_encoder, "models/certification_encoder.pkl")

print("✅ Models saved to 'models/' directory:")
print("   📁 best_resume_model.pkl")
print("   📁 tfidf.pkl")
print("   📁 scaler.pkl")
print("   📁 pca.pkl")
print("   📁 jobrole_encoder.pkl")

# ============================================
# 9. CLASSIFICATION REPORT
# ============================================

print("\n📋 STEP 9: Classification Report")
print("-"*50)
print(classification_report(y_test, best_pred, target_names=role_encoder.classes_))

# ============================================
# 10. TEST PREDICTION
# ============================================

print("\n🧪 STEP 10: Testing Prediction...")
test_skills = "python, sql, machine learning, pandas, numpy, data visualization"
test_education = 2  # Bachelor
test_experience = 3
test_projects = 4
test_internships = 2
test_cgpa = 3.5
test_certification = 1

test_skills_vec = tfidf.transform([test_skills])
test_num = scaler.transform([[test_education, test_experience, test_projects, test_internships, test_cgpa, test_certification]])
test_combined = hstack([test_skills_vec, test_num]).toarray()
test_pca = pca.transform(test_combined)
test_pred = best_model.predict(test_pca)
test_role = role_encoder.inverse_transform(test_pred)[0]
test_proba = best_model.predict_proba(test_pca)[0]
test_confidence = max(test_proba) * 100

print(f"   📝 Test Skills: {test_skills}")
print(f"   🎯 Predicted Role: {test_role}")
print(f"   📈 Confidence: {test_confidence:.2f}%")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"\n📊 SUMMARY:")
print(f"   ├── Total resumes processed: {len(df)}")
print(f"   ├── Features after PCA: {X_final.shape[1]}")
print(f"   ├── Models trained: {len(models)}")
print(f"   ├── Best model: {best_model_name}")
print(f"   ├── Best accuracy: {best_accuracy*100:.2f}%")
print(f"   ├── Models saved in 'models/' folder")
print(f"   └── Graphs saved in 'static/graphs/' folder")
print("\n🚀 Next Step: Run 'python app.py' to start the web application")
print("="*70)