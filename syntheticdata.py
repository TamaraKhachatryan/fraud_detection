import pandas as pd
import numpy as np
import random
import xgboost as xgb
import uuid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from collections import Counter

import seaborn as sns

# Generate synthetic data
n_samples = 150000
countries = ["AM","RU","US","DE","FR","IN","CN"]
statuses = ["active","suspended","closed"]
verif_levels = ["none","basic","full"]
currencies = ["USD","EUR","AMD","RUB"]
tr_types = ["purchase","withdrawal","deposit","refund"]
status_3d_choices = ["passed","failed","not_enrolled"]

data = []
for i in range(n_samples):
    user_id = str(uuid.uuid4())
    user_country = random.choice(countries)
    user_city = f"City_{random.randint(1,500)}"
    user_account_status = random.choice(statuses)
    user_balance = round(random.uniform(0,10000),2)
    user_verification_level = random.choice(verif_levels)
    transaction_id = str(uuid.uuid4())
    transaction_amount = round(random.uniform(1,5000),2)
    transaction_currency = random.choice(currencies)
    transaction_type = random.choice(tr_types)
    card_bin = str(random.randint(400000,499999))
    card_last = str(random.randint(1000,9999))
    status_3d = random.choice(status_3d_choices)
    cvv_result = random.choice([True,False])
    ip = f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}"
    device_id = str(uuid.uuid4())
    shipping_country = random.choice(countries)
    billing_country = random.choice(countries)

    risk_score = 0
    if shipping_country != billing_country: risk_score += 1
    if user_account_status=="suspended": risk_score += 1
    if user_verification_level=="none": risk_score += 1
    if transaction_amount>3000 and not cvv_result: risk_score += 1
    if status_3d=="failed": risk_score += 1

    is_fraud = 1 if (risk_score>=2 and random.random()<0.1) else 0
    data.append([
        user_id,user_country,user_city,user_account_status,user_balance,user_verification_level,
        transaction_id,transaction_amount,transaction_currency,transaction_type,
        card_bin,card_last,status_3d,cvv_result,ip,device_id,shipping_country,billing_country,
        is_fraud,risk_score
    ])

columns = ["user_id","user_country","user_city","user_account_status","user_balance","user_verification_level",
           "transaction_id","transaction_amount","transaction_currency","transaction_type",
           "card_bin","card_last","status_3d","cvv_result","ip","device_id","shipping_country","billing_country",
           "is_fraud","risk_score"]

df = pd.DataFrame(data, columns=columns)
print(df.head())
print(df.info())
fraud_counts = df['is_fraud'].value_counts()
total_count = len(df)

for cls in fraud_counts.index:
    count = fraud_counts[cls]
    percentage = (count / total_count) * 100
    label = "Fraud" if cls == 1 else "Non-Fraud"
    print(f"{label} դեպքերի քանակը: {count}, տոկոսը: {percentage:.2f}%")


# Data preparation
df_model = df.copy()
drop_cols = ['user_id','transaction_id','ip','device_id','card_bin','card_last']
df_model = df_model.drop(columns=drop_cols)

cat_cols = df_model.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop('is_fraud', axis=1)
y = df_model['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
print("նախքան SMOTE-ը։", Counter(y_train))
print("SMOTE-ից հետո։", Counter(y_train_bal))


models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train), use_label_encoder=False, random_state=42),
    "CatBoost": CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=0, random_state=42, class_weights=[1,10])
}
results = []
probas = {}
preds = {}
threshold = 0.3

for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= threshold).astype(int)
    probas[name] = y_proba
    preds[name] = y_pred

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })

results_df = pd.DataFrame(results).set_index("Model")
print("\n=== Models Performance ===")
print(results_df)


# Bar chart
ax = results_df.plot(kind="bar", figsize=(12,6))
plt.title("Fraud Detection Models Performance (150k samples)")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(loc="lower right")
plt.tight_layout()
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}',
                (p.get_x() + p.get_width()/2, p.get_height()),
                ha='center', va='bottom', fontsize=8, rotation=90)
plt.show()

# ROC Curve for all models

plt.figure(figsize=(8,6))
for name, y_proba in probas.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})")  # օգտագործում ենք probas[name]

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - All Models")
plt.legend()
plt.grid(True)
plt.show()

# Առանձին Confusion Matrix բոլոր մոդելների համար
for name, y_pred in preds.items():
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix - {name}:")
    print(cm)
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Risk score distribution
plt.figure(figsize=(8,5))
sns.histplot(df['risk_score'], bins=6, kde=False)
plt.xlabel("Risk Score")
plt.ylabel("Count")
plt.title("Risk Score Distribution")
plt.show()

models["CatBoost"].fit(X_train_bal, y_train_bal)
cat_model = models["CatBoost"]
cat_model.save_model('catboost_fraud_model.cbm')
print("CatBoost model saved as 'catboost_fraud_model.cbm'")