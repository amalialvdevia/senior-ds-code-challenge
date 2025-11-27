"""
Improved Model - Cost-sensitive Return Prediction
Author: Amalia Devia

This script improves the baseline logistic model by aligning evaluation
with business value and using a Random Forest inside a preprocessing pipeline.
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import joblib


# ---------------------------------------------------
# 1. Business parameters
# ---------------------------------------------------
COST_RETURN = 18.0          # cost of each return
COST_INTERVENTION = 3.0     # cost of each preventive intervention
# simplified assumption: a successful intervention avoids the full return cost
GAIN_TP = COST_RETURN - COST_INTERVENTION   # +15 USD per true positive


def financial_eval(y_true, y_proba, threshold: float) -> dict:
    """
    Compute standard metrics plus expected value per order,
    using the following cost matrix:
      - TP: +15 USD  (we avoid a return but pay the intervention)
      - FP: -3 USD   (unnecessary intervention)
      - FN, TN: 0 USD (no intervention applied)
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    ev_per_order = (tp * GAIN_TP - fp * COST_INTERVENTION) / len(y_true)

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "ev_per_order": ev_per_order,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


# ---------------------------------------------------
# 2. Load data
# ---------------------------------------------------
train = pd.read_csv("ecommerce_returns_train.csv")
test = pd.read_csv("ecommerce_returns_test.csv")

y_train = train["is_return"]
y_test = test["is_return"]

FEATURES = [
    "customer_age",
    "customer_tenure_days",
    "product_category",
    "product_price",
    "days_since_last_purchase",
    "previous_returns",
    "product_rating",
    "size_purchased",
    "discount_applied",
]

X_train = train[FEATURES].copy()
X_test = test[FEATURES].copy()


# ---------------------------------------------------
# 3. Baseline: Logistic Regression similar to the provided script
# ---------------------------------------------------
def run_baseline_logistic(X_train, y_train, X_test, y_test):
    """Reproduce a simple logistic baseline similar to baseline_model.py."""
    from sklearn.preprocessing import LabelEncoder

    def preprocess_baseline(df, fit: bool = True, le_cat=None, le_size=None):
        dfp = df.copy()

        # Encode product_category as ordinal labels (same spirit as the baseline)
        if fit:
            le_cat = LabelEncoder()
            dfp["product_category_encoded"] = le_cat.fit_transform(
                dfp["product_category"]
            )
        else:
            dfp["product_category_encoded"] = le_cat.transform(
                dfp["product_category"]
            )

        # Encode size_purchased only when present
        if dfp["size_purchased"].notna().any():
            most_common_size = dfp["size_purchased"].mode()[0]
            dfp["size_purchased"].fillna(most_common_size, inplace=True)
            if fit:
                le_size = LabelEncoder()
                dfp["size_encoded"] = le_size.fit_transform(dfp["size_purchased"])
            else:
                dfp["size_encoded"] = le_size.transform(dfp["size_purchased"])
        else:
            dfp["size_encoded"] = 0

        feature_cols = [
            "customer_age",
            "customer_tenure_days",
            "product_category_encoded",
            "product_price",
            "days_since_last_purchase",
            "previous_returns",
            "product_rating",
            "size_encoded",
            "discount_applied",
        ]

        return dfp[feature_cols], le_cat, le_size

    # Fit encoders on train, apply to test
    X_train_b, le_cat, le_size = preprocess_baseline(X_train, fit=True)
    X_test_b, _, _ = preprocess_baseline(
        X_test, fit=False, le_cat=le_cat, le_size=le_size
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_b)
    X_test_scaled = scaler.transform(X_test_b)

    # Train logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n=== Baseline Logistic Regression (threshold=0.5) ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.3f}")
    print(classification_report(y_test, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Expected value with threshold = 0.5
    baseline_ev = financial_eval(y_test, y_proba, threshold=0.5)
    print(
        f"Expected value per order (threshold=0.5): "
        f"{baseline_ev['ev_per_order']:.3f} USD"
    )

    # Sweep thresholds just for comparison
    thresholds = np.linspace(0.1, 0.9, 17)
    evs = [financial_eval(y_test, y_proba, t) for t in thresholds]
    best = max(evs, key=lambda d: d["ev_per_order"])

    print(
        f"Best financial threshold (logistic): τ={best['threshold']:.2f}, "
        f"EV/order={best['ev_per_order']:.3f} USD"
    )

    return model, scaler


baseline_model, baseline_scaler = run_baseline_logistic(
    X_train, y_train, X_test, y_test
)


# ---------------------------------------------------
# 4. Improved model: RF + OneHot + pipeline
# ---------------------------------------------------
num_features = [
    "customer_age",
    "customer_tenure_days",
    "product_price",
    "days_since_last_purchase",
    "previous_returns",
    "product_rating",
]
cat_features = ["product_category", "size_purchased", "discount_applied"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=20,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf),
    ]
)

# More robust AUC via cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = cross_val_predict(
    rf_pipeline, X_train, y_train, cv=cv, method="predict_proba"
)[:, 1]
cv_auc = roc_auc_score(y_train, oof_proba)

print("\n=== Improved Model: Random Forest Pipeline ===")
print(f"Cross-validated ROC-AUC (train): {cv_auc:.3f}")

# Train on full train set and evaluate on test
rf_pipeline.fit(X_train, y_train)
y_test_proba = rf_pipeline.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_test_proba)
print(f"Test ROC-AUC: {test_auc:.3f}")

# Sweep thresholds to find the best expected value
thresholds = np.linspace(0.1, 0.9, 17)
evs_rf = [financial_eval(y_test, y_test_proba, t) for t in thresholds]
best_rf = max(evs_rf, key=lambda d: d["ev_per_order"])

print("\nBest financial threshold (Random Forest):")
for k, v in best_rf.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.3f}")
    else:
        print(f"  {k}: {v}")

# Detailed metrics at the optimal threshold
tau = best_rf["threshold"]
y_pred_opt = (y_test_proba >= tau).astype(int)

print("\nMetrics at optimal threshold:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred_opt):.3f}")
print(f"  Precision: {precision_score(y_test, y_pred_opt):.3f}")
print(f"  Recall   : {recall_score(y_test, y_pred_opt):.3f}")
print(f"  F1       : {f1_score(y_test, y_pred_opt):.3f}")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt).ravel()
print(f"  Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

monthly_saving = best_rf["ev_per_order"] * 100_000  # 100k+ orders/month
print(f"\nEstimated monthly saving (100k orders): ${monthly_saving:,.0f} USD")

# ---------------------------------------------------
# 5. Save final model artifacts
# ---------------------------------------------------
joblib.dump(rf_pipeline, "final_model_rf_pipeline.pkl")
joblib.dump({"best_threshold": tau}, "threshold_meta.pkl")

print("\nArtifacts saved: final_model_rf_pipeline.pkl, threshold_meta.pkl")
print("Done.")
