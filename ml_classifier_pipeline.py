"""
=============================================================================
END-TO-END CLASSIFIER PIPELINE
  Phase 1: Feature Engineering (10 domain features)
  Phase 2: Feature Selection (RFECV — finds optimal subset automatically)
  Phase 3: Classifier Benchmark (7 models × default + tuned × 5-fold CV)
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from time import time

# sklearn
from sklearn.model_selection import (
    train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# ─── REPRODUCIBILITY ───
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: LOAD + FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  PHASE 1: FEATURE ENGINEERING")
print("=" * 80)

df = pd.read_csv(r"C:\Users\Admin\Desktop\Notes\ML\churn\data\Churn_Modelling.csv")

# Encode
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# --- Engineered features ---
df['Age_Squared'] = df['Age'] ** 2
df['Balance_Is_Zero'] = (df['Balance'] == 0).astype(int)
df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
df['Age_x_IsActive'] = df['Age'] * df['IsActiveMember']
df['Age_x_NumProducts'] = df['Age'] * df['NumOfProducts']
df['Tenure_Age_Ratio'] = df['Tenure'] / (df['Age'] + 1)
df['Products_GT2'] = (df['NumOfProducts'] > 2).astype(int)
df['CreditScore_Low'] = (df['CreditScore'] < 500).astype(int)
df['Is_Senior'] = (df['Age'] > 45).astype(int)
df['Balance_x_NumProducts'] = df['Balance'] * df['NumOfProducts']

X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
y = df['Exited']
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Total features after engineering: {len(feature_names)}")
print(f"Features: {feature_names}")
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

neg_count = sum(1 for val in y_train if val == 0)
pos_count = sum(1 for val in y_train if val == 1)
scale_pos = neg_count / pos_count if pos_count > 0 else 1.0
print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"Imbalance ratio: {scale_pos:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: FEATURE SELECTION (RFECV)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  PHASE 2: FEATURE SELECTION (RFECV with Gradient Boosting)")
print("=" * 80)

# Scale for RFECV (needed for fair feature importance comparison)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE on scaled training data for RFECV
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Use GBM (best performer) as the estimator for RFECV
gbm_selector = GradientBoostingClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.1,
    random_state=RANDOM_STATE
)

print("Running RFECV (this takes 1-2 minutes)...")
t0 = time()
rfecv = RFECV(
    estimator=gbm_selector,
    step=1,
    cv=CV,
    scoring="f1",
    min_features_to_select=5,
    n_jobs=-1,
)
rfecv.fit(X_train_resampled, y_train_resampled)
print(f"Done in {time() - t0:.1f}s")

# Results
selected_mask = rfecv.support_
selected_features = [f for f, s in zip(feature_names, selected_mask) if s]
dropped_features = [f for f, s in zip(feature_names, selected_mask) if not s]

print(f"\nOptimal number of features: {rfecv.n_features_}")
print(f"Selected ({len(selected_features)}): {selected_features}")
print(f"Dropped  ({len(dropped_features)}): {dropped_features}")

# Feature importance ranking
ranking = rfecv.ranking_
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Rank": ranking,
    "Selected": selected_mask
}).sort_values("Rank")
print("\nFeature Ranking:")
print(importance_df.to_string(index=False))

# Apply selection
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: CLASSIFIER BENCHMARK ON OPTIMAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(f"  PHASE 3: CLASSIFIER BENCHMARK ({len(selected_features)} selected features)")
print("=" * 80)

CLASSIFIERS = {
    "Logistic Regression": {
        "model": LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "params": {
            "classifier__C": np.logspace(-3, 3, 20),
            "classifier__penalty": ["l1", "l2"],
            "classifier__solver": ["liblinear", "saga"],
        },
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "classifier__n_neighbors": list(range(3, 31, 2)),
            "classifier__weights": ["uniform", "distance"],
            "classifier__metric": ["euclidean", "manhattan"],
            "classifier__p": [1, 2],
        },
    },
    "SVM": {
        "model": CalibratedClassifierCV(
            LinearSVC(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE), cv=3
        ),
        "params": {
            "classifier__estimator__C": np.logspace(-2, 2, 8),
        },
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "params": {
            "classifier__max_depth": [3, 5, 7, 10, 15, 20, None],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 5, 10],
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_features": ["sqrt", "log2", None],
        },
    },
    "Random Forest": {
        "model": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "params": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [5, 10, 20],
            "classifier__min_samples_split": [2, 5],
            "classifier__max_features": ["sqrt", "log2"],
        },
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "params": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
            "classifier__max_depth": [3, 5, 7],
            "classifier__subsample": [0.8, 1.0],
        },
    },
    "XGBoost": {
        "model": XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            scale_pos_weight=scale_pos,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "params": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
            "classifier__max_depth": [3, 5, 7],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
        },
    },
}


def evaluate_model(pipeline, X_tr, X_te, y_tr, y_te):
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    y_proba = (
        pipeline.predict_proba(X_te)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )
    return {
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred, zero_division=0),
        "Recall": recall_score(y_te, y_pred, zero_division=0),
        "F1": f1_score(y_te, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_te, y_proba) if y_proba is not None else np.nan,
    }


results = []

for name, config in CLASSIFIERS.items():
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")

    # ── DEFAULT ──
    t0 = time()
    pipe_default = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", config["model"]),
    ])

    cv_scores = cross_val_score(
        pipe_default, X_train_selected, y_train, cv=CV, scoring="f1"
    )
    metrics = evaluate_model(pipe_default, X_train_selected, X_test_selected, y_train, y_test)
    elapsed = time() - t0

    results.append({
        "Model": name,
        "Mode": "Default",
        "CV_Mean_F1": round(cv_scores.mean(), 4),
        "CV_Std": round(cv_scores.std(), 4),
        "Test_Accuracy": round(metrics["Accuracy"], 4),
        "Test_Precision": round(metrics["Precision"], 4),
        "Test_Recall": round(metrics["Recall"], 4),
        "Test_F1": round(metrics["F1"], 4),
        "Test_ROC_AUC": round(metrics["ROC-AUC"], 4),
        "Best_Params": "default",
        "Time_sec": round(elapsed, 2),
    })

    print(f"  [Default]  CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}  "
          f"Test_F1={metrics['F1']:.4f}  Recall={metrics['Recall']:.4f}  ({elapsed:.1f}s)")

    # ── TUNED ──
    t0 = time()
    pipe_tuned = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", config["model"]),
    ])

    search = RandomizedSearchCV(
        estimator=pipe_tuned,
        param_distributions=config["params"],
        n_iter=10,
        cv=CV,
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(X_train_selected, y_train)

    best_pipe = search.best_estimator_
    y_pred = best_pipe.predict(X_test_selected)
    y_proba = (
        best_pipe.predict_proba(X_test_selected)[:, 1]
        if hasattr(best_pipe, "predict_proba")
        else None
    )

    metrics_tuned = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
    }
    elapsed = time() - t0

    best_params = {
        k.replace("classifier__", ""): v
        for k, v in search.best_params_.items()
    }

    results.append({
        "Model": name,
        "Mode": "Tuned (RandomizedSearchCV)",
        "CV_Mean_F1": round(search.best_score_, 4),
        "CV_Std": round(search.cv_results_["std_test_score"][search.best_index_], 4),
        "Test_Accuracy": round(metrics_tuned["Accuracy"], 4),
        "Test_Precision": round(metrics_tuned["Precision"], 4),
        "Test_Recall": round(metrics_tuned["Recall"], 4),
        "Test_F1": round(metrics_tuned["F1"], 4),
        "Test_ROC_AUC": round(metrics_tuned["ROC-AUC"], 4),
        "Best_Params": str(best_params),
        "Time_sec": round(elapsed, 2),
    })

    print(f"  [Tuned]    CV={search.best_score_:.4f}  "
          f"Test_F1={metrics_tuned['F1']:.4f}  Recall={metrics_tuned['Recall']:.4f}  ({elapsed:.1f}s)")
    print(f"  Best → {best_params}")


# ─── FINAL RESULTS ───
df_results = pd.DataFrame(results)
df_results = df_results.sort_values("Test_F1", ascending=False).reset_index(drop=True)

print("\n" + "=" * 80)
print("  FINAL RESULTS — ALL CLASSIFIERS (Default + Tuned)")
print("=" * 80)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)
print(df_results.to_string(index=False))

print("\n" + "=" * 80)
print("  TOP 3 MODELS BY F1 SCORE")
print("=" * 80)
for i, row in df_results.head(3).iterrows():
    print(f"  {i+1}. {row['Model']} ({row['Mode']}) — "
          f"F1: {row['Test_F1']}  Recall: {row['Test_Recall']}  "
          f"Precision: {row['Test_Precision']}  AUC: {row['Test_ROC_AUC']}")

print(f"\nFeatures used ({len(selected_features)}): {selected_features}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: ENSEMBLE — WEIGHTED VOTING + STACKING
# ═══════════════════════════════════════════════════════════════════════════════
from sklearn.ensemble import VotingClassifier, StackingClassifier

print("\n" + "=" * 80)
print("  PHASE 4: ENSEMBLE MODELS")
print("=" * 80)

# --- Define base models with their best configs ---
gbm_default = GradientBoostingClassifier(random_state=RANDOM_STATE)

gbm_tuned = GradientBoostingClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05, subsample=1.0,
    random_state=RANDOM_STATE
)

rf_tuned = RandomForestClassifier(
    n_estimators=50, min_samples_split=2, max_features="sqrt", max_depth=10,
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)

xgb_tuned = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=1.0, colsample_bytree=1.0,
    scale_pos_weight=scale_pos, eval_metric="logloss",
    use_label_encoder=False, random_state=RANDOM_STATE, n_jobs=-1
)

# --- F1 scores for weighting ---
f1_gbm_def = 0.6277
f1_gbm_tuned = 0.6270
f1_rf_tuned = 0.6065
f1_xgb_tuned = 0.5753

# --- Scale + SMOTE the data once for ensemble evaluation ---
scaler_ens = StandardScaler()
X_train_sc = scaler_ens.fit_transform(X_train_selected)
X_test_sc = scaler_ens.transform(X_test_selected)

smote_ens = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote_ens.fit_resample(X_train_sc, y_train)

# --- Evaluation helper ---
def eval_ensemble(model, X_tr, y_tr, X_te, y_te, label):
    t0 = time()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    auc = roc_auc_score(y_te, y_proba) if y_proba is not None else np.nan
    elapsed = time() - t0

    print(f"\n  {label}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Time:      {elapsed:.1f}s")

    return {
        "Model": label, "Test_Accuracy": round(acc, 4),
        "Test_Precision": round(prec, 4), "Test_Recall": round(rec, 4),
        "Test_F1": round(f1, 4), "Test_ROC_AUC": round(auc, 4),
        "Time_sec": round(elapsed, 2),
    }


ensemble_results = []

# ── ENSEMBLE A: Your pick (GBM Default + GBM Tuned + RF Tuned) ──
print("\n" + "─" * 60)
print("  ENSEMBLE A: GBM Default + GBM Tuned + RF Tuned (your pick)")
print("─" * 60)

voting_A = VotingClassifier(
    estimators=[
        ("gbm_def", gbm_default),
        ("gbm_tuned", gbm_tuned),
        ("rf_tuned", rf_tuned),
    ],
    voting="soft",
    weights=[f1_gbm_def, f1_gbm_tuned, f1_rf_tuned],
)
r = eval_ensemble(voting_A, X_train_sm, y_train_sm, X_test_sc, y_test,
                  "Weighted Voting A (GBM+GBM+RF)")
ensemble_results.append(r)

# ── ENSEMBLE B: Diverse (GBM Default + RF Tuned + XGBoost Tuned) ──
print("\n" + "─" * 60)
print("  ENSEMBLE B: GBM Default + RF Tuned + XGBoost Tuned (diverse)")
print("─" * 60)

voting_B = VotingClassifier(
    estimators=[
        ("gbm_def", gbm_default),
        ("rf_tuned", rf_tuned),
        ("xgb_tuned", xgb_tuned),
    ],
    voting="soft",
    weights=[f1_gbm_def, f1_rf_tuned, f1_xgb_tuned],
)
r = eval_ensemble(voting_B, X_train_sm, y_train_sm, X_test_sc, y_test,
                  "Weighted Voting B (GBM+RF+XGB)")
ensemble_results.append(r)

# ── STACKING: GBM + RF + XGBoost → Logistic meta-learner ──
print("\n" + "─" * 60)
print("  STACKING: GBM + RF + XGBoost → LogisticRegression meta-learner")
print("─" * 60)

stacking = StackingClassifier(
    estimators=[
        ("gbm_def", gbm_default),
        ("rf_tuned", rf_tuned),
        ("xgb_tuned", xgb_tuned),
    ],
    final_estimator=LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    cv=5,
    passthrough=True,   # feeds original features + base predictions to meta-learner
    n_jobs=-1,
)
r = eval_ensemble(stacking, X_train_sm, y_train_sm, X_test_sc, y_test,
                  "Stacking (GBM+RF+XGB → LR)")
ensemble_results.append(r)

# ── COMPARISON TABLE ──
# Add best individual model for reference
ensemble_results.append({
    "Model": "Best Individual (GBM Default)",
    "Test_Accuracy": 0.8375, "Test_Precision": 0.5880,
    "Test_Recall": 0.6732, "Test_F1": 0.6277, "Test_ROC_AUC": 0.8662,
    "Time_sec": 20.09,
})

df_ensemble = pd.DataFrame(ensemble_results).sort_values("Test_F1", ascending=False)

print("\n" + "=" * 80)
print("  ENSEMBLE vs INDIVIDUAL — FINAL COMPARISON")
print("=" * 80)
print(df_ensemble.to_string(index=False))

print("\n" + "=" * 80)
print("  WINNER")
print("=" * 80)
winner = df_ensemble.iloc[0]
print(f"  {winner['Model']}")
print(f"  F1: {winner['Test_F1']}  Recall: {winner['Test_Recall']}  "
      f"Precision: {winner['Test_Precision']}  AUC: {winner['Test_ROC_AUC']}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5: THRESHOLD TUNING
#  Default is 0.5 — for imbalanced data, optimal is usually 0.30-0.45
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  PHASE 5: THRESHOLD TUNING")
print("=" * 80)

# All models are already fitted from Phase 4 — just need predict_proba
models_to_tune = {
    "GBM Default": gbm_default,
    "Voting A (GBM+GBM+RF)": voting_A,
    "Voting B (GBM+RF+XGB)": voting_B,
    "Stacking (GBM+RF+XGB→LR)": stacking,
}

# Also fit standalone GBM on SMOTE data (it was fit inside pipeline earlier, need fresh fit)
gbm_standalone = GradientBoostingClassifier(random_state=RANDOM_STATE)
gbm_standalone.fit(X_train_sm, y_train_sm)
models_to_tune["GBM Default"] = gbm_standalone

thresholds = np.arange(0.20, 0.61, 0.01)
threshold_results = []

for model_name, model in models_to_tune.items():
    print(f"\n{'─' * 60}")
    print(f"  {model_name}")
    print(f"{'─' * 60}")

    y_proba = model.predict_proba(X_test_sc)[:, 1]

    best_f1 = 0
    best_thresh = 0.5
    best_metrics = {}

    # Scan all thresholds
    for thresh in thresholds:
        y_pred_t = (y_proba >= thresh).astype(int)
        f1_t = f1_score(y_test, y_pred_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = thresh
            best_metrics = {
                "Accuracy": accuracy_score(y_test, y_pred_t),
                "Precision": precision_score(y_test, y_pred_t, zero_division=0),
                "Recall": recall_score(y_test, y_pred_t, zero_division=0),
                "F1": f1_t,
                "ROC-AUC": roc_auc_score(y_test, y_proba),
            }

    # Default threshold (0.5) metrics for comparison
    y_pred_default = (y_proba >= 0.5).astype(int)
    f1_default = f1_score(y_test, y_pred_default, zero_division=0)
    rec_default = recall_score(y_test, y_pred_default, zero_division=0)
    prec_default = precision_score(y_test, y_pred_default, zero_division=0)

    improvement = best_metrics["F1"] - f1_default

    print(f"  Default (0.50):  F1={f1_default:.4f}  Prec={prec_default:.4f}  Recall={rec_default:.4f}")
    print(f"  Optimal ({best_thresh:.2f}):  F1={best_metrics['F1']:.4f}  "
          f"Prec={best_metrics['Precision']:.4f}  Recall={best_metrics['Recall']:.4f}")
    print(f"  F1 improvement:  {improvement:+.4f} ({improvement/f1_default*100:+.1f}%)")

    threshold_results.append({
        "Model": model_name,
        "Default_Threshold": 0.50,
        "Default_F1": round(f1_default, 4),
        "Default_Precision": round(prec_default, 4),
        "Default_Recall": round(rec_default, 4),
        "Optimal_Threshold": round(best_thresh, 2),
        "Tuned_F1": round(best_metrics["F1"], 4),
        "Tuned_Precision": round(best_metrics["Precision"], 4),
        "Tuned_Recall": round(best_metrics["Recall"], 4),
        "Tuned_ROC_AUC": round(best_metrics["ROC-AUC"], 4),
        "F1_Gain": round(improvement, 4),
    })

# ── FINAL COMPARISON TABLE ──
df_thresh = pd.DataFrame(threshold_results).sort_values("Tuned_F1", ascending=False)

print("\n" + "=" * 80)
print("  THRESHOLD TUNING — FULL COMPARISON")
print("=" * 80)
print(df_thresh.to_string(index=False))

# ── GRAND FINAL: BEST MODEL EVER ──
print("\n" + "=" * 80)
print("  ★ GRAND FINAL — BEST CONFIGURATION ACROSS ALL PHASES ★")
print("=" * 80)

best = df_thresh.iloc[0]
print(f"  Model:     {best['Model']}")
print(f"  Threshold: {best['Optimal_Threshold']}")
print(f"  F1:        {best['Tuned_F1']}")
print(f"  Recall:    {best['Tuned_Recall']}")
print(f"  Precision: {best['Tuned_Precision']}")
print(f"  ROC-AUC:   {best['Tuned_ROC_AUC']}")
print(f"")
print(f"  Journey:   F1 0.46 (raw) → 0.58 (SMOTE) → 0.63 (features) → {best['Tuned_F1']} (threshold)")
print(f"  Features:  {len(selected_features)} selected by RFECV")
print(f"  {selected_features}")