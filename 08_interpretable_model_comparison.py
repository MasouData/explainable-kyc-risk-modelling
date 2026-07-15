# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %pip install interpret

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load and prepare the dataset
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

features_sdf = spark.table(
    "main.default.kyc_customer_features"
)

source_numeric_features = [
    "transaction_count",
    "counterparty_count",
    "transactions_per_active_day",
    "maximum_amount_zscore",
    "cross_bank_ratio",
    "self_transfer_ratio",
    "unusually_large_tx_ratio",
    "incoming_outgoing_ratio",
    "account_count",
    "bank_count",
    "currency_count",
    "payment_format_count"
]

categorical_features = [
    "customer_type",
    "country_risk",
    "pep_status",
    "ubo_complexity",
    "industry_risk",
    "onboarding_channel",
    "kyc_documentation"
]

required_columns = [
    "entity_id",
    "future_aml_event",
    "kyc_risk_score",
    *source_numeric_features,
    *categorical_features
]

model_pdf = (
    features_sdf
    .select(*required_columns)
    .toPandas()
)

print(f"Customers: {len(model_pdf):,}")
print(
    f"Positive rate: "
    f"{model_pdf['future_aml_event'].mean():.4%}"
)

# COMMAND ----------

# DBTITLE 1,Transform highly skewed features
log_features = [
    "transaction_count",
    "counterparty_count",
    "transactions_per_active_day",
    "incoming_outgoing_ratio",
    "account_count",
    "bank_count",
    "currency_count",
    "payment_format_count"
]

for feature in log_features:
    model_pdf[f"log_{feature}"] = np.log1p(
        model_pdf[feature]
        .astype(float)
        .clip(lower=0)
    )

# Restrict extreme z-scores to a reasonable range.
model_pdf["maximum_amount_zscore"] = (
    model_pdf["maximum_amount_zscore"]
    .astype(float)
    .clip(lower=-5, upper=10)
)

numeric_features = [
    "log_transaction_count",
    "log_counterparty_count",
    "log_transactions_per_active_day",
    "maximum_amount_zscore",
    "cross_bank_ratio",
    "self_transfer_ratio",
    "unusually_large_tx_ratio",
    "log_incoming_outgoing_ratio",
    "log_account_count",
    "log_bank_count",
    "log_currency_count",
    "log_payment_format_count"
]

model_features = numeric_features + categorical_features

model_pdf = model_pdf.replace(
    [np.inf, -np.inf],
    np.nan
)

for feature in numeric_features:
    model_pdf[feature] = (
        model_pdf[feature]
        .astype(float)
        .fillna(0.0)
    )

for feature in categorical_features:
    model_pdf[feature] = (
        model_pdf[feature]
        .fillna("Unknown")
        .astype(str)
    )

# COMMAND ----------

# DBTITLE 1,Feature Preprocessing
# MAGIC %md
# MAGIC ## 🔧 Normalizing Skewed Data
# MAGIC
# MAGIC Applies **log transformation** to 8 highly skewed count features (transaction_count, account_count, etc.) using `log1p()` to compress their wide ranges. Also **clips extreme z-scores** to [-5, 10] to prevent outliers from dominating the model. This makes the features more suitable for linear models like Logistic Regression.

# COMMAND ----------

# DBTITLE 1,Performance Comparison
# MAGIC %md
# MAGIC ## 📊 Results: LR and EBM Are Nearly Identical
# MAGIC
# MAGIC **At 10% review capacity** (reviewing the top 10% riskiest customers):
# MAGIC
# MAGIC | Model | Recall | Precision | PR-AUC |
# MAGIC |-------|--------|-----------|--------|
# MAGIC | **EBM** | 34.3% | 4.0% | 0.094 |
# MAGIC | **LR** | 33.0% | 4.0% | 0.093 |
# MAGIC | **KYC Scorecard** | 11.8% | 1.3% | 0.012 |
# MAGIC
# MAGIC **Key Findings**:
# MAGIC * **EBM marginally better** (1.3% more recall) but difference is small
# MAGIC * **Both ML models vastly outperform** the traditional KYC scorecard (3x better recall)
# MAGIC * **EBM's complexity** may not justify the minimal performance gain over simpler LR

# COMMAND ----------

# DBTITLE 1,Precision-Recall Trade-off
# MAGIC %md
# MAGIC ## 📈 PR Curves: Nearly Overlapping
# MAGIC
# MAGIC The **precision-recall curves** show LR and EBM are nearly identical across all threshold values. Both models far exceed the baseline (population prevalence at 1.15%). This confirms that **behavioral features drive performance**, not model complexity — a simpler linear model captures almost all available signal.

# COMMAND ----------

# DBTITLE 1,Probability Calibration
# MAGIC %md
# MAGIC ## 🎯 Calibration: Are Probabilities Reliable?
# MAGIC
# MAGIC **Calibration** measures whether predicted probabilities match reality. If a model predicts 20% risk for 100 customers, ideally ~20 should be actual launderers.
# MAGIC
# MAGIC Both LR and EBM track the diagonal "perfect calibration" line closely, meaning their probability estimates are **trustworthy** — crucial for compliance teams making review decisions based on model scores.

# COMMAND ----------

# DBTITLE 1,Champion Selection Logic
# MAGIC %md
# MAGIC ## ✅ Why Logistic Regression Won
# MAGIC
# MAGIC **Decision criteria** (predeclared before testing):
# MAGIC * EBM must provide **≥5% recall improvement** AND **≥2% PR-AUC improvement** AND **acceptable calibration** (Brier score ≤ 110% of LR)
# MAGIC
# MAGIC **Actual results**:
# MAGIC * Recall improvement: **1.3%** (below 5% threshold)
# MAGIC * PR-AUC improvement: **0.08%** (below 2% threshold)
# MAGIC * Calibration: Acceptable ✓
# MAGIC
# MAGIC **Verdict**: **Logistic Regression** chosen because EBM's minimal performance gain doesn't justify its added complexity, computational cost, and validation overhead. LR offers **simplicity, interpretability, and regulatory acceptance** with nearly identical predictive power.

# COMMAND ----------

# DBTITLE 1,Reproducible train, validation and test split
# 60% training
# 20% validation
# 20% final test

SEED = 42

all_indices = np.arange(len(model_pdf))
target = model_pdf["future_aml_event"].to_numpy()

train_val_indices, test_indices = train_test_split(
    all_indices,
    test_size=0.20,
    stratify=target,
    random_state=SEED
)

train_indices, validation_indices = train_test_split(
    train_val_indices,
    test_size=0.25,
    stratify=target[train_val_indices],
    random_state=SEED
)

X_train = model_pdf.iloc[train_indices][model_features].copy()
X_validation = model_pdf.iloc[validation_indices][model_features].copy()
X_test = model_pdf.iloc[test_indices][model_features].copy()

y_train = model_pdf.iloc[train_indices]["future_aml_event"].to_numpy()
y_validation = model_pdf.iloc[validation_indices]["future_aml_event"].to_numpy()
y_test = model_pdf.iloc[test_indices]["future_aml_event"].to_numpy()

scorecard_validation = (
    model_pdf.iloc[validation_indices]["kyc_risk_score"]
    .to_numpy() / 100.0
)

scorecard_test = (
    model_pdf.iloc[test_indices]["kyc_risk_score"]
    .to_numpy() / 100.0
)

print(f"Training customers:   {len(X_train):,}")
print(f"Validation customers: {len(X_validation):,}")
print(f"Test customers:       {len(X_test):,}")

print(f"Training positives:   {y_train.sum():,}")
print(f"Validation positives: {y_validation.sum():,}")
print(f"Test positives:       {y_test.sum():,}")

# COMMAND ----------

# DBTITLE 1,Train logistic regression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

logistic_model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "model",
            LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=2000,
                random_state=SEED
            )
        )
    ]
)

logistic_model.fit(X_train, y_train)

lr_validation_probability = logistic_model.predict_proba(
    X_validation
)[:, 1]

lr_test_probability = logistic_model.predict_proba(
    X_test
)[:, 1]

# COMMAND ----------

# DBTITLE 1,Train Explainable Boosting Machine
# interactions=0 keeps the first version completely additive and easier to validate
from interpret.glassbox import ExplainableBoostingClassifier

feature_types = (
    ["continuous"] * len(numeric_features)
    + ["nominal"] * len(categorical_features)
)

ebm_model = ExplainableBoostingClassifier(
    feature_names=model_features,
    feature_types=feature_types,

    # Purely additive model for stronger interpretability.
    interactions=0,

    # Lightweight settings suitable for Databricks Free Edition.
    max_bins=64,
    outer_bags=4,
    validation_size=0.15,
    learning_rate=0.03,
    max_rounds=2000,
    early_stopping_rounds=50,
    min_samples_leaf=20,

    random_state=SEED,
    n_jobs=-2
)

ebm_model.fit(X_train, y_train)

ebm_validation_probability = ebm_model.predict_proba(
    X_validation
)[:, 1]

ebm_test_probability = ebm_model.predict_proba(
    X_test
)[:, 1]

# COMMAND ----------

# DBTITLE 1,Evaluate at 10% review capacity
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score
)

REVIEW_CAPACITY = 0.10


def capacity_threshold(validation_scores, capacity=0.10):
    return float(
        np.quantile(
            validation_scores,
            1.0 - capacity
        )
    )


def evaluate_model(
    model_name,
    validation_scores,
    test_scores,
    y_test,
    probabilistic=True
):
    threshold = capacity_threshold(
        validation_scores,
        REVIEW_CAPACITY
    )

    test_predictions = (
        test_scores >= threshold
    ).astype(int)

    review_rate = test_predictions.mean()
    precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    prevalence = y_test.mean()

    return {
        "model": model_name,
        "threshold": threshold,
        "test_review_rate": review_rate,
        "pr_auc": average_precision_score(
            y_test,
            test_scores
        ),
        "roc_auc": roc_auc_score(
            y_test,
            test_scores
        ),
        "precision_at_capacity": precision,
        "recall_at_capacity": recall,
        "lift_at_capacity": (
            precision / prevalence
            if prevalence > 0 else np.nan
        ),
        "brier_score": (
            brier_score_loss(y_test, test_scores)
            if probabilistic else np.nan
        )
    }


model_results = pd.DataFrame([
    evaluate_model(
        "KYC Scorecard",
        scorecard_validation,
        scorecard_test,
        y_test,
        probabilistic=False
    ),
    evaluate_model(
        "Logistic Regression",
        lr_validation_probability,
        lr_test_probability,
        y_test
    ),
    evaluate_model(
        "Explainable Boosting Machine",
        ebm_validation_probability,
        ebm_test_probability,
        y_test
    )
])

model_results = model_results.sort_values(
    "recall_at_capacity",
    ascending=False
)

display(spark.createDataFrame(model_results))

# COMMAND ----------

# DBTITLE 1,Save results and test predictions
output_directory = (
    "/Volumes/main/default/aml_dataset/"
    "kyc_model_outputs"
)

os.makedirs(output_directory, exist_ok=True)

model_results.to_csv(
    f"{output_directory}/model_comparison.csv",
    index=False
)

test_predictions_pdf = pd.DataFrame({
    "entity_id": model_pdf.iloc[test_indices][
        "entity_id"
    ].to_numpy(),

    "future_aml_event": y_test,

    "scorecard_score": scorecard_test,
    "logistic_probability": lr_test_probability,
    "ebm_probability": ebm_test_probability
})

spark.createDataFrame(test_predictions_pdf) \
    .write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(
        "main.default.kyc_model_test_predictions"
    )

spark.createDataFrame(model_results) \
    .write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(
        "main.default.kyc_model_comparison"
    )

# COMMAND ----------

display(model_results)

# COMMAND ----------

# DBTITLE 1,Model comparison Visualization
import matplotlib.pyplot as plt

comparison_plot = (
    model_results
    .set_index("model")[
        [
            "pr_auc",
            "recall_at_capacity",
            "precision_at_capacity"
        ]
    ]
)

ax = comparison_plot.plot(
    kind="bar",
    figsize=(11, 6)
)

ax.set_title(
    "KYC Model Comparison at 10% Review Capacity"
)
ax.set_xlabel("")
ax.set_ylabel("Metric value")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=15)
ax.legend(
    [
        "PR-AUC",
        "Recall at 10%",
        "Precision at 10%"
    ]
)

plt.tight_layout()
plt.savefig(
    f"{output_directory}/model_comparison.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()

# COMMAND ----------

# DBTITLE 1,Precision–recall curves
from sklearn.metrics import precision_recall_curve

plt.figure(figsize=(9, 6))

for model_name, probabilities in [
    (
        "Logistic Regression",
        lr_test_probability
    ),
    (
        "Explainable Boosting Machine",
        ebm_test_probability
    )
]:
    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities
    )

    plt.plot(
        recall,
        precision,
        label=model_name
    )

plt.axhline(
    y=y_test.mean(),
    linestyle="--",
    label="Population prevalence"
)

plt.title("Precision–Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{output_directory}/precision_recall_curve.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()

# COMMAND ----------

# DBTITLE 1,Calibration
import numpy as np

for name, probabilities in [
    ("Logistic Regression", lr_test_probability),
    ("EBM", ebm_test_probability)
]:
    print(f"\n{name}")
    print(f"Minimum: {probabilities.min():.6f}")
    print(f"Median:  {np.median(probabilities):.6f}")
    print(f"95th percentile: {np.quantile(probabilities, 0.95):.6f}")
    print(f"99th percentile: {np.quantile(probabilities, 0.99):.6f}")
    print(f"Maximum: {probabilities.max():.6f}")

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 6))

all_predicted_bins = []
all_observed_bins = []

for model_name, probabilities in [
    ("Logistic Regression", lr_test_probability),
    ("Explainable Boosting Machine", ebm_test_probability)
]:
    observed_rate, predicted_rate = calibration_curve(
        y_test,
        probabilities,
        n_bins=10,
        strategy="quantile"
    )

    all_predicted_bins.extend(predicted_rate)
    all_observed_bins.extend(observed_rate)

    plt.plot(
        predicted_rate,
        observed_rate,
        marker="o",
        linewidth=2,
        label=model_name
    )

# Set the chart range based on the actual calibration points.
axis_limit = max(
    max(all_predicted_bins),
    max(all_observed_bins)
) * 1.15

axis_limit = max(axis_limit, 0.02)

plt.plot(
    [0, axis_limit],
    [0, axis_limit],
    linestyle="--",
    label="Perfect calibration"
)

plt.axhline(
    y=np.mean(y_test),
    linestyle=":",
    label=f"Positive rate: {np.mean(y_test):.2%}"
)

plt.xlim(0, axis_limit)
plt.ylim(0, axis_limit)

plt.title("Probability Calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed positive rate")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    f"{output_directory}/calibration_zoomed.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()    

# COMMAND ----------

# DBTITLE 1,Understanding the Calibration Code
# MAGIC %md
# MAGIC ## 🔍 What This Code Does
# MAGIC
# MAGIC **Part 1: Probability Distribution Statistics**
# MAGIC ```python
# MAGIC for name, probabilities in [("LR", lr_test_probability), ("EBM", ebm_test_probability)]:
# MAGIC     print min, median, 95th/99th percentile, max
# MAGIC ```
# MAGIC Shows the range and spread of predicted probabilities for each model.
# MAGIC
# MAGIC **Part 2: Calibration Curve**
# MAGIC ```python
# MAGIC calibration_curve(y_test, probabilities, n_bins=10, strategy="quantile")
# MAGIC ```
# MAGIC * **Divides predictions into 10 equal-sized groups** (quantile bins)
# MAGIC * **For each group**: calculates the **mean predicted probability** and the **actual observed rate** of launderers
# MAGIC * **Plots predicted vs observed** — if calibrated, points follow the diagonal line
# MAGIC
# MAGIC **Example**: If bin 5 has mean predicted probability of 2.5%, and 2.4% actually are launderers → well calibrated ✓
# MAGIC
# MAGIC **The Diagonal Line** = "Perfect calibration" (predicted = actual)
# MAGIC
# MAGIC **Why It Matters**: If the model predicts 3% risk, compliance teams need to trust that ~3% of those customers really are high-risk. Poorly calibrated models give misleading probabilities, even if they rank correctly.

# COMMAND ----------

# DBTITLE 1,EBM global feature importance
ebm_importance = pd.DataFrame({
    "feature": ebm_model.term_names_,
    "importance": ebm_model.term_importances()
})

ebm_importance = (
    ebm_importance
    .sort_values(
        "importance",
        ascending=False
    )
    .head(12)
    .sort_values("importance")
)

plt.figure(figsize=(9, 6))
plt.barh(
    ebm_importance["feature"],
    ebm_importance["importance"]
)

plt.title(
    "EBM Global Feature Importance"
)
plt.xlabel("Average absolute contribution")
plt.ylabel("")
plt.tight_layout()

plt.savefig(
    f"{output_directory}/ebm_global_importance.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()

# COMMAND ----------

# DBTITLE 1,Local explanation for one high-risk customer
positive_positions = np.where(
    y_test == 1
)[0]

highest_risk_positive_position = positive_positions[
    np.argmax(
        ebm_test_probability[positive_positions]
    )
]

customer_row = X_test.iloc[
    [highest_risk_positive_position]
]

customer_id = model_pdf.iloc[
    test_indices[highest_risk_positive_position]
]["entity_id"]

predicted_probability = ebm_test_probability[
    highest_risk_positive_position
]

term_contributions = ebm_model.eval_terms(
    customer_row
)[0]

local_explanation = pd.DataFrame({
    "feature": ebm_model.term_names_,
    "contribution": term_contributions
})

local_explanation["absolute_contribution"] = (
    local_explanation["contribution"].abs()
)

local_explanation = (
    local_explanation
    .sort_values(
        "absolute_contribution",
        ascending=False
    )
    .head(10)
    .sort_values("contribution")
)

plt.figure(figsize=(10, 6))
plt.barh(
    local_explanation["feature"],
    local_explanation["contribution"]
)

plt.axvline(0)
plt.title(
    f"Local EBM Explanation — {customer_id}\n"
    f"Predicted probability: "
    f"{predicted_probability:.2%}"
)
plt.xlabel(
    "Contribution to model score"
)
plt.ylabel("")
plt.tight_layout()

plt.savefig(
    f"{output_directory}/ebm_local_explanation.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()

# COMMAND ----------

print(output_directory)

# COMMAND ----------

# DBTITLE 1,Local explanation for the same customer - Logistic Regression
# Same customer as EBM explanation
lr_predicted_probability = lr_test_probability[
    highest_risk_positive_position
]

# Get the preprocessor and model from pipeline
preprocessor = logistic_model.named_steps["preprocessing"]
lr_classifier = logistic_model.named_steps["model"]

# Transform the customer data through preprocessing
customer_transformed = preprocessor.transform(customer_row)

# Get feature names after preprocessing
feature_names_out = []

# Numeric features (standardized)
for feature in numeric_features:
    feature_names_out.append(feature)

# Categorical features (one-hot encoded)
categorical_encoder = preprocessor.named_transformers_["categorical"]
for i, feature in enumerate(categorical_features):
    categories = categorical_encoder.categories_[i]
    for category in categories:
        feature_names_out.append(f"{feature}_{category}")

# Get coefficients
coefficients = lr_classifier.coef_[0]

# Calculate contributions (coefficient * feature_value)
# Handle both sparse and dense arrays
if hasattr(customer_transformed, "toarray"):
    customer_values = customer_transformed.toarray()[0]
else:
    customer_values = customer_transformed[0]

contributions = coefficients * customer_values

# Create explanation dataframe
lr_local_explanation = pd.DataFrame({
    "feature": feature_names_out,
    "contribution": contributions
})

# Add absolute contribution for sorting
lr_local_explanation["absolute_contribution"] = (
    lr_local_explanation["contribution"].abs()
)

# Get top 10 contributors
lr_local_explanation = (
    lr_local_explanation
    .sort_values(
        "absolute_contribution",
        ascending=False
    )
    .head(10)
    .sort_values("contribution")
)

plt.figure(figsize=(10, 6))
plt.barh(
    lr_local_explanation["feature"],
    lr_local_explanation["contribution"]
)

plt.axvline(0, color="black", linewidth=0.8)
plt.title(
    f"Local LR Explanation — {customer_id}\n"
    f"Predicted probability: "
    f"{lr_predicted_probability:.2%}"
)
plt.xlabel(
    "Contribution to model score"
)
plt.ylabel("")
plt.tight_layout()

plt.savefig(
    f"{output_directory}/lr_local_explanation.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()

# COMMAND ----------

# DBTITLE 1,predeclared rule to choose best model
results_by_model = model_results.set_index("model")

lr_result = results_by_model.loc[
    "Logistic Regression"
]

ebm_result = results_by_model.loc[
    "Explainable Boosting Machine"
]

recall_improvement = (
    ebm_result["recall_at_capacity"]
    - lr_result["recall_at_capacity"]
)

pr_auc_improvement = (
    ebm_result["pr_auc"]
    - lr_result["pr_auc"]
)

calibration_acceptable = (
    ebm_result["brier_score"]
    <= lr_result["brier_score"] * 1.10
)

if (
    recall_improvement >= 0.05
    and pr_auc_improvement >= 0.02
    and calibration_acceptable
):
    champion = "Explainable Boosting Machine"
    decision_reason = (
        "EBM provides material performance improvement "
        "while retaining acceptable calibration and "
        "intrinsic interpretability."
    )
else:
    champion = "Logistic Regression"
    decision_reason = (
        "EBM does not provide enough additional value "
        "to justify its greater modelling and validation "
        "complexity."
    )

print(f"Champion: {champion}")
print(decision_reason)

# COMMAND ----------

# DBTITLE 1,Why LR Won: Decision Breakdown
# MAGIC %md
# MAGIC ## 🎯 The Decision Logic
# MAGIC
# MAGIC This cell implements a **predeclared rule** set **before testing** to avoid cherry-picking:
# MAGIC
# MAGIC **EBM wins only if:**
# MAGIC 1. Recall improvement ≥ 5% **AND**
# MAGIC 2. PR-AUC improvement ≥ 2% **AND** 
# MAGIC 3. Calibration acceptable (Brier score ≤ 110% of LR)
# MAGIC
# MAGIC **Actual measurements:**
# MAGIC * Recall improvement: **1.3%** ❌ (below 5% threshold)
# MAGIC * PR-AUC improvement: **0.08%** ❌ (below 2% threshold)  
# MAGIC * Calibration: **Acceptable** ✓
# MAGIC
# MAGIC **Result: Logistic Regression chosen** ✅
# MAGIC
# MAGIC **Why this matters:**
# MAGIC * EBM's **1.3% better recall** doesn't justify its added complexity, longer training time, harder validation, and lower regulatory acceptance
# MAGIC * LR offers **near-identical performance** (33.0% vs 34.3% recall) with **far simpler implementation**
# MAGIC * In financial compliance, **explainability and auditability** often trump marginal performance gains
# MAGIC * The predeclared thresholds (5% and 2%) represent the **minimum improvement** needed to justify switching from a simple baseline to a more complex model