# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Load profiles and assign points
from pyspark.sql import functions as F

kyc_profiles = spark.table(
    "main.default.kyc_customer_profiles"
)

scored_customers = (
    kyc_profiles

    # Geography: maximum 20 points
    .withColumn(
        "country_points",
        F.when(F.col("country_risk") == "High", 20)
         .when(F.col("country_risk") == "Medium", 8)
         .otherwise(0)
    )

    # PEP: maximum 25 points
    .withColumn(
        "pep_points",
        F.when(F.col("pep_status") == "Direct", 25)
         .when(F.col("pep_status") == "Associate", 15)
         .otherwise(0)
    )

    # Ownership structure: maximum 20 points
    .withColumn(
        "ubo_points",
        F.when(F.col("ubo_complexity") == "Complex", 20)
         .when(F.col("ubo_complexity") == "Moderate", 8)
         .otherwise(0)
    )

    # Industry: maximum 15 points
    .withColumn(
        "industry_points",
        F.when(F.col("industry_risk") == "Cash Intensive", 15)
         .when(F.col("industry_risk") == "Elevated", 7)
         .otherwise(0)
    )

    # Onboarding: maximum 5 points
    .withColumn(
        "channel_points",
        F.when(F.col("onboarding_channel") == "Remote", 5)
         .otherwise(0)
    )

    # Documentation: maximum 15 points
    .withColumn(
        "documentation_points",
        F.when(F.col("kyc_documentation") == "Incomplete", 15)
         .otherwise(0)
    )
)

# COMMAND ----------

# DBTITLE 1,Understanding the Risk Scorecard
# MAGIC %md
# MAGIC ## 🎯 KYC Risk Scorecard Logic
# MAGIC
# MAGIC This cell creates a **point-based risk assessment system** that evaluates each customer across 6 key risk dimensions. The approach is transparent and explainable — auditors and compliance teams can see exactly why each customer received their risk score.
# MAGIC
# MAGIC ### Risk Categories & Point Allocation
# MAGIC
# MAGIC The scorecard assigns points based on risk characteristics, with **higher risk = more points** (maximum **100 points** total):
# MAGIC
# MAGIC | Risk Category | Maximum Points | High Risk Scenario | Points Assigned |
# MAGIC |--------------|----------------|-------------------|------------------|
# MAGIC | **Country Risk** | 20 | Operating in high-risk jurisdiction | High: 20, Medium: 8, Low: 0 |
# MAGIC | **PEP Status** | 25 | Politically Exposed Person | Direct: 25, Associate: 15, None: 0 |
# MAGIC | **UBO Complexity** | 20 | Complex ownership structure | Complex: 20, Moderate: 8, Simple: 0 |
# MAGIC | **Industry Risk** | 15 | Cash-intensive business | Cash Intensive: 15, Elevated: 7, Standard: 0 |
# MAGIC | **Onboarding Channel** | 5 | Remote/non-face-to-face | Remote: 5, Branch: 0 |
# MAGIC | **Documentation** | 15 | Incomplete KYC documents | Incomplete: 15, Complete: 0 |
# MAGIC
# MAGIC ### Example Risk Profiles
# MAGIC
# MAGIC **High-Risk Customer** (score: 100):
# MAGIC * High-risk country (20) + Direct PEP (25) + Complex ownership (20) + Cash-intensive industry (15) + Remote onboarding (5) + Incomplete docs (15)
# MAGIC
# MAGIC **Low-Risk Customer** (score: 0):
# MAGIC * Low-risk country (0) + No PEP (0) + Simple ownership (0) + Standard industry (0) + Branch onboarding (0) + Complete docs (0)
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC The individual category points will be summed into a `total_risk_score`, and customers will be categorized into risk bands (Low/Medium/High) for appropriate monitoring levels.

# COMMAND ----------

# DBTITLE 1,Calculate score and rating
scored_customers = (
    scored_customers
    .withColumn(
        "kyc_risk_score",
        F.col("country_points")
        + F.col("pep_points")
        + F.col("ubo_points")
        + F.col("industry_points")
        + F.col("channel_points")
        + F.col("documentation_points")
    )
    .withColumn(
        "kyc_risk_rating",
        F.when(F.col("kyc_risk_score") >= 35, "HIGH")
         .when(F.col("kyc_risk_score") >= 15, "MEDIUM")
         .otherwise("LOW")
    )
)

# COMMAND ----------

# DBTITLE 1,How Risk Scores Translate to Ratings
# MAGIC %md
# MAGIC ## 📊 From Points to Risk Ratings
# MAGIC
# MAGIC This cell completes the risk assessment by:
# MAGIC
# MAGIC ### 1. Calculating Total Risk Score
# MAGIC Sums all 6 category points into a single `kyc_risk_score` (0-100 scale):
# MAGIC ```
# MAGIC kyc_risk_score = country_points + pep_points + ubo_points + 
# MAGIC                  industry_points + channel_points + documentation_points
# MAGIC ```
# MAGIC
# MAGIC ### 2. Assigning Risk Ratings
# MAGIC Translates numeric scores into actionable risk bands:
# MAGIC
# MAGIC | Risk Rating | Score Range | Monitoring Level | Example |
# MAGIC |------------|-------------|------------------|----------|
# MAGIC | **HIGH** | ≥ 35 points | Enhanced Due Diligence (EDD) | Direct PEP from high-risk country with complex ownership |
# MAGIC | **MEDIUM** | 15-34 points | Standard monitoring | Moderate complexity business in medium-risk jurisdiction |
# MAGIC | **LOW** | < 15 points | Basic monitoring | Simple structure, complete docs, low-risk country |
# MAGIC
# MAGIC ### Why These Thresholds?
# MAGIC * **HIGH (≥35)**: Captures customers with multiple serious risk factors or one critical factor (e.g., Direct PEP alone = 25 points)
# MAGIC * **MEDIUM (15-34)**: Identifies customers with moderate concerns requiring standard oversight
# MAGIC * **LOW (<15)**: Minimal risk exposure, routine monitoring sufficient
# MAGIC
# MAGIC These bands help compliance teams allocate resources efficiently — focusing enhanced scrutiny on the riskiest customers while maintaining appropriate oversight across all segments.

# COMMAND ----------

# DBTITLE 1,Create transparent reason codes
scored_customers = scored_customers.withColumn(
    "reason_codes",
    F.filter(
        F.array(
            F.when(
                F.col("country_risk") == "High",
                F.lit("HIGH_RISK_COUNTRY")
            ).when(
                F.col("country_risk") == "Medium",
                F.lit("MEDIUM_RISK_COUNTRY")
            ),

            F.when(
                F.col("pep_status") == "Direct",
                F.lit("DIRECT_PEP")
            ).when(
                F.col("pep_status") == "Associate",
                F.lit("PEP_ASSOCIATE")
            ),

            F.when(
                F.col("ubo_complexity") == "Complex",
                F.lit("COMPLEX_UBO_STRUCTURE")
            ).when(
                F.col("ubo_complexity") == "Moderate",
                F.lit("MODERATE_UBO_COMPLEXITY")
            ),

            F.when(
                F.col("industry_risk") == "Cash Intensive",
                F.lit("CASH_INTENSIVE_INDUSTRY")
            ).when(
                F.col("industry_risk") == "Elevated",
                F.lit("ELEVATED_INDUSTRY_RISK")
            ),

            F.when(
                F.col("onboarding_channel") == "Remote",
                F.lit("REMOTE_ONBOARDING")
            ),

            F.when(
                F.col("kyc_documentation") == "Incomplete",
                F.lit("INCOMPLETE_KYC_DOCUMENTATION")
            )
        ),
        lambda reason: reason.isNotNull()
    )
)

# COMMAND ----------

# DBTITLE 1,Making Risk Decisions Explainable
# MAGIC %md
# MAGIC ## 🔍 Explainability Through Reason Codes
# MAGIC
# MAGIC This cell adds **transparency** to the risk scoring by creating a `reason_codes` array that captures **why** each customer received their risk rating.
# MAGIC
# MAGIC ### How It Works
# MAGIC
# MAGIC 1. **Creates an array** of conditional checks across all 6 risk categories
# MAGIC 2. **Assigns human-readable reason codes** when risk conditions are met
# MAGIC 3. **Filters out nulls** to keep only the actual risk factors present
# MAGIC
# MAGIC ### Reason Code Mapping
# MAGIC
# MAGIC | Risk Factor | Reason Code |
# MAGIC |-------------|-------------|
# MAGIC | High-risk country | `HIGH_RISK_COUNTRY` |
# MAGIC | Medium-risk country | `MEDIUM_RISK_COUNTRY` |
# MAGIC | Direct PEP | `DIRECT_PEP` |
# MAGIC | PEP associate | `PEP_ASSOCIATE` |
# MAGIC | Complex ownership | `COMPLEX_UBO_STRUCTURE` |
# MAGIC | Moderate UBO complexity | `MODERATE_UBO_COMPLEXITY` |
# MAGIC | Cash-intensive industry | `CASH_INTENSIVE_INDUSTRY` |
# MAGIC | Elevated industry risk | `ELEVATED_INDUSTRY_RISK` |
# MAGIC | Remote onboarding | `REMOTE_ONBOARDING` |
# MAGIC | Incomplete KYC docs | `INCOMPLETE_KYC_DOCUMENTATION` |
# MAGIC
# MAGIC ### Example Output
# MAGIC
# MAGIC **Customer A** (score: 58, rating: HIGH):
# MAGIC ```python
# MAGIC reason_codes = ["HIGH_RISK_COUNTRY", "DIRECT_PEP", "COMPLEX_UBO_STRUCTURE", "INCOMPLETE_KYC_DOCUMENTATION"]
# MAGIC ```
# MAGIC
# MAGIC **Customer B** (score: 8, rating: LOW):
# MAGIC ```python
# MAGIC reason_codes = ["MEDIUM_RISK_COUNTRY"]
# MAGIC ```
# MAGIC
# MAGIC ### Why This Matters
# MAGIC
# MAGIC Reason codes enable:
# MAGIC * **Regulatory compliance**: Auditors can trace every risk decision
# MAGIC * **Customer communication**: Clear explanations for enhanced due diligence requirements
# MAGIC * **Model validation**: Data scientists can analyze which factors drive risk ratings
# MAGIC * **Operational efficiency**: Compliance teams know exactly which issues to investigate

# COMMAND ----------

# DBTITLE 1,Inspect the risk-rating distribution
from pyspark.sql.window import Window

risk_distribution = (
    scored_customers
    .groupBy("kyc_risk_rating")
    .agg(
        F.count("*").alias("customers"),
        F.round(F.avg("kyc_risk_score"), 2).alias("average_score")
    )
    .withColumn(
        "percentage",
        F.round(
            F.col("customers")
            / F.sum("customers").over(Window.partitionBy()) * 100,
            2
        )
    )
    .orderBy(
        F.when(F.col("kyc_risk_rating") == "LOW", 1)
         .when(F.col("kyc_risk_rating") == "MEDIUM", 2)
         .otherwise(3)
    )
)

display(risk_distribution)

# COMMAND ----------

# DBTITLE 1,Compare ratings with future outcomes
rating_outcomes = (
    scored_customers
    .groupBy("kyc_risk_rating")
    .agg(
        F.count("*").alias("customers"),
        F.sum("future_aml_event").alias("future_positive_customers"),
        F.round(
            F.avg("future_aml_event") * 100,
            4
        ).alias("future_positive_rate")
    )
    .orderBy(
        F.when(F.col("kyc_risk_rating") == "LOW", 1)
         .when(F.col("kyc_risk_rating") == "MEDIUM", 2)
         .otherwise(3)
    )
)

display(rating_outcomes)

# COMMAND ----------

# DBTITLE 1,Validating the Scorecard Against Reality
# MAGIC %md
# MAGIC ## ⚠️ Retrospective Validation: Does the Scorecard Predict Reality?
# MAGIC
# MAGIC This cell performs **backtesting** — comparing our KYC risk ratings against **what actually happened** to validate if the traditional scorecard is predictive.
# MAGIC
# MAGIC ### How It Works
# MAGIC
# MAGIC 1. **Groups customers by their KYC risk rating** (LOW, MEDIUM, HIGH)
# MAGIC 2. **Counts how many in each group committed money laundering** in the future
# MAGIC 3. **Calculates the positive rate** — the percentage who became launderers
# MAGIC
# MAGIC ### Understanding "Future Outcome"
# MAGIC
# MAGIC The `future_aml_event` flag comes from notebook **03_temporal_target**:
# MAGIC * **1** = Customer was involved in money laundering transactions **after** the observation period
# MAGIC * **0** = Customer remained clean in the future
# MAGIC
# MAGIC This is **ground truth** — we scored them at time T₀ based on static KYC attributes, and now we're checking: *"Did our HIGH-risk customers actually become money launderers?"*
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚨 The Surprising Results
# MAGIC
# MAGIC | Risk Rating | Total Customers | Future Launderers | Positive Rate |
# MAGIC |-------------|----------------|-------------------|---------------|
# MAGIC | **LOW** | 99,386 | 1,171 | **1.18%** |
# MAGIC | **MEDIUM** | 58,676 | 653 | **1.11%** |
# MAGIC | **HIGH** | 8,021 | 85 | **1.06%** |
# MAGIC
# MAGIC ### ❌ The Scorecard Is NOT Predictive
# MAGIC
# MAGIC **Critical Finding**: HIGH-risk customers have the **lowest** future positive rate (1.06%), while LOW-risk customers have the **highest** (1.18%). The traditional KYC scorecard shows **no predictive power** — it fails to identify who will actually engage in money laundering.
# MAGIC
# MAGIC ### Why This Happens
# MAGIC
# MAGIC **Static KYC factors** (country risk, PEP status, ownership structure, industry) capture **regulatory risk**, not **behavioral risk**:
# MAGIC
# MAGIC * A Direct PEP from a high-risk country with complex ownership (HIGH-rated) may have legitimate reasons for their profile
# MAGIC * A simple business with complete documentation (LOW-rated) can still exhibit suspicious transaction patterns
# MAGIC
# MAGIC **Money laundering is about behavior, not demographics.**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ✅ Why This Validates Our ML Approach
# MAGIC
# MAGIC This finding justifies building a **machine learning model** that incorporates:
# MAGIC
# MAGIC 1. **Behavioral features**: Transaction patterns, velocity, counterparty networks, geographic spread
# MAGIC 2. **Temporal features**: Changes over time, sudden spikes, deviation from historical norms
# MAGIC 3. **Relational features**: Who they transact with, network centrality, exposure to known bad actors
# MAGIC
# MAGIC The ML model will learn which **combinations of static + dynamic features** actually predict money laundering — not just assign points based on compliance checkboxes.
# MAGIC
# MAGIC **Next Step**: Build feature engineering pipelines that capture behavioral signals from transaction data.

# COMMAND ----------

# DBTITLE 1,Inspect example high-risk customers
display(
    scored_customers
    .filter(F.col("kyc_risk_rating") == "HIGH")
    .select(
        "entity_id",
        "entity_name",
        "customer_type",
        "kyc_risk_score",
        "kyc_risk_rating",
        "reason_codes",
        "future_aml_event"
    )
    .orderBy(F.desc("kyc_risk_score"))
    .limit(20)
)

# COMMAND ----------

scored_customers.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("main.default.kyc_customer_scorecard")