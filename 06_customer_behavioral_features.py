# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
from pyspark.sql import functions as F

transactions = (
    spark.table("main.default.kyc_observation_transactions")
    .withColumn(
        "transaction_id",
        F.sha2(
            F.concat_ws(
                "||",
                F.col("transaction_timestamp").cast("string"),
                F.col("from_bank_id"),
                F.col("from_account"),
                F.col("to_bank_id"),
                F.col("to_account"),
                F.col("amount_paid").cast("string"),
                F.col("amount_received").cast("string"),
                F.col("payment_currency"),
                F.col("receiving_currency"),
                F.col("payment_format")
            ),
            256
        )
    )
)

outgoing_events = transactions.select(
    "transaction_id",
    "transaction_date",
    F.col("sender_entity_id").alias("entity_id"),
    F.col("receiver_entity_id").alias("counterparty_entity_id"),
    F.col("from_bank_id").alias("entity_bank_id"),
    F.col("to_bank_id").alias("counterparty_bank_id"),
    F.lit("OUT").alias("direction"),
    F.col("amount_paid").alias("amount"),
    F.col("payment_currency").alias("currency"),
    "payment_format"
)

incoming_events = transactions.select(
    "transaction_id",
    "transaction_date",
    F.col("receiver_entity_id").alias("entity_id"),
    F.col("sender_entity_id").alias("counterparty_entity_id"),
    F.col("to_bank_id").alias("entity_bank_id"),
    F.col("from_bank_id").alias("counterparty_bank_id"),
    F.lit("IN").alias("direction"),
    F.col("amount_received").alias("amount"),
    F.col("receiving_currency").alias("currency"),
    "payment_format"
)

events = (
    outgoing_events
    .unionByName(incoming_events)
    .filter(F.col("entity_id").isNotNull())
)

# COMMAND ----------

# DBTITLE 1,From Transactions to Events
# MAGIC %md
# MAGIC ## 🔄 Transforming Transaction Data Into Entity-Centric Events
# MAGIC
# MAGIC This cell restructures transaction data from a **transaction-centric** view into an **entity-centric** view — the foundation for customer behavioral features.
# MAGIC
# MAGIC ### The Transformation
# MAGIC
# MAGIC **Before**: Each row = 1 transaction with sender and receiver
# MAGIC ```
# MAGIC [Transaction] → Sender A pays Receiver B $100
# MAGIC ```
# MAGIC
# MAGIC **After**: Each transaction becomes 2 events — one per participant
# MAGIC ```
# MAGIC [Event 1] → Entity A: OUT event, $100 paid to Counterparty B
# MAGIC [Event 2] → Entity B: IN event, $100 received from Counterparty A
# MAGIC ```
# MAGIC
# MAGIC ### Key Changes
# MAGIC
# MAGIC | Field | Outgoing Event (OUT) | Incoming Event (IN) |
# MAGIC |-------|---------------------|---------------------|
# MAGIC | **entity_id** | sender_entity_id | receiver_entity_id |
# MAGIC | **counterparty_entity_id** | receiver_entity_id | sender_entity_id |
# MAGIC | **direction** | "OUT" | "IN" |
# MAGIC | **amount** | amount_paid | amount_received |
# MAGIC | **currency** | payment_currency | receiving_currency |
# MAGIC
# MAGIC ### Why This Matters
# MAGIC
# MAGIC This entity-centric structure enables customer-level analysis:
# MAGIC * **Volume features**: How much does each customer send vs. receive?
# MAGIC * **Network features**: Who are their counterparties? How many unique connections?
# MAGIC * **Behavioral patterns**: Do they receive small amounts and send large amounts (layering)?
# MAGIC * **Time-based features**: Transaction frequency, velocity, and timing patterns
# MAGIC
# MAGIC Every customer behavioral feature we build in this notebook starts from this `events` table.

# COMMAND ----------

# DBTITLE 1,Standardise amounts within each currency
events = events.withColumn(
    "log_amount",
    F.log(F.coalesce(F.col("amount"), F.lit(0.0)) + 1.0)
)

currency_statistics = (
    events
    .groupBy("currency")
    .agg(
        F.avg("log_amount").alias("currency_mean"),
        F.stddev_samp("log_amount").alias("currency_std")
    )
)

events = (
    events
    .join(currency_statistics, on="currency", how="left")
    .withColumn(
        "amount_zscore",
        F.when(
            F.col("currency_std").isNull() |
            (F.col("currency_std") == 0),
            F.lit(0.0)
        ).otherwise(
            (
                F.col("log_amount") -
                F.col("currency_mean")
            ) / F.col("currency_std")
        )
    )
    .withColumn(
        "is_cross_bank",
        (F.col("entity_bank_id") != F.col("counterparty_bank_id")).cast("int")
    )
    .withColumn(
        "is_self_transfer",
        (
            F.col("entity_id") ==
            F.col("counterparty_entity_id")
        ).cast("int")
    )
)

# COMMAND ----------

# DBTITLE 1,Making Amounts Comparable Across Currencies
# MAGIC %md
# MAGIC ## 📊 Standardizing Transaction Amounts for Cross-Currency Comparison
# MAGIC
# MAGIC This cell solves a critical problem: **How do you compare a $1,000 transaction to a €500 transaction?** Raw amounts can't be compared across currencies, so we normalize them using z-scores.
# MAGIC
# MAGIC ### The Normalization Process
# MAGIC
# MAGIC **Step 1: Log transformation**
# MAGIC ```python
# MAGIC log_amount = log(amount + 1)
# MAGIC ```
# MAGIC Compresses the wide range of transaction values (some pennies, some millions) into a more manageable scale.
# MAGIC
# MAGIC **Step 2: Calculate currency statistics**
# MAGIC ```python
# MAGIC For each currency:
# MAGIC   - mean of log_amount
# MAGIC   - standard deviation of log_amount
# MAGIC ```
# MAGIC
# MAGIC **Step 3: Z-score standardization**
# MAGIC ```python
# MAGIC amount_zscore = (log_amount - currency_mean) / currency_std
# MAGIC ```
# MAGIC
# MAGIC ### What Z-Score Tells Us
# MAGIC
# MAGIC | Z-Score | Meaning | AML Significance |
# MAGIC |---------|---------|------------------|
# MAGIC | **0** | Average transaction for that currency | Normal activity |
# MAGIC | **+2.5** | 2.5 standard deviations above average | Unusually large - potential structuring or rapid movement |
# MAGIC | **-2.5** | 2.5 standard deviations below average | Unusually small - potential smurfing |
# MAGIC
# MAGIC Now a z-score of **+3.0** means "unusually large" whether it's USD, EUR, or JPY.
# MAGIC
# MAGIC ### Additional Risk Flags
# MAGIC
# MAGIC The cell also creates binary indicators for suspicious patterns:
# MAGIC
# MAGIC * **`is_cross_bank`**: Transaction crosses bank boundaries (higher layering risk)
# MAGIC * **`is_self_transfer`**: Customer sending money to themselves (potential structuring or obfuscation)
# MAGIC
# MAGIC These normalized amounts and flags become the foundation for detecting anomalous behavioral patterns in the next aggregation step.

# COMMAND ----------

# DBTITLE 1,Aggregate to one row per customer
customer_activity = (
    events
    .groupBy("entity_id")
    .agg(
        F.countDistinct("transaction_id").alias("transaction_count"),

        F.countDistinct(
            F.when(
                F.col("direction") == "OUT",
                F.col("transaction_id")
            )
        ).alias("outgoing_tx_count"),

        F.countDistinct(
            F.when(
                F.col("direction") == "IN",
                F.col("transaction_id")
            )
        ).alias("incoming_tx_count"),

        F.countDistinct("transaction_date").alias("active_days"),

        F.countDistinct(
            F.when(
                F.col("entity_id") != F.col("counterparty_entity_id"),
                F.col("counterparty_entity_id")
            )
        ).alias("counterparty_count"),

        F.countDistinct("counterparty_bank_id").alias(
            "counterparty_bank_count"
        ),

        F.countDistinct("currency").alias("currency_count"),

        F.countDistinct("payment_format").alias(
            "payment_format_count"
        ),

        F.avg("amount_zscore").alias("average_amount_zscore"),
        F.max("amount_zscore").alias("maximum_amount_zscore"),

        F.countDistinct(
            F.when(
                F.col("amount_zscore") >= 2.5,
                F.col("transaction_id")
            )
        ).alias("unusually_large_tx_count"),

        F.countDistinct(
            F.when(
                F.col("is_cross_bank") == 1,
                F.col("transaction_id")
            )
        ).alias("cross_bank_tx_count"),

        F.countDistinct(
            F.when(
                F.col("is_self_transfer") == 1,
                F.col("transaction_id")
            )
        ).alias("self_transfer_tx_count")
    )
)

# COMMAND ----------

# DBTITLE 1,Building the Behavioral Feature Set
# MAGIC %md
# MAGIC ## 🎯 Building Behavioral Features: From Events to Customer Profiles
# MAGIC
# MAGIC This cell performs the critical transformation: **collapsing thousands of transaction events into a single behavioral profile per customer**.
# MAGIC
# MAGIC ### The Aggregation
# MAGIC
# MAGIC **Before**: Multiple event rows per customer (one per transaction, split by direction)
# MAGIC ```
# MAGIC Customer A: [OUT $100], [IN $50], [OUT $200], [IN $75], ...
# MAGIC ```
# MAGIC
# MAGIC **After**: One summary row per customer with 13 behavioral features
# MAGIC ```
# MAGIC Customer A: {transaction_count: 245, outgoing_tx_count: 120, counterparty_count: 45, ...}
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Feature Categories
# MAGIC
# MAGIC #### 1. **Volume Features** (Activity Level)
# MAGIC * `transaction_count` - Total transactions in observation period
# MAGIC * `outgoing_tx_count` - Payments made
# MAGIC * `incoming_tx_count` - Payments received
# MAGIC * `active_days` - Number of days with at least one transaction
# MAGIC
# MAGIC **AML Signal**: High volume in short time = potential structuring or rapid movement
# MAGIC
# MAGIC #### 2. **Network Features** (Connectivity)
# MAGIC * `counterparty_count` - Number of unique trading partners (excludes self-transfers)
# MAGIC * `counterparty_bank_count` - Number of unique banks involved
# MAGIC
# MAGIC **AML Signal**: Sudden expansion of network = layering through multiple entities
# MAGIC
# MAGIC #### 3. **Diversity Features** (Complexity)
# MAGIC * `currency_count` - Number of different currencies used
# MAGIC * `payment_format_count` - Variety of payment methods
# MAGIC
# MAGIC **AML Signal**: High diversity = obfuscation attempts or cross-border complexity
# MAGIC
# MAGIC #### 4. **Anomaly Features** (Red Flags)
# MAGIC * `average_amount_zscore` - Typical transaction size (normalized)
# MAGIC * `maximum_amount_zscore` - Largest transaction size (normalized)
# MAGIC * `unusually_large_tx_count` - Transactions >2.5 standard deviations above currency mean
# MAGIC * `cross_bank_tx_count` - Transactions crossing bank boundaries
# MAGIC * `self_transfer_tx_count` - Payments to own accounts
# MAGIC
# MAGIC **AML Signal**: Outlier amounts, cross-bank movement, and self-transfers indicate layering and structuring
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Why These Features Matter
# MAGIC
# MAGIC **Static KYC** (notebook 05) captured **who the customer is** (PEP, country, industry).
# MAGIC
# MAGIC **Behavioral features** capture **what the customer does** (volume, velocity, network patterns, anomalies).
# MAGIC
# MAGIC **Money laundering is a behavior**, not a demographic. These 13 features will form the core predictors in the ML model.

# COMMAND ----------

# DBTITLE 1,Add ratios and join the scorecard
customer_features = (
    spark.table("main.default.kyc_customer_scorecard")
    .join(customer_activity, on="entity_id", how="left")
    .fillna({
        "transaction_count": 0,
        "outgoing_tx_count": 0,
        "incoming_tx_count": 0,
        "active_days": 0,
        "counterparty_count": 0,
        "counterparty_bank_count": 0,
        "currency_count": 0,
        "payment_format_count": 0,
        "average_amount_zscore": 0.0,
        "maximum_amount_zscore": 0.0,
        "unusually_large_tx_count": 0,
        "cross_bank_tx_count": 0,
        "self_transfer_tx_count": 0
    })
    .withColumn(
        "transactions_per_active_day",
        F.when(
            F.col("active_days") > 0,
            F.col("transaction_count") / F.col("active_days")
        ).otherwise(0.0)
    )
    .withColumn(
        "incoming_outgoing_ratio",
        (
            F.col("incoming_tx_count") + 1.0
        ) / (
            F.col("outgoing_tx_count") + 1.0
        )
    )
    .withColumn(
        "cross_bank_ratio",
        F.when(
            F.col("transaction_count") > 0,
            F.col("cross_bank_tx_count") /
            F.col("transaction_count")
        ).otherwise(0.0)
    )
    .withColumn(
        "self_transfer_ratio",
        F.when(
            F.col("transaction_count") > 0,
            F.col("self_transfer_tx_count") /
            F.col("transaction_count")
        ).otherwise(0.0)
    )
    .withColumn(
        "unusually_large_tx_ratio",
        F.when(
            F.col("transaction_count") > 0,
            F.col("unusually_large_tx_count") /
            F.col("transaction_count")
        ).otherwise(0.0)
    )
)

# COMMAND ----------

# DBTITLE 1,Combining Static and Behavioral Risk
# MAGIC %md
# MAGIC ## 🔗 Bringing It All Together: Static KYC + Behavioral Features + Risk Ratios
# MAGIC
# MAGIC This cell creates the **complete customer risk profile** by combining static KYC attributes with behavioral transaction patterns and calculating normalized risk ratios.
# MAGIC
# MAGIC ### The Three-Way Integration
# MAGIC
# MAGIC **1. Static KYC Scorecard** (from notebook 05)
# MAGIC ```
# MAGIC entity_id, country_risk, pep_status, kyc_risk_score, kyc_risk_rating, reason_codes...
# MAGIC ```
# MAGIC
# MAGIC **2. Behavioral Features** (from cell 5)
# MAGIC ```
# MAGIC transaction_count, counterparty_count, average_amount_zscore, cross_bank_tx_count...
# MAGIC ```
# MAGIC
# MAGIC **3. Risk Ratios** (calculated in this cell)
# MAGIC ```
# MAGIC transactions_per_active_day, incoming_outgoing_ratio, cross_bank_ratio...
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Why Add Ratios?
# MAGIC
# MAGIC **Raw counts** are misleading:
# MAGIC * Customer A: 100 cross-bank transactions out of 1,000 total (10%)
# MAGIC * Customer B: 10 cross-bank transactions out of 20 total (50%)
# MAGIC
# MAGIC Customer B has fewer transactions but **higher risk concentration** — ratios capture this.
# MAGIC
# MAGIC ### The 5 Risk Ratios
# MAGIC
# MAGIC #### 1. **Transaction Velocity**
# MAGIC ```python
# MAGIC transactions_per_active_day = transaction_count / active_days
# MAGIC ```
# MAGIC **Meaning**: How many transactions per day when active?  
# MAGIC **AML Signal**: >10 tx/day = rapid movement, potential structuring
# MAGIC
# MAGIC #### 2. **Flow Imbalance**
# MAGIC ```python
# MAGIC incoming_outgoing_ratio = (incoming_tx_count + 1) / (outgoing_tx_count + 1)
# MAGIC ```
# MAGIC **Meaning**: Are they receiving more than sending, or vice versa?  
# MAGIC **AML Signal**: 
# MAGIC * Ratio >> 1 (mostly incoming) = potential collection account for layering
# MAGIC * Ratio << 1 (mostly outgoing) = potential distribution account after integration
# MAGIC
# MAGIC #### 3. **Cross-Bank Concentration**
# MAGIC ```python
# MAGIC cross_bank_ratio = cross_bank_tx_count / transaction_count
# MAGIC ```
# MAGIC **Meaning**: What % of transactions cross bank boundaries?  
# MAGIC **AML Signal**: >50% = excessive layering through multiple institutions
# MAGIC
# MAGIC #### 4. **Self-Transfer Frequency**
# MAGIC ```python
# MAGIC self_transfer_ratio = self_transfer_tx_count / transaction_count
# MAGIC ```
# MAGIC **Meaning**: What % are transfers to own accounts?  
# MAGIC **AML Signal**: >20% = potential obfuscation or breaking audit trails
# MAGIC
# MAGIC #### 5. **Outlier Transaction Rate**
# MAGIC ```python
# MAGIC unusually_large_tx_ratio = unusually_large_tx_count / transaction_count
# MAGIC ```
# MAGIC **Meaning**: What % of transactions are unusually large (z-score > 2.5)?  
# MAGIC **AML Signal**: >10% = inconsistent behavior, potential rapid movement of large sums
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Handling Inactive Customers
# MAGIC
# MAGIC The `.fillna()` ensures customers with **zero transactions** during the observation period get zeros for all behavioral features (not nulls). These customers still have KYC risk scores but no behavioral signals.
# MAGIC
# MAGIC ### Final Feature Set
# MAGIC
# MAGIC **customer_features** now contains:
# MAGIC * **6 static KYC attributes** (country, PEP, industry, etc.)
# MAGIC * **13 behavioral counts** (volume, network, diversity, anomalies)
# MAGIC * **5 behavioral ratios** (velocity, flow, concentration)
# MAGIC * **Target variable**: `future_aml_event` (0/1)
# MAGIC
# MAGIC **Total: ~24 features ready for machine learning model training.**

# COMMAND ----------

# DBTITLE 1,Compare behavioural features by target
feature_comparison = (
    customer_features
    .groupBy("future_aml_event")
    .agg(
        F.count("*").alias("customers"),

        F.round(
            F.avg("transaction_count"), 2
        ).alias("avg_transaction_count"),

        F.round(
            F.avg("counterparty_count"), 2
        ).alias("avg_counterparty_count"),

        F.round(
            F.avg("transactions_per_active_day"), 2
        ).alias("avg_transactions_per_day"),

        F.round(
            F.avg("maximum_amount_zscore"), 3
        ).alias("avg_maximum_amount_zscore"),

        F.round(
            F.avg("cross_bank_ratio"), 4
        ).alias("avg_cross_bank_ratio"),

        F.round(
            F.avg("self_transfer_ratio"), 4
        ).alias("avg_self_transfer_ratio"),

        F.round(
            F.avg("account_count"), 2
        ).alias("avg_account_count"),

        F.round(
            F.avg("bank_count"), 2
        ).alias("avg_bank_count")
    )
    .orderBy("future_aml_event")
)

display(feature_comparison)

# COMMAND ----------

# DBTITLE 1,Do Behavioral Features Actually Predict?
# MAGIC %md
# MAGIC ## ✅ Validation: Behavioral Features ARE Predictive
# MAGIC
# MAGIC This cell answers the critical question: **Do behavioral patterns distinguish future money launderers from clean customers?**
# MAGIC
# MAGIC Remember from notebook 05: **static KYC failed** — HIGH-risk customers had the same (actually lower) laundering rate as LOW-risk customers.
# MAGIC
# MAGIC ### The Comparison
# MAGIC
# MAGIC We split customers by `future_aml_event` (0 = clean, 1 = became launderer) and compare their average behavioral patterns.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚨 The Results: Clear Behavioral Differences
# MAGIC
# MAGIC | Feature | Clean Customers | Future Launderers | Difference |
# MAGIC |---------|----------------|-------------------|------------|
# MAGIC | **Transaction count** | 35.59 | 528.42 | **15x higher** |
# MAGIC | **Counterparties** | 5.67 | 58.28 | **10x higher** |
# MAGIC | **Transactions/day** | 5.62 | 75.94 | **13x higher** |
# MAGIC | **Max amount z-score** | 1.76 | 2.13 | 21% higher |
# MAGIC | **Account count** | 2.56 | 51.65 | **20x higher** |
# MAGIC | **Bank count** | 2.48 | 18.51 | **7x higher** |
# MAGIC | **Self-transfer ratio** | 12.5% | 15.0% | 19% higher |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 💡 What This Tells Us
# MAGIC
# MAGIC Future money launderers exhibit **dramatically different behavior**:
# MAGIC
# MAGIC 1. **Rapid Movement**: ~15x more transactions, ~13x higher daily velocity
# MAGIC 2. **Network Expansion**: ~10x more counterparties, spread across ~7x more banks
# MAGIC 3. **Infrastructure**: ~20x more accounts (layering through multiple accounts)
# MAGIC 4. **Obfuscation**: Higher self-transfer rates (breaking audit trails)
# MAGIC
# MAGIC This is the **classic money laundering pattern**: high volume, rapid movement, complex networks, and layering across multiple accounts and institutions.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ✅ Behavioral Features Win
# MAGIC
# MAGIC **Static KYC** (country, PEP, industry) → **Not predictive** (notebook 05 showed no difference)
# MAGIC
# MAGIC **Behavioral features** (volume, velocity, network, accounts) → **Highly predictive** (10-20x differences)
# MAGIC
# MAGIC **Conclusion**: Machine learning models trained on these behavioral features should significantly outperform traditional rule-based KYC scorecards.

# COMMAND ----------

customer_features.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("main.default.kyc_customer_features")