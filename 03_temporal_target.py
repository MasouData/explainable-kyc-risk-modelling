# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC - Observation window → build customer features
# MAGIC - Future window      → create the target label

# COMMAND ----------

from pyspark.sql import functions as F

transactions = spark.table(
    "main.default.kyc_enriched_transactions"
)

OBS_START = "2022-09-01"
OBS_END = "2022-09-07"

OUTCOME_START = "2022-09-08"
OUTCOME_END = "2022-09-10"

observation_tx = transactions.filter(
    F.col("transaction_date").between(OBS_START, OBS_END)
)

outcome_tx = transactions.filter(
    F.col("transaction_date").between(OUTCOME_START, OUTCOME_END)
)

print(f"Observation transactions: {observation_tx.count():,}")
print(f"Outcome transactions:     {outcome_tx.count():,}")

# COMMAND ----------

# DBTITLE 1,Define customers active during observation
observation_customers = (
    observation_tx
    .select(
        F.col("sender_entity_id").alias("entity_id"),
        F.col("sender_entity_name").alias("entity_name")
    )
    .unionByName(
        observation_tx.select(
            F.col("receiver_entity_id").alias("entity_id"),
            F.col("receiver_entity_name").alias("entity_name")
        )
    )
    .filter(F.col("entity_id").isNotNull())
    .dropDuplicates(["entity_id"])
)

print(
    f"Customers in modelling population: "
    f"{observation_customers.count():,}"
)

# COMMAND ----------

# DBTITLE 1,Find customers involved in future laundering transactions
future_positive_customers = (
    outcome_tx
    .filter(F.col("is_laundering") == 1)
    .select(
        F.col("sender_entity_id").alias("entity_id")
    )
    .unionByName(
        outcome_tx
        .filter(F.col("is_laundering") == 1)
        .select(
            F.col("receiver_entity_id").alias("entity_id")
        )
    )
    .filter(F.col("entity_id").isNotNull())
    .dropDuplicates(["entity_id"])
    .withColumn("future_aml_event", F.lit(1))
)

# COMMAND ----------

# DBTITLE 1,Create the modelling target
from pyspark.sql.window import Window

customer_target = (
    observation_customers
    .join(
        future_positive_customers,
        on="entity_id",
        how="left"
    )
    .fillna({"future_aml_event": 0})
)

target_summary = (
    customer_target
    .groupBy("future_aml_event")
    .count()
    .withColumn(
        "percentage",
        F.round(
            F.col("count") /
            F.sum("count").over(Window.partitionBy()) * 100,
            4
        )
    )
    .orderBy("future_aml_event")
)

display(target_summary)

# COMMAND ----------

observation_tx.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("main.default.kyc_observation_transactions")

customer_target.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("main.default.kyc_customer_target")