# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Load customers and account information
from pyspark.sql import functions as F
from pyspark.sql import Window

customer_target = spark.table(
    "main.default.kyc_customer_target"
)

customer_accounts = spark.table(
    "main.default.kyc_customer_accounts"
)

customer_base = (
    customer_accounts
    .groupBy("entity_id")
    .agg(
        F.first("entity_name", ignorenulls=True).alias("entity_name"),
        F.countDistinct("account_number").alias("account_count"),
        F.countDistinct("bank_id").alias("bank_count")
    )
    .join(
        customer_target,
        on="entity_id",
        how="inner"
    )
)

print(f"Customers: {customer_base.count():,}")
display(customer_base.limit(10))

# COMMAND ----------

# DBTITLE 1,Extract entity type
# Drop duplicate entity_name column from the previous join
customer_base = customer_base.drop(customer_target["entity_name"])

customer_base = (
    customer_base
    .withColumn(
        "customer_type",
        F.when(
            F.col("entity_name").contains("Corporation"),
            "Corporation"
        )
        .when(
            F.col("entity_name").contains("Partnership"),
            "Partnership"
        )
        .when(
            F.col("entity_name").contains("Sole Proprietorship"),
            "Sole Proprietorship"
        )
        .otherwise("Other")
    )
)

display(
    customer_base
    .groupBy("customer_type")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# DBTITLE 1,Generate reproducible synthetic KYC attributes
#hashes of entity_id, so the same customer always receives the same profile.
def deterministic_bucket(seed_number, number_of_buckets=100):
    return F.pmod(
        F.xxhash64("entity_id", F.lit(seed_number)),
        F.lit(number_of_buckets)
    )


kyc_profiles = (
    customer_base

    # Country-risk distribution:
    # 75% low, 20% medium, 5% high
    .withColumn(
        "country_bucket",
        deterministic_bucket(101)
    )
    .withColumn(
        "country_risk",
        F.when(F.col("country_bucket") < 75, "Low")
        .when(F.col("country_bucket") < 95, "Medium")
        .otherwise("High")
    )

    # PEP distribution:
    # 97% none, 2% associate/family, 1% direct
    .withColumn(
        "pep_bucket",
        deterministic_bucket(202)
    )
    .withColumn(
        "pep_status",
        F.when(F.col("pep_bucket") < 97, "None")
        .when(F.col("pep_bucket") < 99, "Associate")
        .otherwise("Direct")
    )

    # Ownership complexity
    .withColumn(
        "ubo_bucket",
        deterministic_bucket(303)
    )
    .withColumn(
        "ubo_complexity",
        F.when(F.col("ubo_bucket") < 70, "Simple")
        .when(F.col("ubo_bucket") < 92, "Moderate")
        .otherwise("Complex")
    )

    # Industry risk
    .withColumn(
        "industry_bucket",
        deterministic_bucket(404)
    )
    .withColumn(
        "industry_risk",
        F.when(F.col("industry_bucket") < 65, "Standard")
        .when(F.col("industry_bucket") < 90, "Elevated")
        .otherwise("Cash Intensive")
    )

    # Onboarding channel
    .withColumn(
        "channel_bucket",
        deterministic_bucket(505)
    )
    .withColumn(
        "onboarding_channel",
        F.when(F.col("channel_bucket") < 70, "Branch")
        .otherwise("Remote")
    )

    # KYC-document completeness
    .withColumn(
        "document_bucket",
        deterministic_bucket(606)
    )
    .withColumn(
        "kyc_documentation",
        F.when(F.col("document_bucket") < 92, "Complete")
        .otherwise("Incomplete")
    )

    .drop(
        "country_bucket",
        "pep_bucket",
        "ubo_bucket",
        "industry_bucket",
        "channel_bucket",
        "document_bucket"
    )
)

# COMMAND ----------

# DBTITLE 1,Inspect distributions
for column_name in [
    "customer_type",
    "country_risk",
    "pep_status",
    "ubo_complexity",
    "industry_risk",
    "onboarding_channel",
    "kyc_documentation"
]:
    print(f"\nDistribution: {column_name}")

    display(
        kyc_profiles
        .groupBy(column_name)
        .count()
        .withColumn(
            "percentage",
            F.round(
                F.col("count") /
                F.sum("count").over(Window.partitionBy()) * 100,
                2
            )
        )
        .orderBy(F.desc("count"))
    )
    

# COMMAND ----------

kyc_profiles.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("main.default.kyc_customer_profiles")