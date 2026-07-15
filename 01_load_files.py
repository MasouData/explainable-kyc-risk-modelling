# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
from pyspark.sql import functions as F

BASE_PATH = "/Volumes/main/default/aml_dataset"

TRANS_PATH = f"{BASE_PATH}/HI-Small_Trans.csv"
ACCOUNTS_PATH = f"{BASE_PATH}/HI-Small_accounts.csv"
PATTERNS_PATH = f"{BASE_PATH}/HI-Small_Patterns.txt"

files = dbutils.fs.ls(BASE_PATH)

display(
    spark.createDataFrame(
        [(f.name, f.size, f.path) for f in files],
        ["file_name", "size_bytes", "path"]
    ).orderBy("file_name")
)

# COMMAND ----------

transactions_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(TRANS_PATH)
)

accounts_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(ACCOUNTS_PATH)
)

patterns_df = spark.read.text(PATTERNS_PATH)

print(f"Transactions: {transactions_df.count():,}")
print(f"Accounts:     {accounts_df.count():,}")
print(f"Pattern rows: {patterns_df.count():,}")

# COMMAND ----------

print("TRANSACTION COLUMNS")
print(transactions_df.columns)
transactions_df.printSchema()
display(transactions_df.limit(10))

print("ACCOUNT COLUMNS")
print(accounts_df.columns)
accounts_df.printSchema()
display(accounts_df.limit(10))

print("PATTERN EXAMPLES")
display(patterns_df.limit(20))

# COMMAND ----------

# DBTITLE 1,Cell 4
from pyspark.sql import Window

label_col = next(
    (
        c for c in transactions_df.columns
        if c.lower().replace(" ", "_") == "is_laundering"
    ),
    None
)

if label_col is None:
    raise ValueError(
        f"Laundering label not found. Available columns: "
        f"{transactions_df.columns}"
    )

label_summary = (
    transactions_df
    .groupBy(label_col)
    .count()
    .withColumn(
        "percentage",
        F.round(
            F.col("count") /
            F.sum("count").over(Window.partitionBy()) * 100,
            4
        )
    )
    .orderBy(label_col)
)

display(label_summary)

# COMMAND ----------

timestamp_col = next(
    (
        c for c in transactions_df.columns
        if c.lower() in {"timestamp", "date", "datetime"}
    ),
    None
)

if timestamp_col:
    transactions_df.select(
        F.min(timestamp_col).alias("minimum_timestamp"),
        F.max(timestamp_col).alias("maximum_timestamp")
    ).show(truncate=False)

null_summary = transactions_df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in transactions_df.columns
])

display(null_summary)

# COMMAND ----------

audit_summary = {
    "dataset": "HI-Small",
    "transaction_rows": transactions_df.count(),
    "account_rows": accounts_df.count(),
    "transaction_columns": len(transactions_df.columns),
    "account_columns": len(accounts_df.columns),
    "label_column": label_col,
    "timestamp_column": timestamp_col
}

audit_summary

# COMMAND ----------

# DBTITLE 1,02_customer_mapping
#Normalise the accounts table
from pyspark.sql import functions as F

def normalise_bank_id(column_name):
    """
    Convert values such as 021174 and 21174 to the same canonical value.
    """
    return (
        F.when(
            F.col(column_name).cast("long").isNotNull(),
            F.col(column_name).cast("long").cast("string")
        )
        .otherwise(F.trim(F.col(column_name).cast("string")))
    )


def normalise_account(column_name):
    return F.upper(F.trim(F.col(column_name).cast("string")))


accounts_clean = (
    accounts_df
    .select(
        F.trim(F.col("Bank Name")).alias("bank_name"),
        normalise_bank_id("Bank ID").alias("bank_id"),
        normalise_account("Account Number").alias("account_number"),
        F.trim(F.col("Entity ID")).alias("entity_id"),
        F.trim(F.col("Entity Name")).alias("entity_name")
    )
)

display(accounts_clean.limit(10))

# COMMAND ----------

# DBTITLE 1,Validate the account-to-customer mapping
#Each bank account should belong to only one entity.
ambiguous_accounts = (
    accounts_clean
    .groupBy("bank_id", "account_number")
    .agg(
        F.countDistinct("entity_id").alias("entity_count"),
        F.count("*").alias("row_count")
    )
    .filter(F.col("entity_count") > 1)
)

ambiguous_count = ambiguous_accounts.count()

print(f"Accounts linked to multiple entities: {ambiguous_count:,}")

if ambiguous_count > 0:
    display(ambiguous_accounts.limit(20))
    raise ValueError(
        "Some bank accounts are connected to multiple Entity IDs."
    )

accounts_clean = accounts_clean.dropDuplicates(
    ["bank_id", "account_number"]
)

# COMMAND ----------

# DBTITLE 1,Clean the transactions
transactions_clean = (
    transactions_df
    .select(
        F.to_timestamp(
            F.col("Timestamp"),
            "yyyy/MM/dd HH:mm"
        ).alias("transaction_timestamp"),

        normalise_bank_id("From Bank").alias("from_bank_id"),
        normalise_account("Account2").alias("from_account"),

        normalise_bank_id("To Bank").alias("to_bank_id"),
        normalise_account("Account4").alias("to_account"),

        F.col("Amount Paid").cast("double").alias("amount_paid"),
        F.col("Amount Received").cast("double").alias("amount_received"),

        F.trim(F.col("Payment Currency")).alias("payment_currency"),
        F.trim(F.col("Receiving Currency")).alias("receiving_currency"),
        F.trim(F.col("Payment Format")).alias("payment_format"),

        F.col("Is Laundering").cast("int").alias("is_laundering")
    )
    .withColumn(
        "transaction_date",
        F.to_date("transaction_timestamp")
    )
)

display(transactions_clean.limit(10))

# COMMAND ----------

# DBTITLE 1,Attach sender and receiver entities
# Same accounts table, different aliases - JOIN conditions assign sender/receiver roles 

sender_accounts = accounts_clean.select(
    F.col("bank_id").alias("sender_bank_id"),
    F.col("account_number").alias("sender_account"),
    F.col("entity_id").alias("sender_entity_id"),
    F.col("entity_name").alias("sender_entity_name")
)

receiver_accounts = accounts_clean.select(
    F.col("bank_id").alias("receiver_bank_id"),
    F.col("account_number").alias("receiver_account"),
    F.col("entity_id").alias("receiver_entity_id"),
    F.col("entity_name").alias("receiver_entity_name")
)

enriched_transactions = (
    transactions_clean
    .join(
        sender_accounts,
        (
            (F.col("from_bank_id") == F.col("sender_bank_id")) &
            (F.col("from_account") == F.col("sender_account"))
        ),
        "left"
    )
    .join(
        receiver_accounts,
        (
            (F.col("to_bank_id") == F.col("receiver_bank_id")) &
            (F.col("to_account") == F.col("receiver_account"))
        ),
        "left"
    )
    .drop(
        "sender_bank_id",
        "sender_account",
        "receiver_bank_id",
        "receiver_account"
    )
)

display(enriched_transactions.limit(10))

# COMMAND ----------

# DBTITLE 1,Check mapping coverage and date range
mapping_summary = (
    enriched_transactions
    .agg(
        F.count("*").alias("transactions"),

        F.sum(
            F.col("sender_entity_id").isNull().cast("long")
        ).alias("unmatched_senders"),

        F.sum(
            F.col("receiver_entity_id").isNull().cast("long")
        ).alias("unmatched_receivers"),

        F.countDistinct("sender_entity_id").alias(
            "distinct_sender_entities"
        ),

        F.countDistinct("receiver_entity_id").alias(
            "distinct_receiver_entities"
        ),

        F.min("transaction_timestamp").alias("minimum_timestamp"),
        F.max("transaction_timestamp").alias("maximum_timestamp")
    )
)

mapping_summary = (
    mapping_summary
    .withColumn(
        "sender_match_percentage",
        F.round(
            (
                1 -
                F.col("unmatched_senders") /
                F.col("transactions")
            ) * 100,
            4
        )
    )
    .withColumn(
        "receiver_match_percentage",
        F.round(
            (
                1 -
                F.col("unmatched_receivers") /
                F.col("transactions")
            ) * 100,
            4
        )
    )
)

display(mapping_summary)

# COMMAND ----------

# DBTITLE 1,customer_accounts_summary = (     accounts_clean     .groupBy("entity_id", "entity_name")     .agg(         F.count("*").alias("account_count"),         F.countDistinct("bank_id").alias("bank_count")     )     .orderBy(F.desc("account_count")) )  display(customer_accounts_summary.limit(20))
customer_accounts_summary = (
    accounts_clean
    .groupBy("entity_id", "entity_name")
    .agg(
        F.count("*").alias("account_count"),
        F.countDistinct("bank_id").alias("bank_count")
    )
    .orderBy(F.desc("account_count"))
)

display(customer_accounts_summary.limit(20))


# COMMAND ----------

# MAGIC %md
# MAGIC Corporation #34854: 
# MAGIC - 7,820 accounts = This entity has 7,820 different bank account numbers across various banks
# MAGIC - 1,185 banks = These accounts are spread across 1,185 different banks
# MAGIC - Average: ~6.6 accounts per bank.
# MAGIC
# MAGIC This pattern is highly suspicious for several reasons:
# MAGIC - Structuring/Layering: Legitimate corporations rarely need thousands of accounts across over a thousand banks
# MAGIC - Classic Money Laundering Pattern: Spreading funds across many accounts and banks makes tracking difficult
# MAGIC - Red Flag Indicator: This is exactly the kind of pattern AML models look for

# COMMAND ----------

accounts_clean.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("main.default.kyc_customer_accounts")

enriched_transactions.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("main.default.kyc_enriched_transactions")

# COMMAND ----------

# DBTITLE 1,Choose temporal modelling windows
from pyspark.sql import functions as F

transactions = spark.table(
    "main.default.kyc_enriched_transactions"
)

daily_summary = (
    transactions
    .groupBy("transaction_date")
    .agg(
        F.count("*").alias("transactions"),
        F.sum("is_laundering").alias("laundering_transactions"),
        F.countDistinct("sender_entity_id").alias("sender_entities"),
        F.countDistinct("receiver_entity_id").alias("receiver_entities")
    )
    .withColumn(
        "laundering_percentage",
        F.round(
            F.col("laundering_transactions")
            / F.col("transactions") * 100,
            4
        )
    )
    .orderBy("transaction_date")
)

display(daily_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC - Observation window: 1–7 September 2022
# MAGIC - Future outcome window: 8–10 September 2022
# MAGIC - Exclude: 11–18 September because transaction volume collapses and the laundering rate becomes unrealistically high.