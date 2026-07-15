# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
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