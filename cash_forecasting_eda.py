# Databricks notebook source
# MAGIC %md
# MAGIC # Transaction Data EDA — Cash Liquidity Forecasting
# MAGIC
# MAGIC **Goal:** Explore ~1.5 years of AR/AP-style transaction data (posting date, due date,
# MAGIC clearing date, amount, customer number, clearing document, HBKID) to:
# MAGIC 1. Understand data quality, coverage and granularity
# MAGIC 2. Engineer features that describe *payment behavior* (delay, processing time, patterns)
# MAGIC 3. Build the actual **daily net cash flow series** — the target for forecasting
# MAGIC 4. Surface seasonality, customer concentration, and anomalies that will drive model design
# MAGIC
# MAGIC Structure: Load → Profile → Feature Engineer → EDA (univariate → time series → entity-level) →
# MAGIC Target construction → Summary of modeling-ready insights.
# MAGIC
# MAGIC Replace the table/query names in the **Config** cell with your actual source.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Config

# COMMAND ----------

# Adjust these to your environment
CATALOG = "your_catalog"
SCHEMA = "your_schema"
TABLE = "your_transactions_table"          # or use a SQL query instead, see below
SOURCE_SQL = None                           # e.g. "SELECT * FROM finance.ar_transactions WHERE posting_date >= '2024-01-01'"

# Column name mapping — edit to match your actual SAP-style source column names
COL_POSTING_DATE = "posting_date"
COL_DUE_DATE = "due_date"
COL_CLEARING_DATE = "clearing_date"
COL_AMOUNT = "amount"
COL_CUSTOMER = "customer_number"
COL_CLEARING_DOC = "clearing_document"
COL_HBKID = "hbkid"                         # House Bank ID (SAP) — which bank account the cash flows through
COL_BUSINESS_AREA = "Business_Area"         # SAP Business Area — organizational/segment split, e.g. 0131, 0019
COL_POSTING_KEY = "Posting_Key"             # SAP Posting Key — transaction nature, e.g. 01, 05, 07
COL_GL_DOC_TYPE = "GL_Document_type"        # SAP Document Type — document/process origin, e.g. F2, AB, DR, ZM, S2, ZP

# Reference mappings — standard SAP meanings for customer (D) reconciliation account posting keys.
# VERIFY against your own FI/SAP team's config before trusting labels for reporting — Z-prefixed
# and some non-standard document types below are company-configured, not SAP-standard.
POSTING_KEY_MAP = {
    "01": "Invoice (debit)",           # new receivable created
    "05": "Outgoing payment (debit)",  # payment-related debit posting
    "07": "Other clearing (debit)",    # clearing/adjustment, not a new invoice or a payment
}
GL_DOC_TYPE_MAP = {
    "F2": "Customer invoice (SD billing)",
    "AB": "Accounting document (general/manual journal entry)",
    "DR": "Customer invoice (FI-posted)",
    "ZM": "Custom/company-specific type — confirm with SAP team (likely manual posting)",
    "S2": "Custom/company-specific type — confirm with SAP team",
    "ZP": "Custom/company-specific type — confirm with SAP team (likely payment-related)",
}

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

# COMMAND ----------

# Central report collector — every EDA section below writes its key findings into this dict.
# Section 11 (end of notebook) turns it into a single formatted report. Keep this as the one
# place that accumulates "final numbers," rather than re-deriving them again at the end.
report = {
    "data_overview": {},
    "data_quality": {},
    "amount": {},
    "payment_delay": {},
    "processing_time": {},
    "posting_key_doc_type": {},
    "target_series": {},
    "customer": {},
    "hbkid": {},
    "business_area": {},
    "outliers": {},
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Data

# COMMAND ----------

if SOURCE_SQL:
    df = spark.sql(SOURCE_SQL)
else:
    df = spark.table(f"{CATALOG}.{SCHEMA}.{TABLE}")

row_count = df.count()
print(f"Row count: {row_count:,}")
print(f"Column count: {len(df.columns)}")
df.printSchema()

report["data_overview"]["n_rows"] = row_count
report["data_overview"]["n_columns"] = len(df.columns)
report["data_overview"]["columns"] = df.columns

# COMMAND ----------

display(df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Quality Profile
# MAGIC
# MAGIC Before any modeling, we need to know: how complete is each column, what's the real date
# MAGIC range, are there duplicate documents, and — critically for cash forecasting — **what
# MAGIC fraction of items are still open (no clearing date yet)**, since only cleared items
# MAGIC represent actual realized cash movement.

# COMMAND ----------

total_rows = df.count()

null_summary = df.select([
    F.round(F.sum(F.col(c).isNull().cast("int")) / total_rows * 100, 2).alias(c)
    for c in df.columns
])
null_pct_pd = null_summary.toPandas().T.rename(columns={0: "pct_null"})
null_pct_pd.sort_values("pct_null", ascending=False)

report["data_quality"]["null_pct_by_column"] = null_pct_pd["pct_null"].to_dict()

# COMMAND ----------

# Date range coverage
date_range_row = df.select(
    F.min(COL_POSTING_DATE).alias("min_posting_date"),
    F.max(COL_POSTING_DATE).alias("max_posting_date"),
    F.min(COL_DUE_DATE).alias("min_due_date"),
    F.max(COL_DUE_DATE).alias("max_due_date"),
    F.min(COL_CLEARING_DATE).alias("min_clearing_date"),
    F.max(COL_CLEARING_DATE).alias("max_clearing_date"),
).collect()[0].asDict()

for k, v in date_range_row.items():
    print(f"{k}: {v}")

report["data_overview"]["date_ranges"] = date_range_row

# COMMAND ----------

# Open vs cleared items — only cleared items have a real, realized cash flow date
open_vs_cleared = (
    df.withColumn("status", F.when(F.col(COL_CLEARING_DATE).isNull(), "open").otherwise("cleared"))
      .groupBy("status")
      .agg(F.count("*").alias("n_items"), F.sum(COL_AMOUNT).alias("total_amount"))
)
display(open_vs_cleared)

report["data_quality"]["open_vs_cleared"] = {
    row["status"]: {"n_items": row["n_items"], "total_amount": row["total_amount"]}
    for row in open_vs_cleared.collect()
}

# COMMAND ----------

# Duplicate check on clearing_document (should usually be unique per cleared cash event,
# but one clearing document can clear multiple line items — check cardinality)
dup_check = (
    df.filter(F.col(COL_CLEARING_DOC).isNotNull())
      .groupBy(COL_CLEARING_DOC)
      .count()
      .orderBy(F.desc("count"))
)
display(dup_check.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC **What to look for here:**
# MAGIC - High null % on `clearing_date` is *expected* (open items) — but track this ratio over
# MAGIC   time; a sudden jump could mean a data pull issue, not a real business shift.
# MAGIC - If `due_date` or `posting_date` has nulls, flag those rows for exclusion from
# MAGIC   date-driven features.
# MAGIC - A `clearing_document` tied to many rows usually means partial/multi-invoice clearing —
# MAGIC   useful to know before aggregating by document.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature Engineering
# MAGIC
# MAGIC Two families of features:
# MAGIC - **Row-level payment behavior features** (delay, processing time, calendar attributes)
# MAGIC - **Entity-level aggregated features** (per customer, per HBKID) — built *after* EDA
# MAGIC   confirms which aggregations matter, in Section 6/7.

# COMMAND ----------

fe = df.withColumn(COL_POSTING_DATE, F.to_date(COL_POSTING_DATE)) \
       .withColumn(COL_DUE_DATE, F.to_date(COL_DUE_DATE)) \
       .withColumn(COL_CLEARING_DATE, F.to_date(COL_CLEARING_DATE))

# --- Status flag ---
fe = fe.withColumn("is_cleared", F.col(COL_CLEARING_DATE).isNotNull())

# --- Payment behavior features (only meaningful for cleared items) ---
fe = fe.withColumn(
    "payment_delay_days",
    F.when(F.col("is_cleared"), F.datediff(F.col(COL_CLEARING_DATE), F.col(COL_DUE_DATE)))
)  # positive = paid late, negative = paid early, 0 = on time

fe = fe.withColumn(
    "processing_time_days",
    F.when(F.col("is_cleared"), F.datediff(F.col(COL_CLEARING_DATE), F.col(COL_POSTING_DATE)))
)  # posting -> clearing lag

fe = fe.withColumn(
    "payment_term_days",
    F.datediff(F.col(COL_DUE_DATE), F.col(COL_POSTING_DATE))
)  # agreed credit term

fe = fe.withColumn(
    "is_overdue",
    F.when(F.col("payment_delay_days") > 0, True).otherwise(False)
)

fe = fe.withColumn(
    "delay_bucket",
    F.when(F.col("payment_delay_days").isNull(), "open")
     .when(F.col("payment_delay_days") <= 0, "on_time_or_early")
     .when(F.col("payment_delay_days") <= 7, "1-7d_late")
     .when(F.col("payment_delay_days") <= 30, "8-30d_late")
     .when(F.col("payment_delay_days") <= 90, "31-90d_late")
     .otherwise("90d+_late")
)

# --- Amount features ---
fe = fe.withColumn("amount_abs", F.abs(F.col(COL_AMOUNT)))
fe = fe.withColumn("flow_direction", F.when(F.col(COL_AMOUNT) >= 0, "inflow").otherwise("outflow"))

fe = fe.withColumn(
    "amount_bucket",
    F.when(F.col("amount_abs") < 1000, "small(<1K)")
     .when(F.col("amount_abs") < 10000, "medium(1K-10K)")
     .when(F.col("amount_abs") < 100000, "large(10K-100K)")
     .otherwise("very_large(100K+)")
)

# --- Calendar features (built on posting_date and clearing_date separately — both matter) ---
for prefix, date_col in [("posting", COL_POSTING_DATE), ("clearing", COL_CLEARING_DATE)]:
    fe = fe.withColumn(f"{prefix}_dow", F.dayofweek(date_col))                  # 1=Sunday..7=Saturday
    fe = fe.withColumn(f"{prefix}_day", F.dayofmonth(date_col))
    fe = fe.withColumn(f"{prefix}_month", F.month(date_col))
    fe = fe.withColumn(f"{prefix}_year", F.year(date_col))
    fe = fe.withColumn(f"{prefix}_week", F.weekofyear(date_col))
    fe = fe.withColumn(f"{prefix}_quarter", F.quarter(date_col))
    fe = fe.withColumn(
        f"{prefix}_is_weekend", F.col(f"{prefix}_dow").isin([1, 7])
    )
    fe = fe.withColumn(
        f"{prefix}_is_month_end",
        F.col(date_col) == F.last_day(date_col)
    )
    fe = fe.withColumn(
        f"{prefix}_days_to_month_end",
        F.datediff(F.last_day(date_col), date_col)
    )
    fe = fe.withColumn(
        f"{prefix}_is_quarter_end",
        F.col(f"{prefix}_month").isin([3, 6, 9, 12]) & F.col(f"{prefix}_is_month_end")
    )

# --- Posting Key / GL Document Type — map codes to human-readable labels for EDA/reporting ---
posting_key_map_expr = F.create_map([F.lit(x) for pair in POSTING_KEY_MAP.items() for x in pair])
gl_doc_type_map_expr = F.create_map([F.lit(x) for pair in GL_DOC_TYPE_MAP.items() for x in pair])

fe = fe.withColumn("posting_key_desc", posting_key_map_expr[F.col(COL_POSTING_KEY)])
fe = fe.withColumn("gl_doc_type_desc", gl_doc_type_map_expr[F.col(COL_GL_DOC_TYPE)])

# Coarse transaction category, driven primarily by posting key — this is the key signal for
# telling "a new invoice was booked" apart from "a payment/clearing event actually moved cash,"
# both of which can otherwise look identical once you only look at clearing_date (Section 5 uses
# this to sharpen the target series — see 5.0 below).
fe = fe.withColumn(
    "txn_category",
    F.when(F.col(COL_POSTING_KEY) == "01", "invoice")
     .when(F.col(COL_POSTING_KEY) == "05", "payment")
     .when(F.col(COL_POSTING_KEY) == "07", "other_clearing")
     .otherwise("unmapped")
)

fe.cache()
print("Feature-engineered row count:", fe.count())
display(fe.select(
    COL_CUSTOMER, COL_HBKID, COL_BUSINESS_AREA, COL_POSTING_KEY, COL_GL_DOC_TYPE,
    "txn_category", COL_POSTING_DATE, COL_DUE_DATE, COL_CLEARING_DATE,
    COL_AMOUNT, "payment_delay_days", "processing_time_days", "delay_bucket",
    "flow_direction", "amount_bucket"
).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Univariate EDA

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Amount distribution
# MAGIC Transaction amounts are almost always heavily right-skewed with a few very large
# MAGIC transactions — check both raw and log scale.

# COMMAND ----------

amount_pd = fe.select("amount_abs", "flow_direction").sample(fraction=0.2, seed=42).toPandas()
# sample for plotting if the table is large; adjust/remove sampling for smaller datasets

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(amount_pd["amount_abs"], bins=80, ax=axes[0])
axes[0].set_title("Transaction Amount — raw scale")
axes[0].set_xlabel("Amount")

sns.histplot(np.log1p(amount_pd["amount_abs"]), bins=80, ax=axes[1])
axes[1].set_title("Transaction Amount — log1p scale")
axes[1].set_xlabel("log(1 + Amount)")
plt.tight_layout()
plt.show()

# COMMAND ----------

amount_summary_pd = (
    fe.select("amount_abs")
      .summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "90%", "99%", "max")
      .toPandas()
)
print(amount_summary_pd)

report["amount"]["summary_stats"] = dict(zip(amount_summary_pd["summary"], amount_summary_pd["amount_abs"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Payment delay distribution
# MAGIC This is one of the most important behavioral signals for cash forecasting — it tells you
# MAGIC *when* promised cash (due_date) actually shows up (clearing_date).

# COMMAND ----------

delay_pd = fe.filter(F.col("is_cleared")).select("payment_delay_days", "delay_bucket").toPandas()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(delay_pd["payment_delay_days"].clip(-30, 90), bins=60, ax=axes[0])
axes[0].axvline(0, color="red", linestyle="--", label="on time")
axes[0].set_title("Payment Delay (days) — clipped to [-30, 90]")
axes[0].legend()

bucket_order = ["on_time_or_early", "1-7d_late", "8-30d_late", "31-90d_late", "90d+_late"]
sns.countplot(
    data=delay_pd, y="delay_bucket", order=bucket_order, ax=axes[1]
)
axes[1].set_title("Delay Buckets")
plt.tight_layout()
plt.show()

# COMMAND ----------

pct_on_time = (delay_pd["payment_delay_days"] <= 0).mean() * 100
median_delay = delay_pd["payment_delay_days"].median()
mean_delay = delay_pd["payment_delay_days"].mean()

print(f"% paid on time or early : {pct_on_time:.1f}%")
print(f"Median delay (days)     : {median_delay:.1f}")
print(f"Mean delay (days)       : {mean_delay:.1f}")

report["payment_delay"]["pct_on_time_or_early"] = round(pct_on_time, 1)
report["payment_delay"]["median_delay_days"] = round(median_delay, 1)
report["payment_delay"]["mean_delay_days"] = round(mean_delay, 1)
report["payment_delay"]["bucket_counts"] = delay_pd["delay_bucket"].value_counts().to_dict()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Processing time (posting → clearing)

# COMMAND ----------

proc_pd = fe.filter(F.col("is_cleared")).select("processing_time_days").toPandas()
sns.histplot(proc_pd["processing_time_days"].clip(0, 120), bins=60)
plt.title("Processing Time: Posting Date → Clearing Date (days, clipped to 120)")
plt.show()

report["processing_time"]["mean_days"] = round(proc_pd["processing_time_days"].mean(), 1)
report["processing_time"]["median_days"] = round(proc_pd["processing_time_days"].median(), 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4b. Posting Key & GL Document Type EDA
# MAGIC These two columns describe *what kind of event* each row actually represents — not just
# MAGIC when it happened. That matters a lot: Section 5's target series groups everything by
# MAGIC `clearing_date`, which silently mixes genuine payment events together with invoice/clearing
# MAGIC postings that merely *happen* to clear on the same date. This section checks whether that
# MAGIC mixing is a real problem here.

# COMMAND ----------

# Frequency and value by posting key
pk_summary = (
    fe.groupBy(COL_POSTING_KEY, "posting_key_desc")
      .agg(F.count("*").alias("n_transactions"), F.sum("amount_abs").alias("total_amount"))
      .orderBy(F.desc("total_amount"))
)
display(pk_summary)

# Frequency and value by GL document type
doc_summary = (
    fe.groupBy(COL_GL_DOC_TYPE, "gl_doc_type_desc")
      .agg(F.count("*").alias("n_transactions"), F.sum("amount_abs").alias("total_amount"))
      .orderBy(F.desc("total_amount"))
)
display(doc_summary)

report["posting_key_doc_type"]["posting_key_summary"] = pk_summary.toPandas().to_dict(orient="records")
report["posting_key_doc_type"]["doc_type_summary"] = doc_summary.toPandas().to_dict(orient="records")

n_unmapped = fe.filter(F.col("txn_category") == "unmapped").count()
report["posting_key_doc_type"]["pct_unmapped_txn_category"] = round(n_unmapped / fe.count() * 100, 2)

# COMMAND ----------

# Cross-tab: which document types actually use which posting keys? A clean, sparse cross-tab
# means the two columns are consistent/validating each other; a dense cross-tab means document
# type alone won't tell you the transaction nature — you need posting key too.
crosstab_pd = fe.crosstab(COL_POSTING_KEY, COL_GL_DOC_TYPE).toPandas().set_index(f"{COL_POSTING_KEY}_{COL_GL_DOC_TYPE}")
plt.figure(figsize=(9, 5))
sns.heatmap(crosstab_pd, annot=True, fmt="d", cmap="Blues")
plt.title("Posting Key × GL Document Type — Transaction Counts")
plt.xlabel("GL Document Type")
plt.ylabel("Posting Key")
plt.show()

# COMMAND ----------

# Does txn_category composition shift over time? A sudden change in the invoice/payment/clearing
# mix usually signals a process change (new SAP config, new business line) worth knowing about
# before it gets baked into the model as if it were organic seasonality.
txn_monthly = (
    fe.withColumn("month", F.date_trunc("month", F.col(COL_POSTING_DATE)))
      .groupBy("month", "txn_category")
      .agg(F.count("*").alias("n"))
      .toPandas()
)
txn_monthly["month"] = pd.to_datetime(txn_monthly["month"])
txn_pivot = txn_monthly.pivot(index="month", columns="txn_category", values="n").fillna(0)
txn_pivot_pct = txn_pivot.div(txn_pivot.sum(axis=1), axis=0) * 100

txn_pivot_pct.plot.area(figsize=(14, 5), stacked=True)
plt.title("Monthly Transaction Mix by Category (% of row count)")
plt.ylabel("% of transactions")
plt.show()

# COMMAND ----------

# How does amount, delay, and flow_direction differ across txn_category? Payment-flagged rows
# should behave differently from invoice rows if the posting key mapping is meaningful here.
display(
    fe.groupBy("txn_category")
      .agg(
          F.count("*").alias("n_transactions"),
          F.round(F.avg("amount_abs"), 2).alias("avg_amount"),
          F.round(F.avg(F.when(F.col("is_cleared"), F.col("payment_delay_days"))), 2).alias("avg_delay_days"),
          F.round(F.avg(F.when(F.col("flow_direction") == "inflow", 1).otherwise(0)) * 100, 1).alias("pct_inflow"),
      )
)

# COMMAND ----------

# MAGIC %md
# MAGIC **What to look for:**
# MAGIC - If `txn_category = "payment"` rows are concentrated on different dates / have different
# MAGIC   amount profiles than `"invoice"` rows, that confirms posting key is a genuinely useful
# MAGIC   signal for sharpening the cash-flow target — see 5.0 below.
# MAGIC - A dense Posting Key × Doc Type cross-tab, or an `"unmapped"` txn_category share that isn't
# MAGIC   ~0%, means the posting-key-only mapping above is too coarse — extend `POSTING_KEY_MAP`/
# MAGIC   `GL_DOC_TYPE_MAP` with your FI team rather than trusting the default guesses.
# MAGIC - A visible shift in the monthly category mix marks a process/config change — note the
# MAGIC   date, since it may explain a level shift in the target series rather than real seasonality.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Building the Target Series: Daily Net Cash Flow
# MAGIC
# MAGIC This is the core deliverable of this EDA — the actual time series the forecasting model
# MAGIC will predict. **Realized cash movement happens on the clearing date, not the posting or
# MAGIC due date** — open items haven't moved cash yet, so they're excluded here (but see 5.3 for
# MAGIC how open items still matter, as a leading indicator of *future* cash flow).
# MAGIC
# MAGIC **5.0 — all-cleared-items vs payment-only target:** Section 4b checked whether
# MAGIC `txn_category` (derived from Posting Key) meaningfully separates payment events from
# MAGIC invoice/clearing postings. Two variants are built below — compare them before picking one:
# MAGIC - `daily_cash_pd`: every cleared row, regardless of category (what Section 5 onward uses by
# MAGIC   default — broadest definition of "cash activity").
# MAGIC - `daily_cash_payment_only_pd`: restricted to `txn_category == "payment"` — closer to actual
# MAGIC   bank cash movement if that mapping held up in 4b. If the two series look meaningfully
# MAGIC   different (not just scaled-down), the payment-only version is very likely the more
# MAGIC   accurate forecasting target — swap it in below.

# COMMAND ----------

daily_cash = (
    fe.filter(F.col("is_cleared"))
      .groupBy(COL_CLEARING_DATE)
      .agg(
          F.sum(COL_AMOUNT).alias("net_cash_flow"),
          F.sum(F.when(F.col("flow_direction") == "inflow", F.col("amount_abs")).otherwise(0)).alias("total_inflow"),
          F.sum(F.when(F.col("flow_direction") == "outflow", F.col("amount_abs")).otherwise(0)).alias("total_outflow"),
          F.count("*").alias("n_transactions"),
          F.countDistinct(COL_CUSTOMER).alias("n_unique_customers"),
      )
      .withColumnRenamed(COL_CLEARING_DATE, "date")
      .orderBy("date")
)

daily_cash_pd = daily_cash.toPandas()
daily_cash_pd["date"] = pd.to_datetime(daily_cash_pd["date"])

# Reindex to a full continuous daily calendar — missing days = zero cash movement
# (business-critical: don't let gaps silently disappear, a forecasting model needs to see them)
full_range = pd.date_range(daily_cash_pd["date"].min(), daily_cash_pd["date"].max(), freq="D")
daily_cash_pd = (
    daily_cash_pd.set_index("date")
                 .reindex(full_range)
                 .fillna({"net_cash_flow": 0, "total_inflow": 0, "total_outflow": 0,
                          "n_transactions": 0, "n_unique_customers": 0})
                 .rename_axis("date")
                 .reset_index()
)

daily_cash_pd.head()

# COMMAND ----------

# Payment-only variant — same construction, restricted to txn_category == "payment"
daily_cash_payment_only = (
    fe.filter(F.col("is_cleared") & (F.col("txn_category") == "payment"))
      .groupBy(COL_CLEARING_DATE)
      .agg(F.sum(COL_AMOUNT).alias("net_cash_flow"))
      .withColumnRenamed(COL_CLEARING_DATE, "date")
      .orderBy("date")
)
daily_cash_payment_only_pd = daily_cash_payment_only.toPandas()
daily_cash_payment_only_pd["date"] = pd.to_datetime(daily_cash_payment_only_pd["date"])
daily_cash_payment_only_pd = (
    daily_cash_payment_only_pd.set_index("date")
                               .reindex(full_range)
                               .fillna({"net_cash_flow": 0})
                               .rename_axis("date")
                               .reset_index()
)

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(daily_cash_pd["date"], daily_cash_pd["net_cash_flow"], linewidth=0.8, label="all cleared items", alpha=0.7)
ax.plot(daily_cash_payment_only_pd["date"], daily_cash_payment_only_pd["net_cash_flow"],
        linewidth=0.8, label="payment-only (txn_category='payment')", alpha=0.7)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Daily Net Cash Flow — All Cleared Items vs Payment-Only")
ax.set_ylabel("Net Cash Flow")
ax.legend()
plt.show()

corr_variants = np.corrcoef(daily_cash_pd["net_cash_flow"], daily_cash_payment_only_pd["net_cash_flow"])[0, 1]
print(f"Correlation between the two target variants: {corr_variants:.3f}")
print("High correlation (~0.9+) -> the two are roughly interchangeable, posting key adds little here.")
print("Lower correlation -> txn_category genuinely changes what the series represents; pick deliberately.")

report["target_series"]["date_range"] = (str(daily_cash_pd["date"].min().date()), str(daily_cash_pd["date"].max().date()))
report["target_series"]["n_days"] = len(daily_cash_pd)
report["target_series"]["all_vs_payment_only_correlation"] = round(float(corr_variants), 3)
report["target_series"]["total_net_cash_flow"] = round(float(daily_cash_pd["net_cash_flow"].sum()), 2)
report["target_series"]["daily_mean"] = round(float(daily_cash_pd["net_cash_flow"].mean()), 2)
report["target_series"]["daily_std"] = round(float(daily_cash_pd["net_cash_flow"].std()), 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Weekly / Monthly aggregation — smoother view of trend

# COMMAND ----------

weekly = daily_cash_pd.set_index("date").resample("W")["net_cash_flow"].sum()
monthly = daily_cash_pd.set_index("date").resample("MS")["net_cash_flow"].sum()

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)
weekly.plot(ax=axes[0], title="Weekly Net Cash Flow")
monthly.plot(ax=axes[1], kind="bar", title="Monthly Net Cash Flow")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Seasonality check — day of week & day of month
# MAGIC Cash flow data almost always has strong calendar effects: weekday vs weekend,
# MAGIC and month-end spikes (payroll, supplier settlement runs).

# COMMAND ----------

daily_cash_pd["dow_name"] = daily_cash_pd["date"].dt.day_name()
daily_cash_pd["day_of_month"] = daily_cash_pd["date"].dt.day

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
sns.boxplot(data=daily_cash_pd, x="dow_name", y="net_cash_flow", order=dow_order, ax=axes[0])
axes[0].set_title("Net Cash Flow by Day of Week")
axes[0].tick_params(axis="x", rotation=45)

day_of_month_avg = daily_cash_pd.groupby("day_of_month")["net_cash_flow"].mean()
axes[1].bar(day_of_month_avg.index, day_of_month_avg.values)
axes[1].set_title("Avg Net Cash Flow by Day of Month (month-end effect check)")
axes[1].set_xlabel("Day of Month")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 STL Decomposition + Stationarity Check
# MAGIC Separates trend / seasonality / residual, and runs the ADF test — exactly the diagnostics
# MAGIC needed to decide differencing order (d) and seasonal period for ARIMA/SARIMA, and to sanity
# MAGIC check that seasonality assumptions (e.g., weekly=7, monthly≈30) hold in your actual data.

# COMMAND ----------

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

ts = daily_cash_pd.set_index("date")["net_cash_flow"]

stl = STL(ts, period=7, robust=True)   # weekly seasonality; try period=30 too and compare
res = stl.fit()

fig = res.plot()
fig.set_size_inches(14, 8)
plt.show()

adf_result = adfuller(ts.dropna())
print(f"ADF Statistic : {adf_result[0]:.4f}")
print(f"p-value       : {adf_result[1]:.4f}")
print("--> Stationary" if adf_result[1] < 0.05 else "--> Non-stationary (consider differencing)")

report["target_series"]["adf_statistic"] = round(adf_result[0], 4)
report["target_series"]["adf_pvalue"] = round(adf_result[1], 4)
report["target_series"]["is_stationary"] = bool(adf_result[1] < 0.05)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 ACF / PACF — how far back does the series "remember"?

# COMMAND ----------

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(ts, lags=60, ax=axes[0])
plot_pacf(ts, lags=60, ax=axes[1])
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Customer-Level EDA
# MAGIC Cash flow is driven by a handful of customers far more than others — this concentration
# MAGIC matters for both feature engineering (customer embeddings / segments) and risk
# MAGIC (concentration risk on a few large payers).

# COMMAND ----------

customer_agg = (
    fe.groupBy(COL_CUSTOMER)
      .agg(
          F.count("*").alias("n_transactions"),
          F.sum("amount_abs").alias("total_amount"),
          F.avg(F.when(F.col("is_cleared"), F.col("payment_delay_days"))).alias("avg_delay_days"),
          F.stddev(F.when(F.col("is_cleared"), F.col("payment_delay_days"))).alias("std_delay_days"),
          F.avg(F.col("is_overdue").cast("int")).alias("pct_overdue"),
          F.min(COL_POSTING_DATE).alias("first_txn"),
          F.max(COL_POSTING_DATE).alias("last_txn"),
      )
      .orderBy(F.desc("total_amount"))
)
customer_pd = customer_agg.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Pareto / concentration check (classic 80/20)

# COMMAND ----------

customer_pd_sorted = customer_pd.sort_values("total_amount", ascending=False).reset_index(drop=True)
customer_pd_sorted["cum_pct_amount"] = customer_pd_sorted["total_amount"].cumsum() / customer_pd_sorted["total_amount"].sum() * 100
customer_pd_sorted["cum_pct_customers"] = (customer_pd_sorted.index + 1) / len(customer_pd_sorted) * 100

n_customers_80pct = (customer_pd_sorted["cum_pct_amount"] <= 80).sum()
print(f"Total unique customers: {len(customer_pd_sorted):,}")
print(f"Customers driving 80% of total transaction value: {n_customers_80pct:,} "
      f"({n_customers_80pct/len(customer_pd_sorted)*100:.1f}% of customer base)")

report["customer"]["n_unique_customers"] = int(len(customer_pd_sorted))
report["customer"]["n_customers_driving_80pct_value"] = int(n_customers_80pct)
report["customer"]["pct_of_base_driving_80pct_value"] = round(n_customers_80pct / len(customer_pd_sorted) * 100, 1)

plt.figure(figsize=(10, 5))
plt.plot(customer_pd_sorted["cum_pct_customers"], customer_pd_sorted["cum_pct_amount"])
plt.axhline(80, color="red", linestyle="--")
plt.xlabel("% of Customers (ranked by value)")
plt.ylabel("Cumulative % of Total Amount")
plt.title("Customer Concentration (Pareto Curve)")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Payment behavior segments
# MAGIC Simple behavioral segmentation: fast/reliable payers vs chronically late payers. This
# MAGIC becomes either a categorical feature (segment) or numeric features (avg_delay, pct_overdue)
# MAGIC directly in the forecasting model.

# COMMAND ----------

def behavior_segment(row):
    if pd.isna(row["avg_delay_days"]):
        return "no_cleared_history"
    if row["avg_delay_days"] <= 0 and row["pct_overdue"] < 0.1:
        return "reliable_on_time"
    if row["avg_delay_days"] <= 15:
        return "mildly_late"
    if row["avg_delay_days"] <= 45:
        return "moderately_late"
    return "chronically_late"

customer_pd["behavior_segment"] = customer_pd.apply(behavior_segment, axis=1)

seg_summary = (
    customer_pd.groupby("behavior_segment")
    .agg(n_customers=("customer_number", "count") if COL_CUSTOMER == "customer_number"
         else (COL_CUSTOMER, "count"),
         total_amount=("total_amount", "sum"))
    .sort_values("total_amount", ascending=False)
)
seg_summary["pct_of_total_amount"] = seg_summary["total_amount"] / seg_summary["total_amount"].sum() * 100
seg_summary

report["customer"]["behavior_segments"] = seg_summary["pct_of_total_amount"].round(1).to_dict()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. HBKID (House Bank) Level EDA
# MAGIC Different bank accounts can have very different flow patterns (currency, region, purpose) —
# MAGIC important if the forecast needs to be produced **per bank account**, not just globally.

# COMMAND ----------

hbkid_daily = (
    fe.filter(F.col("is_cleared"))
      .groupBy(COL_HBKID, COL_CLEARING_DATE)
      .agg(F.sum(COL_AMOUNT).alias("net_cash_flow"))
      .withColumnRenamed(COL_CLEARING_DATE, "date")
)

hbkid_summary = (
    fe.groupBy(COL_HBKID)
      .agg(
          F.count("*").alias("n_transactions"),
          F.sum("amount_abs").alias("total_amount"),
          F.countDistinct(COL_CUSTOMER).alias("n_unique_customers"),
      )
      .orderBy(F.desc("total_amount"))
)
display(hbkid_summary)

hbkid_summary_pd = hbkid_summary.toPandas()
report["hbkid"]["n_unique_hbkids"] = int(hbkid_summary_pd.shape[0])
report["hbkid"]["top_5_by_amount"] = hbkid_summary_pd.head(5).to_dict(orient="records")

# COMMAND ----------

hbkid_daily_pd = hbkid_daily.toPandas()
hbkid_daily_pd["date"] = pd.to_datetime(hbkid_daily_pd["date"])

top_hbkids = hbkid_daily_pd.groupby(COL_HBKID)["net_cash_flow"].sum().abs().sort_values(ascending=False).head(5).index

plt.figure(figsize=(16, 6))
for h in top_hbkids:
    subset = hbkid_daily_pd[hbkid_daily_pd[COL_HBKID] == h].sort_values("date")
    plt.plot(subset["date"], subset["net_cash_flow"].rolling(7).mean(), label=str(h))
plt.legend(title="HBKID")
plt.title("7-Day Rolling Net Cash Flow — Top 5 HBKIDs by Volume")
plt.axhline(0, color="black", linewidth=0.8)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Decision point:** if the HBKID-level series look structurally different (different
# MAGIC seasonality, different scale, different volatility), build **separate models per HBKID**
# MAGIC (or a global model with HBKID as a categorical/embedding feature) rather than one pooled
# MAGIC model across all bank accounts.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7b. Business_Area EDA (0131 vs 0019)
# MAGIC Only two values here, so the main question is simpler than for HBKID/customer: are these
# MAGIC two genuinely different cash-flow regimes (different scale, seasonality, volatility,
# MAGIC customer base), or just a bookkeeping split of the same underlying flow? That answer
# MAGIC decides whether Business_Area becomes a **model split** (two separate forecasts) or just
# MAGIC a **categorical feature** in one pooled model.

# COMMAND ----------

ba_overview = (
    fe.groupBy(COL_BUSINESS_AREA)
      .agg(
          F.count("*").alias("n_transactions"),
          F.sum("amount_abs").alias("total_amount"),
          F.countDistinct(COL_CUSTOMER).alias("n_unique_customers"),
          F.countDistinct(COL_HBKID).alias("n_unique_hbkids"),
          F.avg(F.when(F.col("is_cleared"), F.col("payment_delay_days"))).alias("avg_delay_days"),
          F.min(COL_POSTING_DATE).alias("first_txn"),
          F.max(COL_POSTING_DATE).alias("last_txn"),
      )
)
display(ba_overview)

report["business_area"]["overview"] = ba_overview.toPandas().to_dict(orient="records")

# COMMAND ----------

# Check overlap: does each HBKID/customer sit under one Business_Area only, or do they mix?
# If customers/HBKIDs span both areas, a strict "split the data in two" approach loses information
# that a shared model with Business_Area as a feature would retain.
ba_customer_overlap = (
    fe.groupBy(COL_CUSTOMER)
      .agg(F.countDistinct(COL_BUSINESS_AREA).alias("n_business_areas"))
      .groupBy("n_business_areas")
      .count()
)
display(ba_customer_overlap)

report["business_area"]["customer_overlap"] = {
    row["n_business_areas"]: row["count"] for row in ba_customer_overlap.collect()
}

# COMMAND ----------

# Daily net cash flow, split by Business_Area — the real comparison that matters for forecasting
ba_daily = (
    fe.filter(F.col("is_cleared"))
      .groupBy(COL_BUSINESS_AREA, COL_CLEARING_DATE)
      .agg(F.sum(COL_AMOUNT).alias("net_cash_flow"))
      .withColumnRenamed(COL_CLEARING_DATE, "date")
      .orderBy("date")
)
ba_daily_pd = ba_daily.toPandas()
ba_daily_pd["date"] = pd.to_datetime(ba_daily_pd["date"])

fig, ax = plt.subplots(figsize=(16, 5))
for ba_value, grp in ba_daily_pd.groupby(COL_BUSINESS_AREA):
    grp = grp.sort_values("date")
    ax.plot(grp["date"], grp["net_cash_flow"].rolling(7).mean(), label=str(ba_value))
ax.axhline(0, color="black", linewidth=0.8)
ax.legend(title="Business_Area")
ax.set_title("7-Day Rolling Net Cash Flow — 0131 vs 0019")
plt.show()

# COMMAND ----------

# Distribution comparison — scale and volatility, side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=ba_daily_pd, x=COL_BUSINESS_AREA, y="net_cash_flow", ax=axes[0])
axes[0].set_title("Daily Net Cash Flow Distribution by Business_Area")

for ba_value, grp in ba_daily_pd.groupby(COL_BUSINESS_AREA):
    sns.kdeplot(grp["net_cash_flow"], label=str(ba_value), ax=axes[1])
axes[1].set_title("Density Comparison")
axes[1].legend(title="Business_Area")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Stationarity + dominant seasonality, per Business_Area — confirms whether they need
# different (d, seasonal period) settings if modeled separately
ba_stationarity = {}
for ba_value in [row[COL_BUSINESS_AREA] for row in ba_overview.select(COL_BUSINESS_AREA).collect()]:
    sub = (
        ba_daily_pd[ba_daily_pd[COL_BUSINESS_AREA] == ba_value]
        .set_index("date")["net_cash_flow"]
        .asfreq("D", fill_value=0)
    )
    adf_res = adfuller(sub.dropna())
    print(f"Business_Area {ba_value}: ADF p-value = {adf_res[1]:.4f} "
          f"({'stationary' if adf_res[1] < 0.05 else 'non-stationary'})")
    ba_stationarity[str(ba_value)] = {"adf_pvalue": round(adf_res[1], 4), "is_stationary": bool(adf_res[1] < 0.05)}

report["business_area"]["stationarity_by_area"] = ba_stationarity

# COMMAND ----------

# MAGIC %md
# MAGIC **Decision point:** if 0131 and 0019 show clearly different scale, seasonality or
# MAGIC stationarity behavior above, treat `Business_Area` the same way as HBKID — either two
# MAGIC separate forecasting pipelines, or one global model with `Business_Area` as a categorical
# MAGIC feature (and interaction terms with calendar features, if the *seasonality shape* itself
# MAGIC differs between the two, not just the scale).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Outlier / Anomaly Detection
# MAGIC Large one-off transactions (M&A-related transfers, major settlements) will distort both
# MAGIC EDA statistics and model training if not flagged.

# COMMAND ----------

# Row-level amount outliers via IQR
q1, q3 = fe.approxQuantile("amount_abs", [0.25, 0.75], 0.01)
iqr = q3 - q1
upper_fence = q3 + 3 * iqr   # 3x IQR = conservative "extreme outlier" fence

row_outliers = fe.filter(F.col("amount_abs") > upper_fence)
n_row_outliers = row_outliers.count()
n_total = fe.count()
print(f"Extreme-value rows (>3xIQR): {n_row_outliers:,} out of {n_total:,} ({n_row_outliers/n_total*100:.2f}%)")

report["outliers"]["n_extreme_rows"] = n_row_outliers
report["outliers"]["pct_extreme_rows"] = round(n_row_outliers / n_total * 100, 2)
report["outliers"]["amount_upper_fence"] = round(upper_fence, 2)

display(
    row_outliers.select(COL_CUSTOMER, COL_HBKID, COL_POSTING_DATE, COL_CLEARING_DATE, COL_AMOUNT)
    .orderBy(F.desc("amount"))
    .limit(20)
)

# COMMAND ----------

# Day-level anomalies on the target series itself — days where net cash flow is
# far outside its local rolling behavior (z-score on residual after removing 7-day rolling mean)
daily_cash_pd["roll_mean_7"] = daily_cash_pd["net_cash_flow"].rolling(7, center=True, min_periods=3).mean()
daily_cash_pd["roll_std_30"] = daily_cash_pd["net_cash_flow"].rolling(30, min_periods=10).std()
daily_cash_pd["residual"] = daily_cash_pd["net_cash_flow"] - daily_cash_pd["roll_mean_7"]
daily_cash_pd["z_score"] = daily_cash_pd["residual"] / daily_cash_pd["roll_std_30"]

day_anomalies = daily_cash_pd[daily_cash_pd["z_score"].abs() > 3].sort_values("date")
print(f"Anomalous days (|z| > 3): {len(day_anomalies)}")
day_anomalies[["date", "net_cash_flow", "z_score"]]

report["outliers"]["n_anomalous_days"] = int(len(day_anomalies))
report["outliers"]["anomalous_days"] = [
    {"date": str(r["date"].date()), "net_cash_flow": round(r["net_cash_flow"], 2), "z_score": round(r["z_score"], 2)}
    for _, r in day_anomalies.iterrows()
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Correlation Among Engineered Features
# MAGIC Quick check before modeling — helps catch redundant features (e.g., payment_term_days
# MAGIC and processing_time_days may be highly correlated) and confirms which engineered signals
# MAGIC actually vary meaningfully with amount/delay.

# COMMAND ----------

numeric_cols = ["amount_abs", "payment_delay_days", "processing_time_days", "payment_term_days"]
corr_pd = fe.select(numeric_cols).dropna().sample(fraction=0.3, seed=42).toPandas()

plt.figure(figsize=(7, 6))
sns.heatmap(corr_pd.corr(), annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Correlation — Row-Level Numeric Features")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Summary of Modeling-Ready Insights
# MAGIC
# MAGIC Fill this in from the actual outputs above before moving to modeling — this becomes the
# MAGIC design brief for the forecasting pipeline:
# MAGIC
# MAGIC - **Target series:** daily (or weekly, if daily too sparse) net cash flow, built from
# MAGIC   `clearing_date` only — confirmed continuous after reindexing missing days to zero.
# MAGIC - **Stationarity:** ADF result above → note whether differencing (d) is needed.
# MAGIC - **Seasonality:** confirm dominant period(s) from ACF/STL — weekly (7), monthly (~30),
# MAGIC   or both (may need Fourier terms / SARIMA with multiple seasonal components).
# MAGIC - **Calendar effects:** note which day-of-week / day-of-month patterns were significant —
# MAGIC   these become mandatory calendar features in the model (Part 5 of the theory guide).
# MAGIC - **Customer concentration:** note the 80/20 split — if cash flow is dominated by a small
# MAGIC   customer set, consider modeling their payment behavior explicitly (customer-level
# MAGIC   features or even per-customer models for the largest accounts).
# MAGIC - **Behavior segments:** reliable vs chronically-late payer mix — feed `avg_delay_days`,
# MAGIC   `pct_overdue`, `behavior_segment` into the model as customer-level features.
# MAGIC - **HBKID structure:** decide pooled-global vs per-bank-account modeling based on how
# MAGIC   different the HBKID-level series looked in Section 7.
# MAGIC - **Business_Area (0131 vs 0019):** note from Section 7b whether the two areas are
# MAGIC   structurally different series (scale/seasonality/stationarity) — decide split-model vs
# MAGIC   shared-model-with-feature accordingly, and note the customer/HBKID overlap result (does
# MAGIC   either entity span both areas?).
# MAGIC - **Posting Key / GL Document Type:** record the correlation between the all-cleared and
# MAGIC   payment-only target variants (Section 5.0) — if low, the payment-only series is likely
# MAGIC   the more accurate forecasting target, and `txn_category`/`posting_key_desc` become
# MAGIC   mandatory filters in the production pipeline, not just EDA artifacts. Also record any
# MAGIC   unmapped codes or process-mix shifts found in Section 4b, and confirm the `ZM`/`S2`/`ZP`
# MAGIC   document type guesses with the FI/SAP team before relying on their labels.
# MAGIC - **Open items (uncleared):** not part of the historical target, but their `due_date` and
# MAGIC   `amount` are a **leading indicator** of near-future cash flow — worth engineering as a
# MAGIC   forward-looking feature (e.g., "total open AR due in next 7/14/30 days") separate from
# MAGIC   the lag/rolling features built on historical cleared cash flow.
# MAGIC - **Anomalies:** list flagged outlier transactions/days here with business context once
# MAGIC   confirmed with finance (capex events, M&A, FX shocks) — decide whether to cap, exclude,
# MAGIC   or flag with a binary feature in training.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Generate EDA Report
# MAGIC Every section above wrote its key numbers into the `report` dict as it ran. This cell turns
# MAGIC that dict into one formatted Markdown report — the artifact to hand to stakeholders or keep
# MAGIC alongside the notebook, without re-deriving anything.

# COMMAND ----------

from datetime import datetime

def build_markdown_report(r: dict) -> str:
    do = r["data_overview"]
    dq = r["data_quality"]
    amt = r["amount"]
    delay = r["payment_delay"]
    proc = r["processing_time"]
    pkdt = r["posting_key_doc_type"]
    ts = r["target_series"]
    cust = r["customer"]
    hbk = r["hbkid"]
    ba = r["business_area"]
    out = r["outliers"]

    lines = []
    lines.append(f"# Cash Liquidity Transaction Data — EDA Report")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")

    lines.append("## 1. Data Overview")
    lines.append(f"- Rows: **{do.get('n_rows', 'n/a'):,}** | Columns: **{do.get('n_columns', 'n/a')}**")
    if "date_ranges" in do:
        for k, v in do["date_ranges"].items():
            lines.append(f"- {k.replace('_', ' ')}: **{v}**")
    if "open_vs_cleared" in dq:
        for status, vals in dq["open_vs_cleared"].items():
            lines.append(f"- {status.capitalize()} items: **{vals['n_items']:,}** "
                          f"(total amount: {vals['total_amount']:,.0f})")

    lines.append("\n## 2. Amount & Payment Behavior")
    if "summary_stats" in amt:
        s = amt["summary_stats"]
        lines.append(f"- Mean amount: {float(s.get('mean', 0)):,.0f} | "
                      f"Median: {float(s.get('50%', 0)):,.0f} | "
                      f"99th pct: {float(s.get('99%', 0)):,.0f} | "
                      f"Max: {float(s.get('max', 0)):,.0f}")
    lines.append(f"- % paid on time or early: **{delay.get('pct_on_time_or_early', 'n/a')}%**")
    lines.append(f"- Median delay: **{delay.get('median_delay_days', 'n/a')} days** "
                  f"(mean: {delay.get('mean_delay_days', 'n/a')} days)")
    lines.append(f"- Median processing time (posting→clearing): **{proc.get('median_days', 'n/a')} days**")

    lines.append("\n## 3. Posting Key / GL Document Type")
    lines.append(f"- Unmapped txn_category share: **{pkdt.get('pct_unmapped_txn_category', 'n/a')}%**")
    if "posting_key_summary" in pkdt:
        lines.append("- Posting key breakdown (by total amount):")
        for row in pkdt["posting_key_summary"]:
            lines.append(f"  - `{row[COL_POSTING_KEY]}` ({row['posting_key_desc']}): "
                          f"{row['n_transactions']:,} txns, {row['total_amount']:,.0f} total")

    lines.append("\n## 4. Target Series (Daily Net Cash Flow)")
    lines.append(f"- Date range: **{ts.get('date_range', ('n/a', 'n/a'))[0]} → {ts.get('date_range', ('n/a','n/a'))[1]}** "
                  f"({ts.get('n_days', 'n/a')} days)")
    lines.append(f"- Total net cash flow: **{ts.get('total_net_cash_flow', 'n/a'):,.0f}**")
    lines.append(f"- Daily mean: {ts.get('daily_mean', 'n/a'):,.0f} | Daily std dev: {ts.get('daily_std', 'n/a'):,.0f}")
    lines.append(f"- ADF stationarity p-value: **{ts.get('adf_pvalue', 'n/a')}** "
                  f"({'stationary' if ts.get('is_stationary') else 'non-stationary — differencing likely needed'})")
    lines.append(f"- Correlation, all-cleared vs payment-only target variant: "
                  f"**{ts.get('all_vs_payment_only_correlation', 'n/a')}**")

    lines.append("\n## 5. Customer Analysis")
    lines.append(f"- Unique customers: **{cust.get('n_unique_customers', 'n/a'):,}**")
    lines.append(f"- Customers driving 80% of value: **{cust.get('n_customers_driving_80pct_value', 'n/a'):,}** "
                  f"({cust.get('pct_of_base_driving_80pct_value', 'n/a')}% of customer base)")
    if "behavior_segments" in cust:
        lines.append("- Behavior segment mix (% of total amount):")
        for seg, pct in cust["behavior_segments"].items():
            lines.append(f"  - {seg}: {pct}%")

    lines.append("\n## 6. HBKID (Bank Account) Analysis")
    lines.append(f"- Unique HBKIDs: **{hbk.get('n_unique_hbkids', 'n/a')}**")

    lines.append("\n## 7. Business_Area Analysis (0131 vs 0019)")
    if "stationarity_by_area" in ba:
        for area, res in ba["stationarity_by_area"].items():
            lines.append(f"- Business_Area {area}: ADF p-value {res['adf_pvalue']} "
                          f"({'stationary' if res['is_stationary'] else 'non-stationary'})")
    if "customer_overlap" in ba:
        lines.append(f"- Customer/Business_Area overlap counts: {ba['customer_overlap']}")

    lines.append("\n## 8. Outliers & Anomalies")
    lines.append(f"- Extreme-value transaction rows (>3×IQR): **{out.get('n_extreme_rows', 'n/a'):,}** "
                  f"({out.get('pct_extreme_rows', 'n/a')}% of all rows)")
    lines.append(f"- Anomalous days in target series (|z|>3): **{out.get('n_anomalous_days', 'n/a')}**")
    if out.get("anomalous_days"):
        lines.append("- Flagged dates:")
        for d in out["anomalous_days"][:15]:
            lines.append(f"  - {d['date']}: net flow {d['net_cash_flow']:,.0f} (z={d['z_score']})")

    return "\n".join(lines)


report_markdown = build_markdown_report(report)
print(report_markdown)

# COMMAND ----------

# Save the report alongside the notebook outputs — adjust path to your workspace/DBFS convention
REPORT_PATH = "/dbfs/FileStore/reports/cash_forecasting_eda_report.md"

try:
    dbutils.fs.mkdirs("dbfs:/FileStore/reports")
    with open(REPORT_PATH, "w") as f:
        f.write(report_markdown)
    print(f"Report saved to: {REPORT_PATH}")
except NameError:
    # dbutils not available in this environment (e.g. local testing) — write locally instead
    local_path = "cash_forecasting_eda_report.md"
    with open(local_path, "w") as f:
        f.write(report_markdown)
    print(f"dbutils not found — report saved locally to: {local_path}")

# COMMAND ----------

# Render the report as Markdown directly in the notebook output
displayHTML(f"<pre style='white-space: pre-wrap; font-family: -apple-system, sans-serif;'>{report_markdown}</pre>")
