from pyspark.sql import SparkSession

# Use metastore
spark = SparkSession.builder \
    .appName("create-parquet-tables") \
    .enableHiveSupport() \
    .config("spark.driver.memory", "1g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

print("AQE enabled:", spark.conf.get("spark.sql.adaptive.enabled", "false"))
print("Coalesce enabled:", spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled", "false"))
print("Default Parallelism:", spark.sparkContext.defaultParallelism)
print("Number of partitions during shuffle:", spark.conf.get("spark.sql.shuffle.partitions"))
print("Size of a partition:", spark.conf.get("spark.sql.files.maxPartitionBytes", None))

# create and use database named demo
spark.sql("CREATE DATABASE IF NOT EXISTS demo")
spark.sql("USE demo")

# create a table named items (parquet)
spark.sql("""
CREATE TABLE IF NOT EXISTS items
USING parquet
AS
SELECT
  id,
  CAST(rand() * 1000 AS INT) AS price
FROM RANGE(30000000)
""")
items_df = spark.table("items")   # or spark.sql("SELECT * FROM items")
print("num partitions:", items_df.rdd.getNumPartitions())

spark.sql("""
CREATE TABLE IF NOT EXISTS sales
USING parquet
AS
SELECT
  CASE
    WHEN rand() < 0.8 THEN 100
    ELSE CAST(rand() * 30000000 AS INT)
  END AS item_id,
  CAST(rand() * 100 AS INT) AS quantity,
  DATE_ADD(current_date(), - CAST(rand() * 360 AS INT)) AS date
FROM RANGE(1000000000)
""")
sales_df = spark.table("sales")   # or spark.sql("SELECT * FROM sales")
print("num partitions:", sales_df.rdd.getNumPartitions())

print("Done. Tables created: demo.items, demo.sales")
