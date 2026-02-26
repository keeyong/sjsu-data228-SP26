import os
# Force Spark to bind to localhost
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("gen-sample-data-10M") \
    .enableHiveSupport() \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.warehouse.dir", os.path.abspath("spark-warehouse")) \
    .getOrCreate()

spark.sql("CREATE DATABASE IF NOT EXISTS demo")
spark.sql("USE demo")

# Scaled to 300,000 items
print("Generating 300K row Items table...")
spark.sql("""
CREATE TABLE IF NOT EXISTS items USING parquet AS
SELECT id, CAST(rand() * 1000 AS INT) AS price FROM RANGE(300000)
""")

# Scaled to 10,000,000 sales
print("Generating 10M row skewed Sales table...")
spark.sql("""
CREATE TABLE IF NOT EXISTS sales USING parquet AS
SELECT
  CASE 
    WHEN rand() < 0.8 THEN 100 
    ELSE CAST(rand() * 300000 AS INT) 
  END AS item_id,
  CAST(rand() * 100 AS INT) AS quantity,
  DATE_ADD(current_date(), - CAST(rand() * 360 AS INT)) AS date
FROM RANGE(10000000)
""")

print(f"Tables saved permanently in: {os.path.abspath('spark-warehouse')}")
input("Wait for UI inspection at http://localhost:4040. Press Enter to stop...")
spark.stop()