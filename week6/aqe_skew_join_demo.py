from pyspark.sql import SparkSession
import time

# Use metastore
# Set driver memory to 4G and executor memory to 8G
spark = SparkSession.builder \
    .appName("AQE Skew JOIN Demo") \
    .enableHiveSupport() \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# Print a few configuration parameters
print("AQE enabled:", spark.conf.get("spark.sql.adaptive.enabled"))
print("AQE skewJoin Partition Factor:", spark.conf.get("spark.sql.adaptive.skewJoin.skewedPartitionFactor"))
print("AQE skewJoin Partition Threshold:", spark.conf.get("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes"))

# Show error message only 
spark.sparkContext.setLogLevel("ERROR")

# Create and use database named demo
spark.sql("USE demo")

# Run with AQE on
start = time.time()
df = spark.sql("""
SELECT date, sum(quantity * price) AS total_sales
FROM sales s
JOIN items i ON s.item_id = i.id
GROUP BY 1
ORDER BY 2 DESC;
""")
print("num partitions:", df.rdd.getNumPartitions())
end = time.time()
print("Runtime:", end - start, "seconds")

# Run with AQE off
spark.conf.set("spark.sql.adaptive.enabled", "false")

start = time.time()
df = spark.sql("""
SELECT date, sum(quantity * price) AS total_sales
FROM sales s
JOIN items i ON s.item_id = i.id
GROUP BY 1
ORDER BY 2 DESC;
""")
print("num partitions:", df.rdd.getNumPartitions())
end = time.time()
print("Runtime:", end - start, "seconds")
