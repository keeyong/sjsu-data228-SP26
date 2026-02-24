from pyspark.sql import SparkSession
import time

# Use metastore
spark = SparkSession.builder \
    .appName("AQE JOIN Demo") \
    .enableHiveSupport() \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

print("AQE enabled:", spark.conf.get("spark.sql.adaptive.enabled"))
print("AQE broadcast join threshold:", spark.conf.get("spark.sql.adaptive.autoBroadcastJoinThreshold"))
print("auto broadcast join threshold:", spark.conf.get("spark.sql.autoBroadcastJoinThreshold"))
print("Number of partitions during shuffle:", spark.conf.get("spark.sql.shuffle.partitions"))
print("Size of a partition:", spark.conf.get("spark.sql.files.maxPartitionBytes"))

# Suppress INFO and WARN messages
spark.sparkContext.setLogLevel("ERROR")

# Create and use database named demo
spark.sql("USE demo")

# Run with AQE on
start = time.time()
df = spark.sql("""
SELECT date, sum(quantity * price) AS total_sales
FROM sales s
JOIN items i ON s.item_id = i.id
WHERE price < 10
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
WHERE price < 10
GROUP BY 1
ORDER BY 2 DESC;
""")
print("num partitions:", df.rdd.getNumPartitions())
end = time.time()
print("Runtime:", end - start, "seconds")
