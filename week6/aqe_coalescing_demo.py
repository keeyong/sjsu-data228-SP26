from pyspark.sql import SparkSession

# Use metastore
spark = SparkSession.builder \
    .appName("AQE Coalesce Demo") \
    .enableHiveSupport() \
    .config("spark.driver.memory", "1g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

print("AQE enabled:", spark.conf.get("spark.sql.adaptive.enabled"))
print("Coalesce enabled:", spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled"))
print("Default Parallelism:", spark.sparkContext.defaultParallelism)
print("Number of partitions during shuffle:", spark.conf.get("spark.sql.shuffle.partitions"))
print("Size of a partition:", spark.conf.get("spark.sql.files.maxPartitionBytes"))

# show ERROR messages only
spark.sparkContext.setLogLevel("ERROR")

# create and use database named demo
spark.sql("USE demo")

# Run with AQE coalesce Partition on
df = spark.sql("""
SELECT date, sum(quantity) AS q
FROM sales
GROUP BY 1
ORDER BY 2 DESC;
""")
print("num partitions with coalesce Partition on:", df.rdd.getNumPartitions())

# Run with AQE coalesce Partition off
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "false")

df = spark.sql("""
SELECT date, sum(quantity) AS q
FROM sales
GROUP BY 1
ORDER BY 2 DESC;
""")
print("num partitions coalesce Partition off:", df.rdd.getNumPartitions())
