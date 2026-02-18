from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast

spark = SparkSession \
    .builder \
    .appName("Broadcast Join Demo") \
    .master("local[3]") \
    .config("spark.sql.shuffle.partitions", 3) \
    .config("spark.sql.adaptive.enabled", False) \
    .getOrCreate()

# Load the data
df_large = spark.read.json("large_data/")
df_small = spark.read.json("small_data/")

# --- Spark SQL Implementation ---

# 1. Register DataFrames as Temporary Views
df_large.createOrReplaceTempView("large_table")
df_small.createOrReplaceTempView("small_table")

import time

# Start the timer
start = time.perf_counter()

# 2. Use the BROADCAST hint in the SQL query
# This tells Spark to broadcast 'small_table' specifically
join_df = spark.sql("""
    SELECT /*+ BROADCAST(small_table) */ 
        l.*, s.* FROM large_table l
    JOIN small_table s ON l.id = s.id
""")

# Execute and view results
join_df.collect()
# End the timer
end = time.perf_counter()
print(f"Broadcast Spark SQL Join Time: {end - start:.4f} seconds")
input("Waiting ...")

spark.stop()