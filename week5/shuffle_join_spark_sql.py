from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast

spark = SparkSession \
    .builder \
    .appName("Shuffle Join Demo") \
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
# 2. Perform the JOIN using raw SQL
# We omit the broadcast hint here to ensure a standard Shuffle Join occurs
join_df = spark.sql("""
    SELECT 
        l.*, s.* FROM large_table l
    JOIN small_table s ON l.id = s.id
""")

# Execute and trigger action
join_df.collect()
# End the timer
end = time.perf_counter()
print(f"Shuffle Spark SQL Join Time: {end - start:.4f} seconds")
# Use the same waiting mechanism as the original script for UI inspection
input("Waiting ...")

spark.stop()
