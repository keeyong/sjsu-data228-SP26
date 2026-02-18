from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast

spark = SparkSession \
    .builder \
    .appName("Shuffle Join Demo") \
    .master("local[3]") \
    .config("spark.sql.shuffle.partitions", 3) \
    .config("spark.sql.adaptive.enabled", False) \
    .getOrCreate()

df_large = spark.read.json("large_data/")
df_small = spark.read.json("small_data/")

import time

# Start the timer
start = time.perf_counter()

join_expr = df_large.id == df_small.id
join_df = df_large.join(df_small, join_expr, "inner")

join_df.collect()
# End the timer and calculate duration
end = time.perf_counter()
print(f"Shuffle Join Execution Time: {end - start:.4f} seconds")
input("Waiting ...")

spark.stop()
