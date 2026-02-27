import os, time
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("AQE Coalesce Demo") \
    .enableHiveSupport() \
    .config("spark.sql.warehouse.dir", os.path.abspath("spark-warehouse")) \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
spark.sql("USE demo")

def run_bench(is_enabled):
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", str(is_enabled).lower())
    print(f"\n--- Coalesce Enabled: {is_enabled} ---")
    start = time.perf_counter()
    df = spark.sql("SELECT date, sum(quantity) AS q FROM sales GROUP BY 1 ORDER BY 2 DESC")
    df.collect()
    end = time.perf_counter()
    print(f"Final Partitions: {df.rdd.getNumPartitions()}")
    print(f"Time: {end - start:.4f}s")

run_bench(True)
run_bench(False)

input("\nInspect http://localhost:4040. Press Enter to exit...")
spark.stop()