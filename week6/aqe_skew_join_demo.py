import os, time
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("AQE Skew Join Demo") \
    .enableHiveSupport() \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.warehouse.dir", os.path.abspath("spark-warehouse")) \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
spark.sql("USE demo")

def run_skew_bench(aqe_on):
    spark.conf.set("spark.sql.adaptive.enabled", str(aqe_on).lower())
    print(f"\n--- AQE Skew Join Enabled: {aqe_on} ---")
    start = time.perf_counter()
    df = spark.sql("""
        SELECT date, sum(quantity * price) AS total_sales
        FROM sales s JOIN items i ON s.item_id = i.id
        GROUP BY 1 ORDER BY 2 DESC
    """)
    df.collect()
    print(f"Execution Time: {time.perf_counter() - start:.4f}s")

run_skew_bench(True)
run_skew_bench(False)

input("\nCheck 'AQESkewedJoin' at http://localhost:4040. Press Enter to exit...")
spark.stop()