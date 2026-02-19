from pyspark.sql import SparkSession

# 파일이 저장된 실제 경로로 수정하세요
jar_path = "./spark_jars/graphframes-0.8.3-spark3.5-s_2.12.jar"
slf4j_path = "./spark_jars/slf4j-api-1.7.16.jar"

spark = SparkSession.builder \
    .appName("GraphFrames_PageRank") \
    .config("spark.jars", f"{jar_path},{slf4j_path}") \
    .getOrCreate()

from graphframes import GraphFrame
# -------------------
# Vertices DataFrame
# -------------------
vertices = spark.createDataFrame(
    [
        (1, "Home"),
        (2, "About"),
        (3, "Blog"),
        (4, "Products")
    ],
    ["id", "name"]   # GraphFrames requires column name 'id'
)

# -------------------
# Edges DataFrame
# -------------------
edges = spark.createDataFrame(
    [
        (1, 2, "link"),
        (1, 3, "link"),
        (2, 4, "link"),
        (3, 4, "link"),
        (1, 4, "link")
    ],
    ["src", "dst", "relationship"]  # GraphFrames requires 'src' and 'dst'
)

# -------------------
# GraphFrame
# -------------------
g = GraphFrame(vertices, edges)

# Optional: inspect
g.vertices.show()
g.edges.show()


# PageRank Execution
pr = g.pageRank(
    resetProbability=0.15,
    maxIter=10
)

# PageRank Resules (vertices)
pr.vertices.select("id", "name", "pagerank") \
           .orderBy("pagerank", ascending=False) \
           .show()
