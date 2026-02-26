from pyspark.sql import SparkSession
# 파일이 저장된 실제 경로로 수정하세요
jar_path = "./spark_jars/graphframes-0.8.3-spark3.5-s_2.12.jar"
slf4j_path = "./spark_jars/slf4j-api-1.7.16.jar"

spark = SparkSession.builder \
    .appName("GraphFrames_Manual") \
    .config("spark.jars", f"{jar_path},{slf4j_path}") \
    .getOrCreate()

# 이제 에러 없이 임포트가 가능합니다
from graphframes import GraphFrame

# 1. 정점(Vertices) 생성: (id, property)
# GraphFrames에서는 컬럼명이 반드시 'id'여야 인식합니다.
v = spark.createDataFrame([
    ("1", "Alice"),
    ("2", "Bob"),
    ("3", "Carol")
], ["id", "name"])
v.show()

# 2. 간선(Edges) 생성: (src, dst, property)
# 컬럼명이 반드시 'src'와 'dst'여야 합니다.
e = spark.createDataFrame([
    ("1", "2", "follows"),
    ("2", "3", "follows")
], ["src", "dst", "relationship"])
e.show()
# 3. 그래프 구축
graph = GraphFrame(v, e)

# 결과 확인
graph.vertices.show()
graph.edges.show()
